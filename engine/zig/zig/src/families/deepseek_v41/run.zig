//! M2's call executor: block.zig's calls issued on our own buffers, nothing seeded. The buffer plan (buffers.zig) is
//! bound once: a persistent role ("s.") gets its own allocation, the window and layer roles one zeroed arena at
//! 256-byte offsets (counters start at zero, as Python allocates them). Extension calls go through dsv41_kernels'
//! bindings; Triton calls through the engine's AOT set by name (triton_call.zig: the variant Triton would pick).
//! State (pools, rings, carries) and the window's glue fill these buffers from outside (state.zig, forward.zig).

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const buffers = @import("buffers.zig");
const load = @import("load.zig");
const tc = @import("triton_call.zig");
const rp = dk.replay;
const profile = @import("profile.zig");

pub const Runner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    triton: *const cuda.aot.Set,
    weights: *const load.Weights,
    /// role -> device address
    addrs: std.StringHashMapUnmanaged(u64) = .empty,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    scratch: ?cuda.DeviceBuffer = null,
    /// the forward's glue steps (forward.zig: positions, exchanges over the Collective, Engram rows, kit roundings)
    glue: ?Glue = null,
    /// TF_DSV41_PROFILE: the eager decode windows' calls timed (profile.zig)
    prof: ?*profile.Profile = null,
    /// TF_DSV41_L2PF (l2pf.zig): the L2 prefetch launches on a side stream forked at their site and joined at the
    /// window's end, as l2pf.py; null: skipped (their tables unfilled)
    l2pf: ?*Side = null,
    /// rows of the window being issued (0: unknown): a window wider than `Side.rows` skips every prefetch site
    rows: u32 = 0,
    /// TF_DSV41_BRANCHES (branches.zig): the side stream of the calls the emitter tagged `side`, forked / joined at
    /// their marks (the ext launchers take no scratch: kernels_replay.call leaves it unused). null: every call on main
    br: ?*Side = null,
    /// TF_DSV41_MHC_DEFER (branches.zig): the stream of the mHC coefficient launches tagged `defer_side`, joined by the
    /// next mHC call (`defer_join`, mhc.await_coefs)
    dfr: ?*Side = null,
    /// TF_DSV41_DRAFT_OVERLAP (dspark_slots.zig): the event a row window records (an event record node in its graph)
    /// right after its last call that writes "w.taps", the drafter's ingests on their own stream wait on it; `armed`
    /// for the next window only (batch.zig sets it before the row window, eager or captured, and clears it after)
    taps_mark: ?TapsMark = null,

    pub const TapsMark = struct { ev: cuda.Event, armed: bool = false, recorded: u64 = 0 };

    /// l2pf.py's Prefetcher on the device: the side stream, the fork / join events, whether a fork is open.
    pub const Side = struct {
        stream: cuda.Stream,
        fork: cuda.Event,
        join: cuda.Event,
        /// TF_DSV41_L2PF_ROWS (16): wider windows skip every site
        rows: u32 = 16,
        open: bool = false,
        launches: u64 = 0,
        skipped_rows: u64 = 0,
        /// TF_DSV41_PF_OVERLAP_SITE (the branches' side only): a call's `mark` m records marks[m - 1] on this stream,
        /// a call's `wait` m makes the main stream wait for it
        marks: [max_marks]?cuda.Event = @splat(null),
    };

    pub const max_marks = 8;

    pub const Glue = struct {
        ctx: *anyopaque,
        run: *const fn (ctx: *anyopaque, r: *Runner, c: *const calls.Call) anyerror!void,
    };

    pub fn deinit(r: *Runner) void {
        for (r.owned.items) |*b| b.free();
        r.owned.deinit(r.gpa);
        if (r.scratch) |*s| s.free();
        var it = r.addrs.keyIterator();
        while (it.next()) |k| r.gpa.free(k.*);
        r.addrs.deinit(r.gpa);
    }

    /// A persistent role whose memory another owner holds (M5: the KV pool's family tensors and tables): bind leaves it.
    pub fn external(r: *Runner, role: []const u8, addr: u64) !void {
        // Triton specializes pointers on 16-byte alignment and the AOT sets hold the aligned variants (aot-needs keys
        // offsets from 16-aligned bases): a misaligned base fails here by name, not later as a missing variant
        if (addr % 16 != 0) {
            std.log.scoped(.dsv41).err("role {s} bound at 0x{x}: not 16-byte aligned", .{ role, addr });
            return error.MisalignedRole;
        }
        const g = try r.addrs.getOrPut(r.gpa, role);
        if (!g.found_existing) g.key_ptr.* = try r.gpa.dupe(u8, role);
        g.value_ptr.* = addr;
    }

    /// An external role's address gone (an on-demand role released): a later use fails as error.Unbound.
    pub fn forget(r: *Runner, role: []const u8) void {
        if (r.addrs.fetchRemove(role)) |kv| r.gpa.free(kv.key);
    }

    /// Allocates every role of `plan`: persistent roles one buffer each (but the external ones), the rest one zeroed arena.
    pub fn bind(r: *Runner, plan: *const buffers.Plan) !void {
        var arena_bytes: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent and r.addrs.contains(k)) continue;
            if (buffers.scopeOf(k) == .persistent) {
                const b = try cuda.DeviceBuffer.alloc(r.d, v);
                try b.fill8(0, null);
                try r.owned.append(r.gpa, b);
                try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), b.ptr);
            } else arena_bytes += std.mem.alignForward(u64, v, 256);
        }
        const arena = try cuda.DeviceBuffer.alloc(r.d, arena_bytes);
        try arena.fill8(0, null);
        try r.owned.append(r.gpa, arena);
        var at: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent) continue;
            try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), arena.ptr + at);
            at += std.mem.alignForward(u64, v, 256);
        }
        r.scratch = try cuda.DeviceBuffer.alloc(r.d, 64 << 20);
    }

    /// A role's device address (state.zig and the glue write through it).
    pub fn addressOf(r: *const Runner, role: []const u8) ?u64 {
        return r.addrs.get(role);
    }

    pub fn tensorAddr(r: *const Runner, t: calls.Tensor) !u64 {
        var numel: i64 = 1;
        for (t.shape) |n| numel *= n;
        if (numel == 0) return 0;
        const off: u64 = @intCast(t.offset);
        return switch (t.role) {
            .weight => |w| (r.weights.get(w) orelse return error.MissingWeight).ptr + off,
            .buf => |b| (r.addrs.get(b) orelse return error.Unbound) + off,
            .empty => error.EmptyRole,
        };
    }

    /// The first Triton call of `cs` the AOT set has no variant for (its function's name), null when every one has:
    /// a program's launches resolved without issuing them (a boot-time probe), every tensor taken 16-aligned as the
    /// runner's buffers are.
    pub fn missingTriton(r: *const Runner, cs: []const calls.Call) !?[]const u8 {
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        for (cs) |*c| if (c.triton) {
            var mine: std.ArrayList(cuda.aot.Spec) = .empty;
            for (r.triton.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try mine.append(a, v.spec);
            const sp = tc.split(a, c, mine.items, .{ .ctx = @constCast(r), .of = alignedAddr }) catch return c.name;
            _ = r.triton.find(c.name, sp.args, sp.consts) catch return c.name; // error.MissingTritonVariant
        };
        return null;
    }

    fn alignedAddr(_: *anyopaque, _: calls.Tensor) anyerror!u64 {
        return 1 << 20;
    }

    fn addrOf(ctx: *anyopaque, t: calls.Tensor) anyerror!u64 {
        const r: *const Runner = @ptrCast(@alignCast(ctx));
        return r.tensorAddr(t);
    }

    fn extArg(r: *const Runner, a: std.mem.Allocator, x: calls.Arg) !rp.Arg {
        return switch (x) {
            .i => |v| .{ .int = v },
            .f => |v| .{ .float = v },
            .b => |v| .{ .boolean = v },
            .none => .none,
            .list => |items| blk: {
                const out = try a.alloc(rp.Arg, items.len);
                for (items, out) |y, *z| z.* = try r.extArg(a, y);
                break :blk .{ .list = out };
            },
            .t, .opaque_table => |t| .{ .tensor = .{ .ptr = try r.tensorAddr(t), .dtype = std.meta.stringToEnum(rp.DType, @tagName(t.dt)).?, .shape = t.shape, .stride = t.stride } },
        };
    }

    /// Issues one call on the runner's stream. The L2 prefetchers (they read weights into L2 and write nothing) go to
    /// the side stream, forked here so they start after every launch before them (l2pf.py `mark`); without
    /// TF_DSV41_L2PF they are skipped.
    pub fn issue(r: *Runner, c: *const calls.Call) !void {
        if (r.br) |b| {
            if (c.join) try r.joinBranch();
            if (c.wait > 0) {
                // the main stream after the side's work up to mark `wait` only (the fork stays open)
                if (!b.open) return error.BranchNotForked;
                try r.stream.wait(b.marks[c.wait - 1] orelse return error.NoBranchMark);
            }
            if (c.fork) {
                try b.fork.record(r.stream);
                try b.stream.wait(b.fork);
                b.open = true;
                b.launches += 1;
            }
            if (c.side) {
                if (!b.open) return error.BranchNotForked;
                // the call and anything it runs (a glue step's launches) on the side stream
                const main = r.stream;
                r.stream = b.stream;
                defer r.stream = main;
                try r.issueOn(b.stream, c, r.scratch);
                if (c.mark > 0) try (b.marks[c.mark - 1] orelse return error.NoBranchMark).record(b.stream);
                return;
            }
        }
        if (r.dfr) |d| {
            if (c.defer_join) try r.joinDeferred();
            if (c.defer_side) {
                try d.fork.record(r.stream);
                try d.stream.wait(d.fork);
                d.open = true;
                d.launches += 1;
                const main = r.stream;
                r.stream = d.stream;
                defer r.stream = main;
                return r.issueOn(d.stream, c, r.scratch);
            }
        }
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) {
            const p = r.l2pf orelse return;
            if (r.rows > p.rows) {
                p.skipped_rows += 1;
                return;
            }
            try p.fork.record(r.stream);
            try p.stream.wait(p.fork);
            p.open = true;
            p.launches += 1;
            return r.issueOn(p.stream, c, r.scratch);
        }
        return r.issueOn(r.stream, c, r.scratch);
    }

    /// The main stream waits for the deferred coefficients' stream (mhc.await_coefs).
    fn joinDeferred(r: *Runner) !void {
        const d = r.dfr orelse return;
        if (!d.open) return;
        d.open = false;
        try d.join.record(d.stream);
        try r.stream.wait(d.join);
    }

    /// The main stream waits for the branches' side stream (its open fork, if any).
    fn joinBranch(r: *Runner) !void {
        const b = r.br orelse return;
        if (!b.open) return;
        b.open = false;
        try b.join.record(b.stream);
        try r.stream.wait(b.join);
    }

    /// The main stream waits for the side stream's prefetches (l2pf.py `join`): every window joins before it returns,
    /// so a captured graph holds no open fork.
    pub fn join(r: *Runner) !void {
        try r.joinBranch();
        try r.joinDeferred();
        const p = r.l2pf orelse return;
        if (!p.open) return;
        p.open = false;
        try p.join.record(p.stream);
        try r.stream.wait(p.join);
    }

    fn dropForks(r: *Runner) void {
        if (r.l2pf) |p| p.open = false;
        if (r.br) |b| b.open = false;
        if (r.dfr) |d| d.open = false;
    }

    fn issueOn(r: *Runner, stream: cuda.Stream, c: *const calls.Call, scratch_buf: ?cuda.DeviceBuffer) !void {
        if (c.glue) {
            const g = r.glue orelse return error.NoGlue;
            return g.run(g.ctx, r, c);
        }
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        if (c.triton) {
            var mine: std.ArrayList(cuda.aot.Spec) = .empty;
            for (r.triton.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try mine.append(a, v.spec);
            const sp = try tc.split(a, c, mine.items, .{ .ctx = @constCast(r), .of = addrOf });
            const g = c.grid;
            return r.triton.run(stream, c.name, .{ @intCast(g[0]), @intCast(g[1]), @intCast(g[2]) }, sp.args, sp.consts);
        }
        const dot = std.mem.lastIndexOfScalar(u8, c.name, '.') orelse return error.BadName;
        const args = try a.alloc(rp.Arg, c.args.len);
        for (c.args, args) |x, *y| y.* = try r.extArg(a, x.arg);
        var scratch: rp.Scratch = .{ .ptr = scratch_buf.?.ptr, .len = scratch_buf.?.len };
        if (!try rp.call(r.kernels, stream, c.name[0..dot], c.name[dot + 1 ..], args, &.{}, &scratch)) return error.NoBinding;
    }

    /// Issues a window's calls in order. TF_DSV41_SYNC_EACH=1 (diagnosis): the stream synced after every call, and a
    /// failing call named with its index and row count (an asynchronous fault otherwise surfaces windows later).
    pub fn window(r: *Runner, cs: []const calls.Call) !void {
        if (syncEach()) return r.windowChecked(cs);
        errdefer r.dropForks(); // a failed window (or capture) leaves no fork for the next one to join
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            try r.issue(c);
            if (i == mark) try r.recordMark(c);
        }
        try r.join();
    }

    /// Issues `cs` on `stream` instead of the runner's (TF_DSV41_DRAFT_OVERLAP: the drafter's ingests beside a row
    /// window): a drafter program has no branch, deferral or prefetch marks, and no glue step; the runner's stream is
    /// this one while they issue. Nothing joins: the caller orders `stream` with its own events.
    pub fn windowOn(r: *Runner, stream: cuda.Stream, cs: []const calls.Call) !void {
        const main = r.stream;
        r.stream = stream;
        defer r.stream = main;
        for (cs) |*c| {
            if (c.side or c.fork or c.join or c.defer_side or c.defer_join or c.glue or std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) return error.NotASideProgram;
            try r.issueOn(stream, c, r.scratch);
        }
    }

    /// The index of the call after which an armed window records the taps mark (its last "w.taps" writer); null: not
    /// armed, or no call writes taps. The arming lasts until its owner clears it (a capture that falls back to an
    /// eager issue records again).
    fn markAt(r: *Runner, cs: []const calls.Call) ?usize {
        const m = r.taps_mark orelse return null;
        if (!m.armed) return null;
        return lastTaps(cs);
    }

    /// The mark after call `c`, on the stream `c` ran on (a branch's side stream when the emitter put it there). In a
    /// capture it is an external record node (each launch of the graph records it at that point); an eager window (a
    /// capture that fell back, or a key the row graphs had no room for under TF_DSV41_GRAPH_FLOOR_GIB) records it as a
    /// plain record at the same point: CU_EVENT_RECORD_EXTERNAL is invalid outside a capture (cuEventRecordWithFlags:
    /// CUDA_ERROR_ILLEGAL_STATE, Spark P4 2026-10-09: the first eager row window with TF_DSV41_DRAFT_OVERLAP=1).
    fn recordMark(r: *Runner, c: *const calls.Call) !void {
        const m = &r.taps_mark.?;
        const s = if (c.side and r.br != null) r.br.?.stream else if (c.defer_side and r.dfr != null) r.dfr.?.stream else r.stream;
        if (try cuda.graph.captureStatus(s) != .none) try m.ev.recordExternal(s) else try m.ev.record(s);
        m.recorded += 1;
    }

    fn windowChecked(r: *Runner, cs: []const calls.Call) !void {
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            r.issue(c) catch |e| {
                std.log.err("sync-each: call {d}/{d} {s} failed to issue ({t})", .{ i, cs.len, c.name, e });
                return e;
            };
            if (i == mark) try r.recordMark(c);
            r.stream.synchronize() catch |e| {
                std.log.err("sync-each: call {d}/{d} {s} faulted ({t})", .{ i, cs.len, c.name, e });
                return e;
            };
        }
        try r.join();
    }

    /// An eager decode window, each call timed when TF_DSV41_PROFILE is set (never inside a graph capture).
    pub fn timedWindow(r: *Runner, cs: []const calls.Call) !void {
        const p = r.prof orelse return r.window(cs);
        if (syncEach()) return r.windowChecked(cs);
        p.begin();
        errdefer r.dropForks();
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            try p.before(r.stream, c);
            try r.issue(c);
            if (i == mark) try r.recordMark(c);
            try p.after(r.stream);
        }
        try r.join();
        try p.end(r.stream);
    }
};

/// The last call of `cs` with a "w.taps" tensor argument (the boundary that writes the last DSpark target's taps);
/// null when none does.
pub fn lastTaps(cs: []const calls.Call) ?usize {
    var at: ?usize = null;
    for (cs, 0..) |c, i| for (c.args) |x| switch (x.arg) {
        .t => |t| if (t.role == .buf and std.mem.eql(u8, t.role.buf, "w.taps")) {
            at = i;
        },
        else => {},
    };
    return at;
}

var sync_each: ?bool = null;
fn syncEach() bool {
    if (sync_each) |v| return v;
    const v = if (std.c.getenv("TF_DSV41_SYNC_EACH")) |x| std.mem.eql(u8, std.mem.span(x), "1") else false;
    sync_each = v;
    return v;
}
