//! The forward's own prefill: a prompt in segments of `opts.prefill_rows` (2,048, Python's pchunk) through
//! block_prefill.zig's program, run eagerly by run.zig with this file's glue handler, each segment then committed as a
//! keep of all its rows. Python's Forward.prompt: the head (and kit_logits) only on the last segment.
//!
//! - A segment of at most 16 rows (the prompt's ragged tail, or a whole short prompt) is block_prefill's short segment:
//!   Python's mix of prefill projections and the decode kernels' row paths, committed the same way.
//! - The commit (keep's equivalent, Python's Forward.keep(slot, n - 1)): each ratio-2 KV source's carry is its fp32
//!   projection's last row ("w.L<i>.proj", the f32 of the bf16 rows, so the same bits as decode's carry kernel), the
//!   Engram tail is (tail + the segment's ids)[-(max_ngram - 1):], the position moves by the rows, nothing pending.
//!   The rings, pools and index keys the segment's kernels already wrote.
//! - After a prefill the last step's logits are in "w.logits", `Forward.last_rows` rows of them (the last segment's):
//!   greedy takes `out[0..last_rows]` and the caller reads the last row.
//! - Run once before the first segment: the persistent tables no loader form builds, the prefill GEMM's Hadamard
//!   ("s.pf.had", exl3.prefill's bf16 +-1) and x3gm's pointer tables ("s.L<i>.gm.t{g,u,d}": the routed experts'
//!   trellis addresses, the prefix of forms' "s.L<i>.ex.tp_*" before the shared expert's), and the exchange / Engram
//!   buffers grown to a segment (so a graphed decode window captured after it keeps its addresses).
//! - Long prompts (longpf.zig): past TF_DSV41_INDEX_STREAM_MIN visible keys the full-mode layers select with the
//!   stream top-k (glue "stream_merge"); under split KV a union past the cap runs its attentions in row blocks
//!   (longpf.window).
//! - CED replay (TF_DSV41_PREFILL=replay, ced.zig): each segment is the encoder pass (block_prefill.emitReplay
//!   `.encoder`, no head) whose glue "ced_stash" keeps its last rows entering the decoder in the ring; `finish` then
//!   runs the decoder over the prompt's tail from the ring (glue "ced_load", windows from R0) and commits nothing.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const vision_rows = @import("vision_rows.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const ced = @import("ced.zig");
const graphs = @import("graphs.zig");
const prod_knobs = @import("prod_knobs.zig");
const Forward = fwd.Forward;

const log = std.log.scoped(.dsv41);

pub const Error = error{ NoExpertTable, NoK2Table, BadGlue, StashGap };

/// The prefill's own state on the forward (forward.zig: `prefill_state`).
pub const State = struct {
    /// the persistent tables and grown buffers are in place (ready())
    ready: bool = false,
    /// the prompt's own copy (a caller's slice may alias the window's ids)
    ids: std.ArrayList(u32) = .empty,
    /// gm_plan's width-masked picks (int32 [P])
    masked: ?cuda.DeviceBuffer = null,
    /// the segment's layers and the MoE blocks seen: gm_ticket starts each layer's MoE, gm_plan reads its K2 table
    layers: []const u32 = &.{},
    moe: usize = 0,
    layer: u32 = 0,
    /// TF_DSV41_PREFETCH_AHEAD (prod_knobs.zig): the next segment's Engram reads issued before this one runs
    ahead: bool = false,
    /// long prompts' buffers (longpf.zig)
    long: longpf.State = .{},
    /// TF_DSV41_PF_TBO: micro-batch B's own long-prompt buffers (its stream top-k merge grows its own), and the pair in
    /// flight (the glue runs each micro-batch's steps with its own ids, window, Engram tail and MoE count)
    long_b: longpf.State = .{},
    tbo: ?*Pair = null,
    /// TF_DSV41_PF_4K (prod_knobs.apply): 4,096-row segments in the prefill's own workspace (`K4`); the forward's
    /// options, buffer plan and arena stay the 2,048-row ones, so decode windows and the drafter are untouched
    k4: bool = false,
    k4ws: K4 = .{},
    /// the current prompt call may use the workspace (4K segments, TBO pairs): every rank's MemAvailable leaves the
    /// workspace above the hard floor (wsAgreed, one collective a call); false: this call runs 2,048-row segments, the
    /// same bits (segmentation-invariant), never a failure. True until a call decides (the planner reads it at boot)
    ws_ok: bool = true,
    ws_fallbacks: u64 = 0,
    agreement: ?graphs.Agreement = null,
    /// the next prompt piece's ids (Forward.prefillAhead, op 60: the round planner's next piece of this prompt, as
    /// Python's rounds._prefetch(ahead=prefetch.next_pieces)): this prompt call's last segment reads their Engram rows
    /// ahead, as a segment inside one call reads the next segment's (TF_DSV41_PREFETCH_AHEAD). Cleared by the call
    next: std.ArrayList(u32) = .empty,
    /// TF_DSV41_PREFILL (prod_knobs.apply): `.replay` = CED (ced.zig), segments are encoder passes and `finish` runs
    /// the decoder; `.full`: every layer over every row
    mode: prod_knobs.PrefillMode = .full,
    /// the decoder replay's ids (the glue reads them through Forward.ids while it runs)
    tail_ids: [ced.keep]u32 = undefined,
    /// what "w.taps" holds after the prompt: positions [taps_at, taps_at + taps_rows) at rows 0 .. (the last full
    /// segment's, or the decoder replay's; an encoder segment writes none): the DSpark hand-off (ced.handoff)
    taps_at: u64 = 0,
    taps_rows: u32 = 0,

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        s.long.deinit(gpa);
        if (s.masked) |*b| b.free();
        s.ids.deinit(gpa);
        s.next.deinit(gpa);
        if (s.agreement) |*x| x.deinit();
        var it = s.k4ws.keep.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        s.k4ws.keep.deinit(gpa);
    }
};

/// Adds the prefill program's roles at `rows` (the largest segment) to `p`: at start 0 and at the last start its
/// emitter takes (the indexer's scores grow with the keys), a 17-row segment's (the small-segment paths: the Triton
/// mHC site, the out-of-place boundary streams) and a 16-row short segment's (the decode kernels' roles).
pub fn plan(f: *Forward, p: *buffers.Plan, rows: u32) !void {
    if (rows <= 16) return error.BadWindow;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const layers = try f.backbone(la.allocator());
    try planPrograms(f, p, rows, layers);
    if (f.prefill_state.k4 or (f.prefill_state.mode == .replay and f.opts.pf_tbo)) {
        const ws = &f.prefill_state.k4ws;
        if (ws.plan == null) {
            ws.plan = .{ .a = f.gpa };
            try planWorkspace(f, &ws.plan.?, rows, layers);
            if (keepsShared(f)) {
                const before = planBytes(&ws.plan.?);
                try k4Trim(f.gpa, &ws.plan.?, p, &ws.keep);
                log.info("prefill 4K: rank {d}: {d} roles shared with the forward (TF_DSV41_PF_WS_SHARE): workspace {d} -> {d} MiB", .{ f.comm.rank(), ws.keep.count(), before >> 20, planBytes(&ws.plan.?) >> 20 });
            }
        }
    }
}

/// The forward's prefill programs at `rows` (`plan` without the workspace).
fn planPrograms(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const sizes = [_]u32{ rows, 17, 16 };
    for (sizes[if (rows > 17) 0 else 1..]) |n| {
        const top = try lastStart(f, layers, n, .whole);
        for ([_]i64{ 0, top }) |start| {
            var ea = std.heap.ArenaAllocator.init(f.gpa);
            defer ea.deinit();
            try p.add(try block_prefill.emitPrefill(ea.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), layers, n, start, true));
        }
    }
    if (f.prefill_state.mode == .replay) try planReplay(f, p, rows, layers);
    // the multi-segment runs (TF_DSV41_PIECE_RUNS / _REPLAY_RUNS): their own roles and their rows' at `rows`
    var ra = std.heap.ArenaAllocator.init(f.gpa);
    defer ra.deinit();
    const replay = f.prefill_state.mode == .replay;
    const runs = try block_prefill.planRuns(ra.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), if (replay) try encoderLayers(f, layers) else layers, if (replay) try decoderLayers(f, layers) else null, rows, if (replay) .encoder else .whole);
    for (runs.items) |cs| try p.add(cs);
}

/// TF_DSV41_PF_WS_SHARE=1 (default 0) with TF_DSV41_PF_4K alone (no TBO pair, whose micro-batches run at once): a 4K
/// segment runs by itself, so a role its
/// program names that the forward's plan already holds at least as large stays the forward's (`K4.keep`, at boot)
/// instead of a second copy in the workspace (the index stream / keys budget, the pf.w scratch: ~0.44 GiB). Each
/// role is its own allocation or arena offset (buffers.zig), and the 2K segments use those roles the same way
/// between the same programs, so nothing changes but where the 4K segment's bytes live.
fn keepsShared(f: *const Forward) bool {
    return f.prefill_state.k4 and !f.opts.pf_tbo and envOn("TF_DSV41_PF_WS_SHARE");
}

fn envOn(name: [:0]const u8) bool {
    const v = std.mem.span(std.c.getenv(name) orelse return false);
    return std.mem.eql(u8, v, "1");
}

/// Moves the workspace roles (`wp`, in the namespace) that `forward` holds at least as large into `keep` (their bare
/// names): the 4K program then names the forward's role, and the workspace no longer sizes it.
pub fn k4Trim(gpa: std.mem.Allocator, wp: *buffers.Plan, forward: *const buffers.Plan, keep: *block_prefill.Keep) !void {
    var i: usize = 0;
    while (i < wp.sizes.count()) {
        const k = wp.sizes.keys()[i];
        const v = wp.sizes.values()[i];
        const at = std.mem.indexOf(u8, k, K4.tag) orelse {
            i += 1;
            continue;
        };
        const bare = try std.mem.concat(gpa, u8, &.{ k[0..at], k[at + K4.tag.len ..] });
        const have = forward.sizes.get(bare) orelse 0;
        if (have < v or keep.contains(bare)) {
            gpa.free(bare);
            if (have >= v) wp.sizes.orderedRemoveAt(i) else i += 1;
            continue;
        }
        try keep.put(gpa, bare, {});
        wp.sizes.orderedRemoveAt(i);
    }
}

/// The workspace's programs (nothing of them in the forward's plan): TF_DSV41_PF_4K's 4,096-row encoder segment in
/// its namespace, and TF_DSV41_PF_TBO's pair of whole segments (wsPair; 4,096-row micro-batches under 4K), each at
/// 0 and at its last start.
fn planWorkspace(f: *Forward, wp: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const replay = f.prefill_state.mode == .replay;
    if (f.prefill_state.k4) {
        const enc = try encoderLayers(f, layers);
        const o4 = k4Options(vision_rows.planOptions(f));
        const top4 = try lastStartWith(f, o4, enc, K4.rows, .encoder);
        for ([_]i64{ 0, top4 }) |start| {
            var ka = std.heap.ArenaAllocator.init(f.gpa);
            defer ka.deinit();
            const cs = try block_prefill.emitReplay(ka.allocator(), f.cfg, f.widths, o4, enc, K4.rows, start, .encoder);
            try wp.add(try block_prefill.ownCalls(ka.allocator(), cs, K4.tag));
        }
    }
    if (replay and f.opts.pf_tbo and rows >= 2 * block_prefill.tbo_min) {
        const enc = try encoderLayers(f, layers);
        const k4 = f.prefill_state.k4;
        const op = if (k4) k4Options(vision_rows.planOptions(f)) else vision_rows.planOptions(f);
        const seg: u32 = if (k4) K4.rows else rows;
        const top = @max(0, try lastStartWith(f, op, enc, seg, .encoder) - seg); // B at the last start a segment takes
        for ([_]i64{ 0, top }) |start| {
            var ta = std.heap.ArenaAllocator.init(f.gpa);
            defer ta.deinit();
            const cs = try block_prefill.emitTbo(ta.allocator(), f.cfg, f.widths, op, enc, seg, seg, start);
            try wp.add(try wsPair(ta.allocator(), cs, k4));
        }
    }
}

/// The bytes the workspace allocates at a long prompt (its roles in the namespace, 256-aligned as k4Acquire lays them
/// out); 0 when neither TF_DSV41_PF_4K nor TF_DSV41_PF_TBO is on. Sized before the KV pool (kv_state.bootCheck prices it).
pub fn workspaceBytes(f: *Forward, rows: u32) !u64 {
    if (!(f.prefill_state.k4 or (f.prefill_state.mode == .replay and f.opts.pf_tbo))) return 0;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    var wp: buffers.Plan = .{ .a = la.allocator() };
    const layers = try f.backbone(la.allocator());
    try planWorkspace(f, &wp, rows, layers);
    if (keepsShared(f)) {
        // the same trim `plan` makes: the roles the forward's prefill programs hold already
        var fp: buffers.Plan = .{ .a = la.allocator() };
        try planPrograms(f, &fp, rows, layers);
        var keep: block_prefill.Keep = .empty;
        try k4Trim(la.allocator(), &wp, &fp, &keep);
    }
    return planBytes(&wp);
}

fn planBytes(wp: *const buffers.Plan) u64 {
    var total: u64 = 0;
    for (wp.sizes.keys(), wp.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, K4.tag) == null) continue;
        total += std.mem.alignForward(u64, v, 256);
    }
    return total;
}

/// CED replay's programs (ced.zig): an encoder segment (the whole segment's roles up to the decoder's first layer, and
/// the stash ring), and the decoder pass at the row counts where its paths change (a short segment up to 16 rows, the
/// selection's attn_cuda top-k up to 64, the materialised selection past it, the replay's 127), at 0 and its last start.
fn planReplay(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const enc = try encoderLayers(f, layers);
    const dec = try decoderLayers(f, layers);
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    try p.add(try block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), enc, rows, 0, .encoder));
    for ([_]u32{ 16, 17, 32, 33, 64, 65, ced.keep }) |m| {
        if (m > rows) continue;
        const top = try lastStart(f, dec, m, .decoder);
        for ([_]i64{ 0, top }) |start| {
            var da = std.heap.ArenaAllocator.init(f.gpa);
            defer da.deinit();
            try p.add(try block_prefill.emitReplay(da.allocator(), f.cfg, f.widths, vision_rows.planOptions(f), dec, m, start, .decoder));
        }
    }
}

/// The encoder pass's layers: the forward's up to the decoder's first (ced.decoderStart), inclusive.
fn encoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[0 .. i + 1];
}

/// The decoder pass's layers: the forward's from the decoder's first on.
fn decoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[i..];
}

/// Whether an `n`-row segment (`part` of it) at `start` is on the prefill path (block_prefill's emitter takes it).
fn emits(f: *Forward, o: block.Options, layers: []const u32, n: u32, start: i64, part: block_prefill.Part) !bool {
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    _ = block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, o, layers, n, start, part) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// The last start an `n`-row segment emits at (the stream top-k's threshold moves with the segment's end). The decoder
/// pass ends a row before the prompt's last token: its last start is one lower.
fn lastStart(f: *Forward, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    return lastStartWith(f, f.opts, layers, n, part);
}

/// `lastStart` with options `o` (TF_DSV41_PF_4K's segments: k4Options).
fn lastStartWith(f: *Forward, o: block.Options, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    var hi: i64 = o.limit - n - @intFromBool(part == .decoder);
    if (hi < 0) return error.BadWindow;
    if (try emits(f, o, layers, n, hi, part)) return hi;
    if (!try emits(f, o, layers, n, 0, part)) {
        log.warn("prefill: a {d}-row segment is not on the prefill path at any position (block_prefill: Unsupported)", .{n});
        return error.Unsupported;
    }
    var lo: i64 = 0;
    while (hi - lo > 1) {
        const mid = lo + @divFloor(hi - lo, 2);
        if (try emits(f, o, layers, n, mid, part)) lo = mid else hi = mid;
    }
    log.info("prefill: {d}-row segments emit up to start {d}", .{ n, lo });
    return lo;
}

/// TF_DSV41_PF_4K's workspace: every role a 4,096-row segment's program names in its own namespace ("k4~":
/// block_prefill.ownRole; the slot's state and the constants are the forward's), sized by its own plan at boot and
/// allocated at a prompt's first 4K segment, released at the prompt's CED finish. Nothing of it is resident while no
/// long prompt prefills: the 4K path adds no byte to the forward's plan, decode's graphs keep their room.
pub const K4 = struct {
    pub const rows: u32 = prod_knobs.pf4k_rows;
    pub const tag = "k4~";
    plan: ?buffers.Plan = null,
    /// the bare roles a 4K segment shares with the forward (k4Trim under TF_DSV41_PF_WS_SHARE=1; else none)
    keep: block_prefill.Keep = .empty,
    buf: ?cuda.DeviceBuffer = null,
    bytes: u64 = 0,
    allocs: u64 = 0,
    releases: u64 = 0,
    /// the exchange's send buffer and Engram's gathered rows while a 4K segment runs (the glue grows them here): the
    /// forward's own, which decode graphs hold the addresses of, are never regrown by a 4K segment
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
};

/// The rows a prompt segment holds: 4,096 under TF_DSV41_PF_4K, else the forward's (the round planner's rows too).
pub fn segmentRows(f: *const Forward) usize {
    const st = &f.prefill_state;
    return if (st.k4 and st.ws_ok) K4.rows else @intCast(f.opts.prefill_rows);
}

/// Whether this prompt call may use the workspace, on every rank alike: held already (a later piece of the prompt),
/// or every rank's MemAvailable less the workspace clears the hard floor (one all-gathered flag, graphs.Agreement).
/// No: the call runs 2,048-row segments (4K and 2K give the same bits), logged; nothing fails.
fn wsAgreed(f: *Forward) !bool {
    const st = &f.prefill_state;
    const ws = &st.k4ws;
    if (ws.buf != null) return true;
    const pl = ws.plan orelse return false;
    const bytes = planBytes(&pl);
    const hard = wsFloorGiB();
    const avail = graphs.memAvailableGiB();
    const left = if (avail) |v| v - @as(f64, @floatFromInt(bytes)) / (1 << 30) else hard;
    if (st.agreement == null) st.agreement = try graphs.Agreement.init(f.runner.d, f.comm, f.runner.stream);
    const ok = try st.agreement.?.agree().all(st.agreement.?.agree().ctx, left >= hard);
    if (!ok) {
        st.ws_fallbacks += 1;
        if (st.ws_fallbacks <= 4 or st.ws_fallbacks % 64 == 0)
            log.warn("prefill workspace: {d} MiB does not fit on every rank (rank {d}: {d:.2} GiB MemAvailable leaves {d:.2}, floor {d:.1} GiB, TF_DSV41_PF_WS_FLOOR_GIB): this prompt piece runs 2,048-row segments (fallback {d})", .{ bytes >> 20, f.comm.rank(), avail orelse -1, left, hard, st.ws_fallbacks });
    }
    return ok;
}

/// The options a 4,096-row segment emits with: the forward's at 4,096 rows with x3gm one block a segment.
pub fn k4Options(o: block.Options) block.Options {
    var x = o;
    x.prefill_rows = K4.rows;
    x.pf4k = true;
    return x;
}

/// A TF_DSV41_PF_TBO pair's calls with their private roles in the workspace: B's ("b~", its own scratch and
/// arena) always, A's too when the pair's micro-batches are 4,096 rows (`all`: the forward's 2K roles cannot hold
/// them). A 2K pair's A keeps the forward's roles (sized for it). So a pair adds no byte to the forward's plan either.
pub fn wsPair(a: std.mem.Allocator, cs: []const calls.Call, all: bool) ![]calls.Call {
    if (all) return block_prefill.ownCalls(a, cs, K4.tag);
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *x| {
        x.* = c;
        const args = try a.alloc(calls.Named, c.args.len);
        for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try wsArg(a, y.arg) };
        x.args = args;
    }
    return out;
}

fn wsArg(a: std.mem.Allocator, x: calls.Arg) !calls.Arg {
    switch (x) {
        .t, .opaque_table => |t| {
            if (t.role != .buf or std.mem.indexOf(u8, t.role.buf, "b~") == null) return x;
            var u = t;
            u.role = .{ .buf = try block_prefill.ownRole(a, t.role.buf, K4.tag) };
            return if (x == .t) .{ .t = u } else .{ .opaque_table = u };
        },
        .list => |l| {
            const o = try a.alloc(calls.Arg, l.len);
            for (l, o) |y, *z| z.* = try wsArg(a, y);
            return .{ .list = o };
        },
        else => return x,
    }
}

/// The workspace's roles bound to one fresh zeroed allocation (the first 4K segment of a prompt).
fn k4Acquire(f: *Forward) !void {
    const ws = &f.prefill_state.k4ws;
    if (ws.buf != null) return;
    const pl = ws.plan orelse return error.Unbound;
    const total = planBytes(&pl);
    // the room was agreed first (wsAgreed): every rank fits the workspace above the hard floor
    const t0 = nowMs();
    const b = cuda.DeviceBuffer.alloc(f.runner.d, @max(total, 256)) catch |e| {
        log.err("prefill workspace: allocating {d} MiB failed ({t})", .{ total >> 20, e });
        return e;
    };
    errdefer {
        var x = b;
        x.free();
    }
    try b.fill8(0, null);
    var at: u64 = 0;
    for (pl.sizes.keys(), pl.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, K4.tag) == null) continue;
        try f.runner.external(k, b.ptr + at);
        at += std.mem.alignForward(u64, v, 256);
    }
    ws.buf = b;
    ws.bytes = total;
    ws.allocs += 1;
    // the allocation's cost a prompt (the GB10 maps host pages for it): outside the prefill profile's segments
    try f.runner.stream.synchronize();
    const ms = nowMs() - t0;
    if (ws.allocs <= 4 or ws.allocs % 64 == 0) log.info("prefill 4K: rank {d}: workspace {d} MiB allocated and zeroed in {d:.1} ms (prompt {d}), held while the prompt prefills", .{ f.comm.rank(), total >> 20, ms, ws.allocs });
}

fn nowMs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

/// TF_DSV41_PF_WS_FLOOR_GIB: the MemAvailable the workspace must leave on every rank, else the prompt piece runs
/// 2,048-row segments; default TF_DSV41_FLOOR_HARD_GIB (4, the boot check's).
fn wsFloorGiB() f64 {
    const v = std.c.getenv("TF_DSV41_PF_WS_FLOOR_GIB") orelse return hardFloorGiB();
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch hardFloorGiB();
}

/// TF_DSV41_FLOOR_HARD_GIB (4, the boot check's).
fn hardFloorGiB() f64 {
    const v = std.c.getenv("TF_DSV41_FLOOR_HARD_GIB") orelse return 4.0;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 4.0;
}

/// The workspace back (a prompt's CED finish: no 4K segment runs until the next prompt's first).
fn k4Release(f: *Forward) !void {
    const ws = &f.prefill_state.k4ws;
    var b = ws.buf orelse return;
    try f.runner.stream.synchronize();
    const before = graphs.memAvailableGiB();
    b.free();
    ws.buf = null;
    // the prefill -> decode handoff: what the row graphs' memory floor sees right after the free (every rank: rank 1
    // is the tighter node on the Sparks; the first prompts and every 64th after)
    ws.releases += 1;
    if (ws.releases <= 4 or ws.releases % 64 == 0)
        log.info("prefill 4K: rank {d}: workspace {d} MiB freed at the prompt's end (release {d}); MemAvailable {d:.2} -> {d:.2} GiB", .{ f.comm.rank(), ws.bytes >> 20, ws.releases, before orelse -1, graphs.memAvailableGiB() orelse -1 });
    if (ws.send) |*x| x.free();
    if (ws.gathered) |*x| x.free();
    ws.send = null;
    ws.gathered = null;
}

/// TF_DSV41_PF_TBO: one micro-batch's host side of a pair in flight (what `segment` sets on the forward for its one
/// segment): its ids, its window, the slot's Engram tail before its rows, its MoE calls seen.
pub const Micro = struct {
    ids: []const u32,
    pending: @TypeOf(@as(fwd.Slot, undefined).pending),
    tail: Tail,
    moe: usize = 0,
    layer: u32 = 0,
};

pub const Pair = struct { a: Micro, b: Micro };

/// The prompt `ids` from the slot's position (forward.prefill: the leader has sent it to the followers).
pub fn prompt(f: *Forward, ids_in: []const u32) !void {
    const st = &f.prefill_state;
    if (f.slot.pending != null) return error.BadWindow;
    if (ids_in.len == 0) return;
    if (f.slot.pos + ids_in.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
    if (ids_in.ptr != st.ids.items.ptr) {
        st.ids.clearRetainingCapacity();
        try st.ids.appendSlice(f.gpa, ids_in);
    }
    const ids = st.ids.items;
    st.taps_rows = 0;
    try ready(f);
    // TF_DSV41_PF_4K / _PF_TBO: the workspace for this call, decided on every rank alike (else 2,048-row segments)
    st.ws_ok = true;
    if (st.k4 or (st.mode == .replay and f.opts.pf_tbo)) st.ws_ok = try wsAgreed(f);
    const seg: usize = segmentRows(f);
    defer st.next.clearRetainingCapacity();
    var at: usize = 0;
    while (at < ids.len) {
        // TF_DSV41_PF_TBO: two segments at a time (the second at least tbo_min rows), else one
        if (pairOf(f, ids.len - at)) |pr| {
            const n = pr[0] + pr[1];
            if (st.ahead) ahead(f, ids, at, n, seg);
            if (try segmentPair(f, ids[at .. at + pr[0]], ids[at + pr[0] .. at + n])) {
                at += n;
                continue;
            }
        }
        const n = @min(seg, ids.len - at);
        if (st.ahead) ahead(f, ids, at, n, seg);
        try segment(f, ids[at .. at + n], at + n == ids.len and st.mode == .full);
        at += n;
    }
}

/// TF_DSV41_PF_TBO's split of the `left` prompt rows ahead: two whole segments, or what is left halved on the 16-row
/// grid; null when TBO is off, the prompt is in full mode, or a half would be under tbo_min.
fn pairOf(f: *const Forward, left: usize) ?[2]usize {
    if (!f.opts.pf_tbo or f.prefill_state.mode != .replay or !f.prefill_state.ws_ok) return null;
    return pairSplit(segmentRows(f), left);
}

/// `pairOf`'s split of `left` rows at segments of `seg` rows (null: one segment at a time).
pub fn pairSplit(seg: usize, left: usize) ?[2]usize {
    const min: usize = @intCast(block_prefill.tbo_min);
    if (left >= 2 * seg) return .{ seg, seg };
    if (left < 2 * min) return null;
    const na = @min(seg, (left / 2) / 16 * 16);
    if (left - na > seg or na < min) return null;
    return .{ na, left - na };
}

/// Two consecutive encoder segments as one two-stream program (block_prefill.emitTbo), committed once as the second
/// would leave the slot: B's last rows' carries, the Engram tail and position after both, the stash's ids. false: the
/// emitter refused the pair (the caller runs the segments one at a time).
fn segmentPair(f: *Forward, ia: []const u32, ib: []const u32) !bool {
    const st = &f.prefill_state;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try encoderLayers(f, try f.backbone(a));
    const start = f.slot.pos;
    const na: u32 = @intCast(ia.len);
    const nb: u32 = @intCast(ib.len);
    // under TF_DSV41_PF_4K a micro-batch past the forward's rows is a 4K one (both in the workspace: wsPair)
    const big = st.k4 and @max(na, nb) > f.opts.prefill_rows;
    const o0 = vision_rows.segmentOptions(f, ia.ptr[0 .. na + nb]);
    const o = if (big) k4Options(o0) else o0;
    const plain = block_prefill.emitTbo(a, f.cfg, f.widths, o, layers, na, nb, @intCast(start)) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    const cs = try wsPair(a, plain, big);
    try k4Acquire(f);
    if (f.kv) |kx| try kx.reserve(start + na + nb);
    // the micro-batches' host sides: B's Engram tail is the slot's after A's ids (tailOf, as A's commit leaves it)
    var pair: Pair = .{ .a = .{ .ids = ia, .pending = .{ .start = start, .n = na }, .tail = .{} }, .b = .{ .ids = ib, .pending = .{ .start = start + na, .n = nb }, .tail = .{} } };
    pair.a.tail.len = f.slot.tail_len;
    @memcpy(pair.a.tail.ids[0..pair.a.tail.len], f.slot.tail[0..f.slot.tail_len]);
    if (f.engram) |h| {
        var s2 = f.slot;
        tailOf(&s2, ia, h.max_ngram - 1);
        pair.b.tail.len = s2.tail_len;
        @memcpy(pair.b.tail.ids[0..s2.tail_len], s2.tail[0..s2.tail_len]);
    } else pair.b.tail = pair.a.tail;
    f.ids = ia;
    f.slot.pending = pair.a.pending;
    st.tbo = &pair;
    defer st.tbo = null;
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // the pair's send / gathered buffers are the workspace's while it runs (as a 4K segment's)
    std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
    std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    defer {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(na + nb), false);
    // the commit: both micro-batches' rows as one segment, the carries from B's last rows (its roles in the workspace)
    const both = ia.ptr[0 .. na + nb];
    f.ids = both;
    f.slot.pending = .{ .start = start, .n = na + nb };
    try commitRoles(f, nb, 0, K4.tag ++ "b~", na + nb);
    f.slot.ced.extend(start, both);
    f.last_rows = 0;
    st.taps_at = start;
    st.taps_rows = 0;
    return true;
}

/// prefetch.py's one segment ahead: the reads of the segment after `ids[at .. at + n]` (its `next` rows, with the
/// lookback they will see: the slot's tail and this segment's ids) start before this segment runs, so they are in
/// flight while it does; its glue then waits for them (engram_rows keeps an unread bulk batch). A wrong guess costs a
/// read, never a bit: the rows are the table's whatever the path.
/// The read-ahead of the rows after segment ids[at .. at + n]: the call's next segment, or past the call's last
/// segment the next piece's (`next`, op 60).
fn ahead(f: *Forward, ids: []const u32, at: usize, n: usize, seg: usize) void {
    const st = &f.prefill_state;
    if (at + n < ids.len) return readAhead(f, ids, at, n, ids[at + n .. at + n + @min(seg, ids.len - at - n)]);
    if (st.next.items.len > 0) readAhead(f, ids, at, n, st.next.items);
}

fn readAhead(f: *Forward, ids: []const u32, at: usize, n: usize, next: []const u32) void {
    const h = f.engram orelse return;
    if (!f.engram_prefetch) return;
    const src = (f.engramSource(h) catch return) orelse return;
    const keep: usize = h.max_ngram - 1;
    const b = at + n;
    var buf: [2 * eh.max_ngram]u32 = undefined;
    const lb: []const u32 = if (n >= keep) ids[b - keep .. b] else blk: {
        const old = @min(f.slot.tail_len, keep - n);
        @memcpy(buf[0..old], f.slot.tail[f.slot.tail_len - old .. f.slot.tail_len]);
        @memcpy(buf[old .. old + n], ids[at..b]);
        break :blk buf[0 .. old + n];
    };
    h.prefetch(f.gpa, src, next, lb, f.comm.rank(), f.comm.world());
}

/// One segment at the slot's position, run eagerly, then committed.
fn segment(f: *Forward, ids: []const u32, head: bool) !void {
    const st = &f.prefill_state;
    const n: u32 = @intCast(ids.len);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    // CED replay: the encoder pass (layers up to the decoder's first, its site and compressor, the stash; no head)
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const start = f.slot.pos;
    if (f.kv) |kx| try kx.reserve(start + n);
    // TF_DSV41_PF_4K: a segment past the forward's rows runs in the workspace (its own roles: block_prefill.ownCalls)
    const big = st.k4 and st.ws_ok and n > f.opts.prefill_rows;
    const o = if (big) k4Options(vision_rows.segmentOptions(f, ids)) else vision_rows.segmentOptions(f, ids);
    const emitted = if (replay) block_prefill.emitReplay(a, f.cfg, f.widths, o, layers, n, @intCast(start), .encoder) else block_prefill.emitPrefill(a, f.cfg, f.widths, o, layers, n, @intCast(start), head);
    const plain = emitted catch |e| {
        if (e == error.Unsupported) log.warn("prefill: the {d}-row segment at {d} is not on the prefill path (block_prefill: Unsupported)", .{ n, start });
        return e;
    };
    const cs = if (big) try block_prefill.ownCallsKeep(a, plain, K4.tag, &st.k4ws.keep) else plain;
    if (big) try k4Acquire(f);
    // the glue reads the ids (embedding, Engram) and the pending window (positions)
    f.ids = ids;
    f.slot.pending = .{ .start = start, .n = n };
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // TF_DSV41_PF_4K: the segment's send / gathered buffers are the workspace's while it runs
    if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    defer if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(n), false);
    if (big) try commitRoles(f, n, 0, K4.tag, n) else try commit(f, n);
    // replay.encode: the stash's ids follow its rows (a segment that does not continue it starts it over)
    if (replay) f.slot.ced.extend(start, ids);
    f.last_rows = if (replay) 0 else n;
    st.taps_at = start;
    st.taps_rows = if (replay) 0 else n;
}

/// One segment of a multi-segment run: a slot's prompt ids from its position `start`.
pub const MultiSeg = struct { slot: u32, start: u64, ids: []const u32 };

/// An Engram tail (a slot's last ids before the run).
pub const Tail = struct {
    ids: [eh.max_ngram]u32 = undefined,
    len: usize = 0,

    pub fn items(t: *const Tail) []const u32 {
        return t.ids[0..t.len];
    }
};

/// The run in flight (Forward.multi): its segments, each one's first row in the run and its slot's Engram tail at
/// the run's start, and every segment's ids in run order (the embedding's).
pub const Multi = struct { segs: []const MultiSeg, row0: []const usize, tails: []const Tail, ids: []const u32 };

/// Several slots' prompt segments in one run (block_prefill.emitMulti; Python's slots.prefill / replay.encode over
/// prefill_runs: GLM 0560's one expert pass over every prefilling slot). Each segment starts at its slot's position;
/// the rows' launches run over all segments, each segment's CSA2 steps on its own slot (glue pf_seg, Forward.pf_switch),
/// then each slot commits its segment as `segment` would (carries from its last row, Engram tail, position, CED's
/// stash ids). error.Unsupported: the caller runs the segments one at a time (a short segment, an image, split KV).
pub fn promptMulti(f: *Forward, segs: []const MultiSeg) !void {
    const st = &f.prefill_state;
    const sw = f.pf_switch orelse return error.Unsupported;
    if (segs.len == 0 or segs.len > 16) return error.BadWindow;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    const row0 = try a.alloc(usize, segs.len);
    const tails = try a.alloc(Tail, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans, row0, tails) |sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, sg.slot);
        if (f.slot.pending != null or f.slot.pos != sg.start or sg.ids.len == 0) return error.BadWindow;
        if (sg.start + sg.ids.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return error.Unsupported;
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole);
    try ready(f);
    // the pages of every segment's positions, before any launch (each on its slot's table)
    for (segs) |sg| {
        try sw.run(sw.ctx, sg.slot);
        if (f.kv) |kx| try kx.reserve(sg.start + sg.ids.len);
    }
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(all.items.len), false);
    // each slot keeps its segment (Python's keep(slot, n - 1) a segment, replay.encode's stash ids)
    for (segs, row0) |sg, r0| {
        try sw.run(sw.ctx, sg.slot);
        f.ids = sg.ids;
        f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
        try commitAt(f, @intCast(sg.ids.len), r0);
        if (replay) f.slot.ced.extend(sg.start, sg.ids);
    }
    f.last_rows = 0;
}

/// Whether a multi-segment run takes `segs` (promptMulti's refusals), checked before the leader sends it.
pub fn runnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or segs.len < 2 or segs.len > 16) return false;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = f.prefill_state.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// One slot's decoder replay in a batched run (finishMulti): the replayed tail's first position and rows, and its
/// first row in the run (its rows of "w.taps": the drafter's context, ced.handoff).
pub const Replayed = struct { slot: u32, at: u64, rows: u32, row0: u32 };

/// Whether a batched decoder replay takes `segs` (each a slot's prefilled tail from its R0), checked before the leader
/// sends it (emitMulti's refusals: a tail of 32 rows or fewer, more rows than a prefill segment, an image, split KV).
pub fn replayRunnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or f.prefill_state.mode != .replay or segs.len < 2 or segs.len > 16) return false;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, try decoderLayers(f, try f.backbone(a)), spans, .decoder) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// Several slots' CED decoder replays in one run (block_prefill.emitMulti `.decoder`; Python's rounds.py replays its
/// finals a slot at a time): each slot's prefilled tail [R0, end) from its stash, the rows' launches over every tail,
/// each tail's CSA2 steps on its slot. Nothing is committed: each slot stays at its prompt's last token, as `finish`
/// leaves it. `out[j]`: slot j's tail and its rows of "w.taps". A tail batched is its one-slot replay row for row
/// (emitMulti's row floor keeps every tail on the one-slot replay's kernel paths).
pub fn finishMulti(f: *Forward, slots: []const u32, out: []Replayed) !void {
    const st = &f.prefill_state;
    const sw = f.pf_switch orelse return error.Unsupported;
    if (st.mode != .replay or slots.len < 2 or slots.len > 16 or out.len < slots.len) return error.BadWindow;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const segs = try a.alloc(MultiSeg, slots.len);
    const spans = try a.alloc(block_prefill.Span, slots.len);
    const row0 = try a.alloc(usize, slots.len);
    const tails = try a.alloc(Tail, slots.len);
    var all: std.ArrayList(u32) = .empty;
    for (slots, segs, spans, row0, tails) |slot, *sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, slot);
        if (f.slot.pending != null) return error.BadWindow;
        const end = f.slot.pos;
        const tl = ced.tailOf(end);
        if (tl.m == 0) return error.BadWindow;
        if (!f.slot.ced.covers(tl.r0, end)) {
            log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ tl.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
            return error.StashGap;
        }
        const ids = try a.alloc(u32, tl.m);
        f.slot.ced.tail(tl.r0, end, ids);
        sg.* = .{ .slot = slot, .start = tl.r0, .ids = ids };
        sp.* = .{ .slot = slot, .start = @intCast(tl.r0), .n = @intCast(tl.m) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, ids);
    }
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, vision_rows.segmentOptions(f, all.items), layers, spans, .decoder);
    try ready(f);
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(m.ids.len), true);
    for (segs, row0, out[0..slots.len]) |sg, r0, *o| {
        try sw.run(sw.ctx, sg.slot);
        f.slot.pending = null;
        o.* = .{ .slot = sg.slot, .at = sg.start, .rows = @intCast(sg.ids.len), .row0 = @intCast(r0) };
    }
    f.last_rows = 0;
    try k4Release(f); // TF_DSV41_PF_4K: these prompts are prefilled
}

/// The glue's view of the whole run (Forward.multi): the first segment's slot, every row's ids, the run's rows pending.
fn enterRun(f: *Forward, m: *const Multi) !void {
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, m.segs[0].slot);
    f.ids = m.ids;
    f.slot.pending = .{ .start = m.segs[0].start, .n = @intCast(m.ids.len) };
}

/// glue pf_seg: segment j's slot current, its rows pending, its positions (`w.pos`, `w.pos64`, `w.lo`); -1: the run's
/// view again (enterRun).
fn segGlue(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const j = c.args[0].arg.i;
    if (j < 0) return enterRun(f, m);
    const sg = m.segs[@intCast(j)];
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, sg.slot);
    f.ids = sg.ids;
    f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
    const k = r.kernels.others(r.stream);
    const lo = try r.tensorAddr(c.args[3].arg.t);
    try k.positions(@intCast(sg.start), @intCast(sg.ids.len), try r.tensorAddr(c.args[1].arg.t), try r.tensorAddr(c.args[2].arg.t), lo);
    // a decoder replay: every row's window starts at the segment's first row (R0)
    if (c.args[4].arg.b) try r.d.check(r.d.api.cuMemsetD32Async(lo, @bitCast(@as(i32, @intCast(sg.start))), sg.ids.len, r.stream.handle), "cuMemsetD32Async");
}

/// glue mpositions: every segment's positions into the run's "w.mpos64" (rope over the run's rows); the window starts
/// and the start word land in the scratch the segments' pf_seg rewrites.
fn runPositions(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const k = r.kernels.others(r.stream);
    const mpos = try r.tensorAddr(c.args[0].arg.t);
    const pos = r.addressOf("w.pos") orelse return error.Unbound;
    const lo = r.addressOf("w.lo") orelse return error.Unbound;
    for (m.segs, m.row0) |sg, r0| try k.positions(@intCast(sg.start), @intCast(sg.ids.len), pos, mpos + 8 * r0, lo);
}

/// CED's decoder replay (replay.finish; Forward.finishPrompt on every rank): layers D .. over the prompt's prefilled
/// tail [R0, end) from the stash, end = the slot's position, every window from R0. Nothing is committed: the slot
/// stays at `end`, the decoder's SWA rings hold [R0, end), and the prompt's last token is the next window's row.
pub fn finish(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.mode != .replay) return;
    if (f.slot.pending != null) return error.BadWindow;
    const end = f.slot.pos;
    const t = ced.tailOf(end);
    if (t.m == 0) return;
    if (!f.slot.ced.covers(t.r0, end)) {
        log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ t.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
        return error.StashGap;
    }
    try ready(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const ids = st.tail_ids[0..t.m];
    f.slot.ced.tail(t.r0, end, ids);
    // the tail's image positions route with gate.bias_vl (replay._image); no decoder layer reads Engram
    const cs = try block_prefill.emitReplay(a, f.cfg, f.widths, vision_rows.segmentOptions(f, ids), layers, t.m, @intCast(t.r0), .decoder);
    f.ids = ids;
    f.slot.pending = .{ .start = t.r0, .n = t.m };
    defer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, t.r0, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(t.m), true);
    f.last_rows = 0;
    // TF_DSV41_PF_4K: the prompt is prefilled, its workspace goes back
    try k4Release(f);
    // the drafter's context rows: the tail's taps (replay.finish's, which slots.finish_prompt hands to `replayed`)
    st.taps_at = t.r0;
    st.taps_rows = t.m;
}

/// ced_stash (`store`): the segment's last min(n, ring) rows of each tensor (the streams, the normed input, the
/// coefficients) into the ring, position p at row p % ring; ced_load: the ring's rows of the pass's positions into the
/// tensors' rows 0 .. n.
fn stashCopy(f: *Forward, r: *run.Runner, c: *const calls.Call, store: bool) !void {
    const p = f.slot.pending orelse return error.BadGlue;
    if (f.pf_stash) |h| try h.run(h.ctx); // the ring holds this slot's rows (slots.zig swaps it lazily)
    const x = c.args;
    if (x.len != 2 * ced.roles.len) return error.BadGlue;
    const k: u64 = @min(p.n, ced.ring);
    const a0: u64 = p.start + p.n - k;
    var sp: [2]ced.Span = undefined;
    for (0..ced.roles.len) |j| {
        const t = x[j].arg.t;
        const rg = x[ced.roles.len + j].arg.t;
        const row = dim(t, 1) * buffers.dtSize(t.dt);
        if (dim(rg, 1) * buffers.dtSize(rg.dt) != row or ld(t) != dim(t, 1) or dim(rg, 0) != ced.ring) return error.BadGlue;
        const ta = try r.tensorAddr(t);
        const ra = try r.tensorAddr(rg);
        for (ced.spans(a0, a0 + k, &sp)) |s| {
            const tr = ta + (a0 - p.start + s.at) * row;
            const rr = ra + @as(u64, s.row) * row;
            if (store) try copy(r, rr, tr, s.n * row) else try copy(r, tr, rr, s.n * row);
        }
    }
}

/// Keep of all the pending segment's rows: the ratio-2 carries, the Engram tail, the position.
fn commit(f: *Forward, n: u32) !void {
    return commitAt(f, n, 0);
}

/// `commit` of a segment whose rows are rows [row0, row0 + n) of the run's projections.
fn commitAt(f: *Forward, n: u32, row0: usize) !void {
    return commitRoles(f, n, row0, "", n);
}

/// `commitAt` with the carries from rows of "w.<pre>L<i>.proj" (TF_DSV41_PF_TBO's B: "b~") whose last row is the
/// `n`th; the slot advances by `adv` rows (f.ids' first `adv`).
fn commitRoles(f: *Forward, n: u32, row0: usize, comptime pre: []const u8, adv: u32) !void {
    const p = f.slot.pending orelse return error.BadWindow;
    const r = f.runner;
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const row: usize = 4 * 2 * @as(usize, f.cfg.head_dim); // fp32 [kv | gate]
    for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
        // a role the 4K segment shares with the forward (K4.keep) is under its bare name
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "w." ++ pre ++ "L{d}.proj", .{L})) orelse r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.proj", .{L})) orelse return error.Unbound;
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
        try copy(r, dst, src + (row0 + n - 1) * row, row);
    };
    if (f.engram) |h| tailOf(&f.slot, f.ids[0..adv], h.max_ngram - 1);
    f.slot.pos = p.start + adv;
    f.slot.pending = null;
    if (f.sess) |x| try x.commit(x.ptr, f.ids[0..adv]);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep:].
fn tailOf(s: *fwd.Slot, ids: []const u32, keep: usize) void {
    if (ids.len >= keep) {
        @memcpy(s.tail[0..keep], ids[ids.len - keep ..]);
        s.tail_len = keep;
        return;
    }
    const old = @min(s.tail_len, keep - ids.len); // the old tail's last ones that stay
    std.mem.copyForwards(u32, s.tail[0..old], s.tail[s.tail_len - old .. s.tail_len]);
    @memcpy(s.tail[old .. old + ids.len], ids);
    s.tail_len = old + ids.len;
}

/// Once: the tables no loader form builds and the buffers a segment's exchanges / Engram rows need.
fn ready(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.ready) return;
    const r = f.runner;
    // exl3.prefill's Hadamard: H[i, j] = (-1)^popcount(i & j), bf16
    if (r.addressOf("s.pf.had")) |p| {
        const h = try f.gpa.alloc(u16, 128 * 128);
        defer f.gpa.free(h);
        for (0..128) |i| for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80;
        };
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = p, .len = h.len * 2 }, 0, std.mem.sliceAsBytes(h));
    }
    // x3gm's pointer tables: the routed experts' trellis addresses (forms' tables hold the shared expert's last)
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "tg", "tp_g" }, .{ "tu", "tp_u" }, .{ "td", "tp_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoExpertTable;
        try copy(r, dst, src, 8 * @as(usize, f.cfg.expertsOf(L).count));
    };
    // x3gm v2's width tables (gm2pf.zig, x3gm.Ragged.k2g / k2d): the routed experts' K2s, then a 0 sentinel
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "k2g", "k2_g" }, .{ "k2d", "k2_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoK2Table;
        const Et: usize = f.cfg.expertsOf(L).count;
        try copy(r, dst, src, 4 * Et);
        try zero(r, dst + 4 * Et, 4);
    };
    // a segment's exchange send and Engram rows at their largest (decode's would grow them on the first segment)
    const rows: usize = @intCast(f.opts.prefill_rows);
    const W: usize = f.comm.world();
    _ = try f.sendBuffer(r, 2 * rows * f.cfg.hidden);
    if (f.engram) |h| {
        const cols = h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
        _ = try grow(&f.gathered, r.d, W * 2 * rows * cols);
        for (0..h.layer_ids.len) |li| _ = try f.rows_stage[li].take(r.d, 2 * rows * cols);
    }
    _ = try f.ids_stage.take(r.d, 8 * rows);
    try r.stream.synchronize();
    st.ready = true;
}

fn grow(b: *?cuda.DeviceBuffer, d: *const cuda.Driver, bytes: usize) !u64 {
    if (b.* == null or b.*.?.len < bytes) {
        if (b.*) |*x| x.free();
        b.* = try cuda.DeviceBuffer.alloc(d, @max(bytes, 256));
    }
    return b.*.?.ptr;
}

fn copy(r: *run.Runner, dst: u64, src: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, r.stream.handle), "cuMemcpyDtoDAsync");
}

fn zero(r: *run.Runner, dst: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemsetD8Async(dst, 0, bytes, r.stream.handle), "cuMemsetD8Async");
}

fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n;
}

fn bytesOf(t: calls.Tensor) usize {
    return numel(t) * buffers.dtSize(t.dt);
}

fn dim(t: calls.Tensor, i: usize) usize {
    return @intCast(t.shape[i]);
}

fn ld(t: calls.Tensor) usize {
    return @intCast(t.stride[0]);
}

/// A prefill segment's glue steps (block_prefill.zig's module doc); the ones a decode window has too (positions,
/// embed, exchange, kit_logits, engram_rows, trace, and the materialised selection's topk / counts / block_pos) are the
/// decode glue's.
fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = &f.prefill_state;
    const pr = st.tbo orelse return glueOne(ctx, r, c);
    // TF_DSV41_PF_TBO: the step's micro-batch (B's are the side stream's) on the forward while it runs
    const m = if (c.side) &pr.b else &pr.a;
    const keep_ids = f.ids;
    const keep_pending = f.slot.pending;
    var keep_tail: Tail = .{ .len = f.slot.tail_len };
    @memcpy(keep_tail.ids[0..keep_tail.len], f.slot.tail[0..f.slot.tail_len]);
    f.ids = m.ids;
    f.slot.pending = m.pending;
    f.slot.tail_len = m.tail.len;
    @memcpy(f.slot.tail[0..m.tail.len], m.tail.items());
    st.moe = m.moe;
    st.layer = m.layer;
    if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
    defer {
        if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
        m.moe = st.moe;
        m.layer = st.layer;
        f.ids = keep_ids;
        f.slot.pending = keep_pending;
        f.slot.tail_len = keep_tail.len;
        @memcpy(f.slot.tail[0..keep_tail.len], keep_tail.items());
    }
    return glueOne(ctx, r, c);
}

fn glueOne(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = &f.prefill_state;
    const step = c.name["glue.".len..];
    const k = r.kernels.others(r.stream);
    const x = c.args;
    const eql = std.mem.eql;
    if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.BadGlue).glue(r, c, step);
    if (eql(u8, step, "proj_carry")) {
        // torch.cat([the slot's carry, the projection]): position start - 1's row, then the segment's
        const carry = x[0].arg.t;
        const buf = try r.tensorAddr(x[2].arg.t);
        try copy(r, buf, try r.tensorAddr(carry), bytesOf(carry));
        return copy(r, buf + bytesOf(carry), try r.tensorAddr(x[1].arg.t), bytesOf(x[1].arg.t));
    }
    if (eql(u8, step, "stream_merge")) return longpf.merge(&st.long, f.gpa, r, c);
    // TF_DSV41_PF_TBO: A's ratio-2 carry into the slot at its layer's end, and the pair's closing join (Runner.issue
    // made it: nothing to run)
    if (eql(u8, step, "copy_rows")) return copy(r, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "tbo_join")) return;
    if (eql(u8, step, "pf_seg")) return segGlue(f, r, c);
    if (eql(u8, step, "mpositions")) return runPositions(f, r, c);
    if (eql(u8, step, "ced_stash") or eql(u8, step, "ced_load")) return stashCopy(f, r, c, eql(u8, step, "ced_stash"));
    if (eql(u8, step, "zeros")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "image_keep")) return vision_rows.keep(f, r, c);
    if (eql(u8, step, "image_split") or eql(u8, step, "image_merge")) return vision_rows.splitRows(f, r, c, eql(u8, step, "image_merge"));
    if (eql(u8, step, "gm_ticket_again")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t)); // vision.moe's text call: the same layer
    if (eql(u8, step, "cand_keys")) {
        const blocks = x[0].arg.t;
        return k.candKeys(try r.tensorAddr(blocks), ld(blocks), dim(blocks, 0), dim(blocks, 1), @intCast(x[2].arg.i), try r.tensorAddr(x[1].arg.t));
    }
    if (eql(u8, step, "proj")) {
        // blocks._projection: f32(kv), or ratio 2 [f32(kv) | f32(gate)]
        const kv = x[0].arg.t;
        const gate: u64 = if (x.len == 3) try r.tensorAddr(x[1].arg.t) else 0;
        return k.widenCat(try r.tensorAddr(kv), gate, dim(kv, 0), dim(kv, 1), try r.tensorAddr(x[x.len - 1].arg.t));
    }
    if (eql(u8, step, "ix_w")) return k.f64Bf16F32(try r.tensorAddr(x[0].arg.t), try r.tensorAddr(x[1].arg.t), numel(x[0].arg.t));
    if (eql(u8, step, "stage_in") or eql(u8, step, "stage_out")) {
        // blocks.stage_rows(dst, dst_ring, src, src_ring, lo, hi), a tensor at a time (values, scales)
        const dst_ring: usize = @intCast(x[2].arg.i);
        const src_ring: usize = @intCast(x[5].arg.i);
        const lo: u64 = @intCast(x[6].arg.i);
        const hi: u64 = @intCast(x[7].arg.i);
        for (0..2) |j| {
            const dst = x[j].arg.t;
            const src = x[3 + j].arg.t;
            const row = dim(dst, 1) * buffers.dtSize(dst.dt);
            if (dim(src, 1) * buffers.dtSize(src.dt) != row) return error.BadGlue;
            try k.ringCopy(try r.tensorAddr(src), src_ring, try r.tensorAddr(dst), dst_ring, row, lo, hi);
        }
        return;
    }
    if (eql(u8, step, "gm_picks")) {
        const pick = x[0].arg.t;
        return k.gmPicks(try r.tensorAddr(pick), try r.tensorAddr(x[1].arg.t), dim(pick, 1), dim(x[2].arg.t, 1), dim(pick, 0), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t));
    }
    if (eql(u8, step, "gm_ticket")) {
        // a layer's MoE starts here: the layer gm_plan's K2 tables are of
        if (st.moe >= st.layers.len) return error.BadGlue;
        st.layer = st.layers[st.moe];
        st.moe += 1;
        return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    }
    if (eql(u8, step, "gm_plan")) {
        // x3gm._run_ragged: the picks of one width (the others past the table), then x3gm.plan's passes
        const pk = x[0].arg.t;
        const P = numel(pk);
        const k2: u32 = @intCast(x[6].arg.i);
        const down = x[7].arg.b;
        const bm: usize = @intCast(x[8].arg.i);
        const Et: usize = @intCast(x[9].arg.i);
        // k2 0 (x3gm._run_v2's one plan, gm2pf.zig): every pick, no width mask
        if (k2 == 0) return r.kernels.experts(r.stream).gmPlan(try r.tensorAddr(pk), P, Et, bm, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t));
        var nb: [64]u8 = undefined;
        // forms' "s.L<i>.ex.k2_{g,d}": each routed expert's K2 (gate = up), the shared expert's past Et (never read)
        const tab = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.k2_{s}", .{ st.layer, if (down) "d" else "g" })) orelse return error.NoK2Table;
        const masked = try grow(&st.masked, r.d, 4 * P);
        try k.widthMask(try r.tensorAddr(pk), P, tab, Et, k2, masked);
        const t = [5]u64{ try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t) };
        return r.kernels.experts(r.stream).gmPlan(masked, P, Et, bm, t[0], t[1], t[2], t[3], t[4]);
    }
    if (eql(u8, step, "swiglu")) {
        // the shared expert's silu(min(g, limit)) * clamp(u, +-limit), fp32
        const g = x[0].arg.t;
        const u = x[1].arg.t;
        const act = x[2].arg.t;
        return k.swiglu(try r.tensorAddr(g), ld(g), try r.tensorAddr(u), ld(u), try r.tensorAddr(act), ld(act), dim(g, 0), dim(g, 1), @floatCast(x[3].arg.f));
    }
    if (eql(u8, step, "moe_sum")) {
        // bf16(routed + shared): one RNE of the fp32 sum; no shared expert: the routed partial's cast
        const routed = x[0].arg.t;
        if (x.len == 3) return k.addBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), numel(routed));
        return k.castBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), numel(routed));
    }
    return Forward.glueFn(ctx, r, c);
}
