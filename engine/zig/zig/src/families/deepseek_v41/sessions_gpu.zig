//! M5: sessions on the GPU path (sessions.py / residency.py / sessdisk.py over the Zig forward). The family-neutral store
//! (zig/src/sessions: prefix entries, the RAM tier with its residency budget, the streamed NVMe tier) runs over the paged
//! pool (kv_state.zig); this file is DeepSeek's side of it:
//! - the slot's bounded state as the store snapshots it (`Snap`, store.Bounded): every backbone layer's SWA ring
//!   ("s.L<i>.swa.v" / ".s"), the ratio-2 carries ("s.L<i>.carry"), the position and the Engram tail. A RAM entry keeps
//!   a host copy; a park writes the same pieces as named blobs next to the entry's pages;
//! - `save`: an entry for the slot's committed ids (its pages shared, not copied), then the slot lets go of its pages;
//! - the served shape (own prefill, verify tail; Python prod's fast prefill tag): no turn entries; `savePrompt` keeps an
//!   entry at the prompt's replay point while the slot goes on prefilling (CED replay, ced.zig: its bounded state with
//!   the stash ring "s.ced.*" and the stash's positions and ids); entries carry the fast tag, replay's its own
//!   (ced.tag), so full and replay entries never resume each other, nor the old exact-tag turn entries;
//! - `resume`: the request's pool admission first (batch.py `_admit`, kv/sess.zig: its pages reserved for the slot, the
//!   coldest RAM entries parked or dropped when the pool is short), then the longest saved entry that strictly prefixes
//!   a prompt, from RAM (full pages adopted, the partial last one copied) or NVMe (streamed into fresh pages through the
//!   staging set), its bounded state uploaded: the slot continues at the entry's position, and the caller prefills the
//!   rest of the prompt;
//! - the RAM tier's budget counts each entry as prod's snapshot arrays would (kv/sess.zig Measure: the SWA rows below
//!   the decoder in replay mode, the stash, the float32 carries, the drafter's rows), so it trims at prod's saves;
//! - the residency budget (`Store.enforce`: idle chats / pool bytes held by entries alone) parks or drops the least
//!   recently used chat after every save.
//! Several live slots (TF_DSV41_SLOTS > 1, slots.zig): one store over the pool's slots, the operations on the slot
//!   slots.zig activated (its views of the stacked rings and carries, its `Forward.Slot`, the pool's current slot),
//!   each slot's committed ids and drafter context its own; row windows commit through `commitAt`.
//! Both ranks make the same calls in the same order (the leader's operations: forward.zig `op_sess_*`), so entries,
//! pages and evictions agree; a split rank parks and restores only the comp rows it owns.
//! Knobs as Python's: TF_DSV41_SESSIONS (0: no store), TF_DSV41_SESSION_DISK (the NVMe tier's directory; unset: RAM
//! only), TF_DSV41_SESSION_DISK_GIB (64: the tier's files, least recently used deleted past it),
//! TF_DSV41_SESSION_DISK_MIN (1024: shorter entries are dropped, not parked), TF_DSV41_SESSION_RAM_MIB (256),
//! TF_DSV41_SESSION_RESIDENT (idle chats kept), TF_DSV41_SESSION_RESIDENT_GIB, TF_DSV41_SESSION_STAGING_MIB;
//! TF_DSV41_SESSION_SKIP_COVERED=1 (drop, never park, an entry a newer one of its chat covers) and
//! TF_DSV41_SESSION_DELTA=1 (park a newer turn as a delta on its older file) are off by default, as prod (8474f31)
//! has neither. TF_DSV41_SESSION_SLOTS=1 opens the store with several live slots (default off: model.zig says so at
//! boot and serves without sessions, as before).

const std = @import("std");
const cuda = @import("cuda");
const kv = @import("kv");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const kvs = @import("kv_state.zig");
const buffers = @import("buffers.zig");
const ced = @import("ced.zig");
const rowmode = @import("rowmode.zig");
const slots_mod = @import("slots.zig");
const graphs = @import("graphs.zig");
const lockstep = @import("lockstep.zig");
const dspark_emit = @import("dspark_emit.zig");

const sessions = kv.sessions;
const Store = sessions.Store;
const Blob = sessions.disk.Blob;

pub const Error = error{ NoPool, BadBlob, SlotNotActive };

pub const Options = struct {
    /// TF_DSV41_SESSIONS (default 0; explicit 1 enables reuse)
    enabled: bool = false,
    disk: ?[]const u8 = null,
    disk_budget: u64 = 64 << 30,
    disk_min_tokens: u64 = 1024,
    resident: u32 = 0,
    resident_bytes: u64 = 0,
    /// TF_DSV41_SESSION_RAM_MIB (stack.py's default 256)
    ram_bytes: u64 = 256 << 20,
    staging_mib: u32 = 128,
    skip_covered: bool = false,
    delta: bool = false,
    /// TF_DSV41_SESSION_SLOTS=1: the store with several live slots too (default off until the sess4 gate passes on
    /// hardware: tools/zig/dsv41_sess4/job.sh, the port doc's NEXT SPARK WINDOW S4x)
    several: bool = false,
};

fn flag(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

pub fn optionsFromEnv() !Options {
    var o: Options = .{};
    if (std.c.getenv("TF_DSV41_SESSIONS")) |v| o.enabled = !std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "0");
    if (std.c.getenv("TF_DSV41_SESSION_DISK")) |v| if (std.mem.trim(u8, std.mem.span(v), " ").len > 0) {
        o.disk = std.mem.trim(u8, std.mem.span(v), " ");
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_GIB")) |v| if (std.mem.span(v).len > 0) {
        o.disk_budget = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_MIN")) |v| if (std.mem.span(v).len > 0) {
        o.disk_min_tokens = try std.fmt.parseInt(u64, std.mem.span(v), 10);
    };
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT")) |v| o.resident = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT_GIB")) |v| o.resident_bytes = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    if (std.c.getenv("TF_DSV41_SESSION_RAM_MIB")) |v| if (std.mem.span(v).len > 0) {
        o.ram_bytes = (try std.fmt.parseInt(u64, std.mem.span(v), 10)) << 20;
    };
    if (std.c.getenv("TF_DSV41_SESSION_STAGING_MIB")) |v| o.staging_mib = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    o.skip_covered = flag("TF_DSV41_SESSION_SKIP_COVERED");
    o.delta = flag("TF_DSV41_SESSION_DELTA");
    o.several = flag("TF_DSV41_SESSION_SLOTS");
    return o;
}

/// What the store is opened over: the live slots (TF_DSV41_SLOTS) and the drafter's blocks (0: drafts off), the latter
/// for the RAM measure only (prod's snapshot carries the drafter's rows).
pub const Shape = struct { slots: u32 = 1, ds_blocks: u32 = 0 };

/// One device role of the bounded state: a slot's bytes (a stacked role's per-slot view, slots.zig), at the role's
/// address when the operation runs (the activated slot's view).
/// `stride`: a role stacked over the slots that no activation views (the drafter's rings): the slot's bytes at
/// base + slot x stride.
const Piece = struct { role: []const u8, len: usize, stride: u64 = 0 };

pub const Sessions = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    f: *fwd.Forward,
    kx: *kvs.Kv,
    store: Store,
    disk: ?*sessions.Disk = null,
    pieces: []Piece,
    bounded_bytes: u64,
    /// each slot's committed ids (prefill and kept windows append): a saved entry's ids
    hist: []std.ArrayList(i32),
    /// each slot's drafter context start (drafter.valid: 0 after a reset, pos - n after restoring n rows), for the measure
    ds_valid: []u64,
    measure: kv.sess.Measure,
    /// several slots: the operations run on the activated slot (checked)
    slot_set: ?*slots_mod.SlotSet = null,
    names: std.heap.ArenaAllocator,
    stats: struct { admits: u64 = 0, spills: u64 = 0, waits: u64 = 0, damaged: u64 = 0 } = .{},
    /// several ranks: a restore's verdict agreed (a rank's NVMe files are its own: one damaged file sends every rank's
    /// slot back to a fresh prefill)
    agreement: ?graphs.Agreement = null,
    /// full mode: the drafter's rings ride in the snapshot (drafter.py `snapshot` / `restore`); a restore sets the
    /// pass's context start through `ds_set` (model.zig: the pass the lanes drive)
    ds_rings: bool = false,
    ds_set: ?DsSet = null,

    pub const DsSet = struct { ctx: *anyopaque, set: *const fn (ctx: *anyopaque, slot: u32, valid: u64) void };

    /// The store over the forward's pool (after Runner.bind: the bounded roles have addresses).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, f: *fwd.Forward, plan: *const buffers.Plan, o: Options, shape: Shape) !*Sessions {
        const kx = f.kv orelse return error.NoPool;
        if (kx.slots.len != shape.slots) return error.SlotRange;
        const s = try gpa.create(Sessions);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .io = io, .f = f, .kx = kx, .store = undefined, .pieces = &.{}, .bounded_bytes = 0, .hist = &.{}, .ds_valid = &.{}, .measure = undefined, .names = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.names.deinit();
        const na = s.names.allocator();
        s.hist = try na.alloc(std.ArrayList(i32), shape.slots);
        @memset(s.hist, .empty);
        s.ds_valid = try na.alloc(u64, shape.slots);
        @memset(s.ds_valid, 0);
        // the bounded roles in a fixed order (both ranks; a park's blobs carry their names); the rings and carries are
        // stacked over the slots in row mode (slots.zig): a slot's bytes are the plan's / S
        var list: std.ArrayList(Piece) = .empty;
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const layers = try f.backbone(la.allocator());
        for (layers) |L| {
            for ([_][]const u8{ "swa.v", "swa.s", "carry" }) |part| {
                const role = try std.fmt.allocPrint(na, "s.L{d}.{s}", .{ L, part });
                const size = plan.sizes.get(role) orelse continue;
                if (f.runner.addressOf(role) == null) return error.Unbound;
                const stacked = shape.slots > 1 and (rowmode.isRing(role) or rowmode.isCarry(role));
                if (stacked and size % shape.slots != 0) return error.Unbound;
                const len = if (stacked) size / shape.slots else size;
                try list.append(na, .{ .role = role, .len = len });
                s.bounded_bytes += len;
            }
        }
        // CED replay: the stash ring (planned in replay mode only; one ring: a prompt's prefill runs on one slot alone)
        for (ced.roles) |role| if (plan.sizes.get(role)) |len| {
            if (f.runner.addressOf(role) == null) return error.Unbound;
            try list.append(na, .{ .role = role, .len = len });
            s.bounded_bytes += len;
        };
        // full mode with drafts: the drafter's rings (prod's snapshot holds their rows [max(pos - window, valid), pos)
        // and its restore puts them back: drafter.py 296-317); replay mode: the decoder replay rewrites every row the
        // drafter reads after a resume (ced.handoff), as prod's does over its restored rows
        const replay = f.prefill_state.mode == .replay;
        if (!replay and shape.ds_blocks > 0) {
            const rr: u64 = @intCast(dspark_emit.ringRows(f.cfg));
            for (plan.sizes.keys(), plan.sizes.values()) |role, size| {
                if (!std.mem.startsWith(u8, role, "s.L")) continue;
                const v = std.mem.endsWith(u8, role, ".ring.v");
                if (!v and !std.mem.endsWith(u8, role, ".ring.s")) continue;
                const L = std.fmt.parseInt(u32, role["s.L".len .. std.mem.indexOfScalarPos(u8, role, 3, '.') orelse continue], 10) catch continue;
                if (L < f.cfg.layers) continue;
                const row: u64 = if (v) 576 else 8;
                if (size % (rr * row) != 0) return error.Unbound;
                const ring_slots = size / (rr * row);
                if (f.runner.addressOf(role) == null) return error.Unbound;
                const len = size / ring_slots;
                try list.append(na, .{ .role = try na.dupe(u8, role), .len = len, .stride = if (ring_slots > 1) len else 0 });
                s.bounded_bytes += len;
                s.ds_rings = true;
            }
        }
        s.pieces = list.items;
        // prod's snapshot as the RAM tier measures it (kv/sess.zig): replay keeps the rings below the decoder
        var rings: u64 = 0;
        var carries: u64 = 0;
        const d = if (replay) try ced.decoderStart(f.cfg) else f.cfg.layers;
        for (layers) |L| {
            if (L < d) rings += 1;
            if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) carries += 1;
        }
        s.measure = .{ .rings = rings, .window = f.cfg.window, .carries = carries, .head_dim = f.cfg.head_dim, .hidden = f.cfg.hidden, .hc = f.cfg.hc_mult, .ds_blocks = shape.ds_blocks, .replay = replay };
        if (o.disk) |root| {
            // the tier's compat ident: the families' layout, the split and this rank (a file is one rank's rows)
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            for (kx.layout.families()) |fam| h.update(std.mem.asBytes(&[_]u32{ fam.ratio, fam.row_bytes, @intFromBool(fam.split) }));
            h.update(std.mem.asBytes(&[_]u32{ kx.world, kx.rank, kx.pool.page }));
            var full: [32]u8 = undefined;
            h.final(&full);
            // the staging set (TF_DSV41_SESSION_STAGING_MIB) in 8 MiB chunks: a park's or restore's whole host transient
            s.disk = try sessions.Disk.open(gpa, io, .{ .root = root, .compat = full[0..16].*, .rank = kx.rank, .world = kx.world, .budget = o.disk_budget, .min_tokens = o.disk_min_tokens, .staging = @max(2, o.staging_mib / 8), .chunk = 8 << 20 });
        }
        errdefer if (s.disk) |dd| dd.close();
        s.store = try Store.init(gpa, io, &kx.pool, kx.dev.store(), s.disk, .{ .ram_bytes = o.ram_bytes, .chats = o.resident, .resident_bytes = o.resident_bytes, .skip_covered = o.skip_covered, .delta = o.delta });
        if (f.comm.world() > 1) s.agreement = try graphs.Agreement.init(f.runner.d, f.comm, f.runner.stream);
        f.sess = .{ .ptr = s, .commit = commitFn, .save = saveFn, .restore = restoreFn, .reset = resetFn, .prompt = promptFn, .admit = admitFn };
        return s;
    }

    fn commitFn(ptr: *anyopaque, ids: []const u32) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.commitAt(s.cur(), ids);
    }

    fn saveFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.save();
    }

    fn promptFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.savePrompt();
    }

    fn resetFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.reset();
    }

    fn restoreFn(ptr: *anyopaque, id: u32, ids: []const i64) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        const p = try s.gpa.alloc(u32, ids.len);
        defer s.gpa.free(p);
        for (ids, p) |t, *y| y.* = @intCast(t);
        _ = try s.restore(id, if (ids.len > 0) p else null);
    }

    /// A follower: the leader's admission (forward.zig op_sess_admit: [op, need, n, spilled ids...]).
    fn admitFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        if (msg.len < 3) return error.BadPlan;
        const n: usize = @intCast(msg[2]);
        if (msg.len != 3 + n) return error.BadPlan;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(s.gpa);
        for (msg[3..]) |x| try ids.append(s.gpa, @intCast(x));
        try s.applyAdmit(@intCast(msg[1]), ids.items);
    }

    pub fn deinit(s: *Sessions) void {
        s.f.sess = null;
        if (s.agreement) |*a| a.deinit();
        s.store.deinit();
        if (s.disk) |d| d.close();
        for (s.hist) |*h| h.deinit(s.gpa);
        s.names.deinit();
        s.gpa.destroy(s);
    }

    /// The slot the one-slot operations run on (the pool's current slot: slots.zig `activate`).
    pub fn cur(s: *const Sessions) u32 {
        return s.kx.slot.id;
    }

    /// Several slots: the slot's views must be the bound ones (its rings, carries and `Forward.Slot`).
    fn active(s: *const Sessions) !void {
        const ss = s.slot_set orelse return;
        if (ss.active != s.cur()) return error.SlotNotActive;
    }

    /// Slot `slot` committed `ids` (prefill segments and kept windows on the active slot; row windows by slot, batch.zig).
    pub fn commitAt(s: *Sessions, slot: u32, ids: []const u32) !void {
        for (ids) |t| try s.hist[slot].append(s.gpa, @intCast(t));
    }

    // -------------------------------------------------------------------------------------------------------------
    // the bounded state (store.Bounded)

    const Snap = struct {
        owner: *Sessions,
        bytes: []u8,
        pos: u64,
        tail: [eh.max_ngram]u32,
        tail_len: usize,
        stash: ced.Stash,
        /// the drafter's rows prod's snapshot holds (a restore's context start is pos - ds_rows)
        ds_rows: u64,
        /// prod's `Bounded.nbytes` of this snapshot: the RAM tier's measure
        measure: u64,

        fn bounded(p: *Snap) sessions.store.Bounded {
            return .{ .ptr = p, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
        }

        fn nbytes(ptr: *anyopaque) u64 {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            return p.measure;
        }

        /// The park's blobs: each role's bytes under its name, then "slot" (position, tail).
        fn blobs(ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            const ps = p.owner.pieces;
            const out = try gpa.alloc(Blob, ps.len + 1);
            var at: usize = 0;
            for (ps, out[0..ps.len]) |pc, *b| {
                b.* = .{ .name = try gpa.dupe(u8, pc.role), .bytes = try gpa.dupe(u8, p.bytes[at..][0..pc.len]) };
                at += pc.len;
            }
            var slot: [slot_bytes]u8 = undefined;
            std.mem.writeInt(u64, slot[0..8], p.pos, .little);
            std.mem.writeInt(u64, slot[8..16], p.tail_len, .little);
            for (0..eh.max_ngram) |i| std.mem.writeInt(u32, slot[16 + 4 * i ..][0..4], if (i < p.tail_len) p.tail[i] else 0, .little);
            // CED: the stash's positions and ids (a full-mode entry's are zeros)
            const c = slot[ced_at..];
            std.mem.writeInt(u64, c[0..8], p.stash.from, .little);
            std.mem.writeInt(u64, c[8..16], p.stash.hi, .little);
            for (p.stash.ids, 0..) |t, i| std.mem.writeInt(u32, c[16 + 4 * i ..][0..4], t, .little);
            std.mem.writeInt(u64, slot[ds_at..][0..8], p.ds_rows, .little);
            out[ps.len] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, &slot) };
            return out;
        }

        fn release(ptr: *anyopaque) void {
            const p: *Snap = @ptrCast(@alignCast(ptr));
            const gpa = p.owner.gpa;
            gpa.free(p.bytes);
            gpa.destroy(p);
        }
    };

    /// The "slot" blob: position, tail length, the tail; then (CED) the stash's from / hi and its ids; then the drafter's
    /// rows of the measure. Older parks have the shorter forms (their stash empty, no drafter rows).
    const ced_at: usize = 16 + 4 * eh.max_ngram;
    const ds_at: usize = ced_at + 16 + 4 * ced.ring;
    const slot_bytes: usize = ds_at + 8;

    /// The active slot's bounded state now (device to host, after the stream's work).
    fn snapshot(s: *Sessions) !*Snap {
        try s.active();
        const r = s.f.runner;
        try r.stream.synchronize();
        const p = try s.gpa.create(Snap);
        errdefer s.gpa.destroy(p);
        const sl = &s.f.slot;
        const ds = if (s.measure.ds_blocks > 0) kv.sess.dsRows(sl.pos, s.ds_valid[s.cur()], s.measure.window) else 0;
        const stash = if (s.measure.replay) kv.sess.stashRows(sl.pos, sl.ced.from) else 0;
        p.* = .{ .owner = s, .bytes = try s.gpa.alloc(u8, s.bounded_bytes), .pos = sl.pos, .tail = sl.tail, .tail_len = sl.tail_len, .stash = sl.ced, .ds_rows = ds, .measure = s.measure.bytes(sl.pos, stash, ds) };
        errdefer s.gpa.free(p.bytes);
        var at: usize = 0;
        for (s.pieces) |pc| {
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = addr, .len = pc.len }, 0, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        return p;
    }

    fn uploadSnap(s: *Sessions, p: *const Snap) !void {
        const r = s.f.runner;
        var at: usize = 0;
        for (s.pieces) |pc| {
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = addr, .len = pc.len }, 0, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        s.f.slot.pos = p.pos;
        s.f.slot.tail = p.tail;
        s.f.slot.tail_len = p.tail_len;
        s.f.slot.ced = p.stash;
        s.ds_valid[s.cur()] = p.pos - p.ds_rows;
        s.dsRestored();
    }

    /// The drafter's rings came back with the entry: its context starts where prod's restore puts it (pos - rows).
    fn dsRestored(s: *Sessions) void {
        if (!s.ds_rings) return;
        const h = s.ds_set orelse return;
        h.set(h.ctx, s.cur(), s.ds_valid[s.cur()]);
    }

    fn uploadBlobs(s: *Sessions, bs: []const Blob) !void {
        const r = s.f.runner;
        for (bs) |b| {
            if (std.mem.eql(u8, b.name, "slot")) {
                if (b.bytes.len < 16) return error.BadBlob;
                s.f.slot.pos = std.mem.readInt(u64, b.bytes[0..8], .little);
                s.f.slot.tail_len = @intCast(std.mem.readInt(u64, b.bytes[8..16], .little));
                for (0..eh.max_ngram) |i| s.f.slot.tail[i] = std.mem.readInt(u32, b.bytes[16 + 4 * i ..][0..4], .little);
                s.f.slot.ced = .{};
                if (b.bytes.len >= ds_at) {
                    const c = b.bytes[ced_at..];
                    s.f.slot.ced.from = std.mem.readInt(u64, c[0..8], .little);
                    s.f.slot.ced.hi = std.mem.readInt(u64, c[8..16], .little);
                    for (&s.f.slot.ced.ids, 0..) |*t, i| t.* = std.mem.readInt(u32, c[16 + 4 * i ..][0..4], .little);
                }
                const ds: u64 = if (b.bytes.len >= slot_bytes) std.mem.readInt(u64, b.bytes[ds_at..][0..8], .little) else 0;
                s.ds_valid[s.cur()] = s.f.slot.pos -| ds;
                s.dsRestored();
                continue;
            }
            const pc = for (s.pieces) |pc| {
                if (std.mem.eql(u8, pc.role, b.name)) break pc;
            } else return error.BadBlob;
            if (pc.len != b.bytes.len) return error.BadBlob;
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = addr, .len = pc.len }, 0, b.bytes);
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    // the operations (every rank, the leader's order)

    /// An entry for the slot's committed ids (null: a duplicate), then the slot's pages go back (the entry holds them).
    pub fn save(s: *Sessions) !?u32 {
        const h = &s.hist[s.cur()];
        if (h.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        try s.f.sendSess(.save, 0);
        if (h.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), h.items, s.tag(), 0) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.reset();
        // the residency budget past the save: the least recently used chats parked (streamed) or dropped
        try s.store.settleAll();
        return id;
    }

    /// The served shape's prompt snapshot (batch.py `save_at`, rounds.py `_save(slot, "prompt")`): an entry for the committed
    /// ids at the slot's position (the prompt's replay point), its bounded state with the stash; the slot keeps its
    /// pages and goes on prefilling (it only writes positions past the entry's, which a resume rewrites before reading).
    pub fn savePrompt(s: *Sessions) !?u32 {
        const h = &s.hist[s.cur()];
        if (h.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        if (s.f.prefill_state.mode == .replay and s.f.slot.ced.hi != s.f.slot.pos) return error.BadWindow;
        try s.f.sendSess(.prompt, 0);
        if (h.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), h.items, s.tag(), prompt_kind) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.store.settleAll();
        return id;
    }

    /// sessions.Entry.kind: a prompt's replay point (a turn's end is 0)
    const prompt_kind: u32 = 1;

    /// The entries' tag (sessions.Tag's CED field): replay entries never resume full ones, nor full ones replay's.
    fn tag(s: *const Sessions) u32 {
        return if (s.f.prefill_state.mode == .replay) ced.tag else ced.tag_fast;
    }

    /// The slot empty, its pages back (a session entry may hold them), its reservation gone.
    pub fn reset(s: *Sessions) !void {
        try s.kx.release();
        s.f.slot = .{};
        s.hist[s.cur()].clearRetainingCapacity();
        s.ds_valid[s.cur()] = 0;
    }

    /// The request admitted into the (empty) slot, then the longest saved entry that strictly prefixes `prompt`: its
    /// pages and bounded state; the slot's position is the entry's (the caller prefills prompt[pos..]). 0: nothing found,
    /// or its NVMe file did not read back (prod: a fresh prefill). `max_new` (the request's reply tokens; null: unknown)
    /// prices its pool pages as prod's admission does (`admit`).
    pub fn resume_(s: *Sessions, prompt: []const u32, max_new: ?u64) !u64 {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        const hit = try s.find(prompt);
        if (max_new) |mn| try s.admit(hit, prompt.len, mn);
        s.ds_valid[s.cur()] = 0;
        return (try s.resumeHit(prompt, hit)).at;
    }

    /// The longest saved entry that strictly prefixes `prompt` (a row must remain to prefill), null: none.
    pub fn find(s: *Sessions, prompt: []const u32) !?sessions.store.Hit {
        const ids = try s.gpa.alloc(i32, prompt.len);
        defer s.gpa.free(ids);
        for (ids, prompt) |*d, t| d.* = @intCast(t);
        const h = (try s.store.find(s.tag(), ids)) orelse return null;
        return if (h.pos >= prompt.len) null else h;
    }

    /// batch.py's pool decision for a request (kv/sess.zig `admit`; the round planner's: kv/sched.zig).
    pub fn poolPlan(s: *Sessions, n: u64, max_new: u64, hit: ?sessions.store.Hit, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) !kv.sess.Plan {
        return kv.sess.admit(&s.store, hit, n, max_new, s.kx.slot.capacity, held, extra, spills);
    }

    pub const Resumed = struct { at: u64 = 0, damaged: bool = false };

    /// The round planner's admission into the (empty) slot (rounds.py `_admit`): `spills` evicted (the round's, with
    /// its first admission), `need` pages reserved, then `hit` restored. Every rank, in the leader's order.
    pub fn resumePlanned(s: *Sessions, prompt: []const u32, need: u32, hit: ?sessions.store.Hit, spills: []const u32) !Resumed {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        s.stats.admits += 1;
        try s.f.sendSessAdmit(need, spills);
        try s.applyAdmit(need, spills);
        s.ds_valid[s.cur()] = 0;
        return s.resumeHit(prompt, hit);
    }

    fn resumeHit(s: *Sessions, prompt: []const u32, hit: ?sessions.store.Hit) !Resumed {
        const h = hit orelse return .{};
        const n = prompt.len;
        // out of RAM no tokens are indexed: the history is the prompt's prefix (rounds.py `_admit`)
        const ids: ?[]const u32 = if (s.store.entry(h.id).ram) null else prompt[0..@intCast(h.pos)];
        try s.f.sendRestore(h.id, ids);
        if (!try s.restore(h.id, ids)) return .{ .damaged = true };
        std.log.scoped(.dsv41).info("sessions: slot {d} resumed {d} of {d} prompt tokens from {s}", .{ s.cur(), h.pos, n, if (h.ram) "RAM" else "NVMe" });
        return .{ .at = h.pos };
    }

    /// batch.py `_admit` for the request (kv/sess.zig): the leader decides its pages and the entries to let go of, every
    /// rank evicts those and reserves the pages for the slot (rounds.py `_admit`: before the restore). When even every
    /// RAM entry is not enough prod keeps the request queued; the lanes cannot hold it back, so it goes on unreserved
    /// (pages taken as positions are written, today's one-slot behaviour) and the count is kept (`stats.waits`).
    fn admit(s: *Sessions, hit: ?sessions.store.Hit, n: usize, max_new: u64) !void {
        var spills: std.ArrayList(u32) = .empty;
        defer spills.deinit(s.gpa);
        const held: []const u32 = if (hit) |h| &.{h.id} else &.{};
        const plan = try kv.sess.admit(&s.store, hit, n, max_new, s.kx.slot.capacity, held, 0, &spills);
        s.stats.admits += 1;
        if (plan.wait) {
            s.stats.waits += 1;
            std.log.scoped(.dsv41).warn("sessions: a {d}-token request (+{d}) needs {d} pool pages, {d} are free and unreserved even with every RAM entry gone; prod would queue it, it runs unreserved", .{ n, max_new, plan.want, s.kx.pool.available() });
            return;
        }
        try s.f.sendSessAdmit(plan.need, spills.items);
        try s.applyAdmit(plan.need, spills.items);
    }

    fn applyAdmit(s: *Sessions, need: u32, spills: []const u32) !void {
        for (spills) |id| _ = try s.store.evict(id);
        try s.store.settleAll();
        s.stats.spills += spills.len;
        s.kx.slot.reserve(need);
    }

    /// Entry `id` into the empty slot, agreed (lockstep.zig); false: not back on some rank, so a miss everywhere.
    pub fn restore(s: *Sessions, id: u32, ids: ?[]const u32) !bool {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        try s.active();
        const pos = s.store.entry(id).pos;
        const g: ?lockstep.Agree = if (s.agreement) |*a| blk: {
            const x = a.agree();
            break :blk .{ .ctx = x.ctx, .all = x.all };
        } else null;
        if (!try lockstep.agreed(g, s.restoreOn(id, ids), "a session restore", s.f.comm.rank())) {
            try s.kx.truncate(0);
            s.f.slot = .{};
            s.hist[s.cur()].clearRetainingCapacity();
            s.ds_valid[s.cur()] = 0;
            // a cache miss from here: the entry out of RAM and its file deleted, on every rank alike
            if (s.store.entry(id).ram) try s.store.drop(id);
            if (s.store.entry(id).live) try s.store.forget(id); // drop removed an entry no file holds
            s.stats.damaged += 1;
            std.log.scoped(.dsv41).warn("sessions: slot {d}: the entry of {d} tokens did not come back; its prompt prefills from 0", .{ s.cur(), pos });
            return false;
        }
        try s.kx.dev.syncTables();
        return true;
    }

    /// This rank's restore of `id`: pages and bounded state (RAM or NVMe), the position, the slot's history.
    fn restoreOn(s: *Sessions, id: u32, ids: ?[]const u32) !void {
        const e = s.store.entry(id);
        const pos = e.pos;
        if (e.ram) {
            const b = try s.store.restoreRam(id, s.kx.slot);
            try s.uploadSnap(@ptrCast(@alignCast(b.ptr)));
        } else try s.restoreDisk(id);
        if (s.f.slot.pos != pos) return error.RestoredPosition;
        const h = &s.hist[s.cur()];
        try h.resize(s.gpa, pos);
        // RAM: the path's indexed tokens; NVMe: the ids the leader sent (its prompt's prefix), as Python's history
        const given: ?[]i32 = if (ids) |x| try s.gpa.alloc(i32, x.len) else null;
        defer if (given) |gv| s.gpa.free(gv);
        if (given) |gv| for (ids.?, gv) |t, *y| {
            y.* = @intCast(t);
        };
        try s.store.historyOf(id, given, h.items);
    }

    fn restoreDisk(s: *Sessions, id: u32) !void {
        const j = try s.store.beginRestore(id, s.kx.slot);
        var restored = (try s.store.settle(j)) orelse return error.BadBlob;
        defer restored.deinit(s.gpa);
        try s.uploadBlobs(restored.blobs);
    }

    /// Every RAM entry parked to the NVMe tier now (the gate's interruption; a server parks through the budget).
    pub fn parkAll(s: *Sessions) !u32 {
        var n: u32 = 0;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(s.gpa);
        var id = s.store.oldest;
        while (id != sessions.store.none) : (id = s.store.entry(id).extra.newer) try ids.append(s.gpa, id);
        for (ids.items) |e| if (try s.store.beginPark(e)) |j| {
            _ = try s.store.settle(j);
            n += 1;
        };
        return n;
    }
};
