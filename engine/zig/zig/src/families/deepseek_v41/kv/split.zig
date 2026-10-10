//! Split KV's row exchange (kvsplit.py, LONG-CONTEXT.md 7.2): after a selection, every rank sends the selected comp rows it owns and one
//! all-gather (tp's Collective) brings each rank all of them; the attention then reads the received rows unpaged with remapped tokens.
//!
//! Logical page k of a slot lives on rank k % W, so a row's owner is its logical page's residue. The core reads the same bytes in the same
//! list order as from the replicated pool: split == replicated, bit for bit.
//! - dense (decode windows, row mode: fixed shapes, inside CUDA graphs): entry (r, j) of the [R, K] selection becomes token owner*R*K + r*K + j;
//! - union (prefill segments, host-synced): the sorted unique rows, each owner's list padded to the longest; token owner*M + index in its list;
//!   halves of the segment's rows while send + receive pass the cap (rows are independent in every attention kernel).
//! - packed (TF_DSV41_KV_SPLIT_COMPACT, kvsplit.py's dense_packed / kvsplit_pack.pack2): dense with each owner's valid entries packed in list
//!   order. Every rank computes, from the selection and the page residues alike, each entry's owner o and its place i among o's valid entries;
//!   rank r gathers its owned rows to send[0:c_r], lens[r] = c_r x row bytes, entry (r, j) becomes token o*R*K + i. The stride stays R x K
//!   rows (dense's buffers and fit); a transport that takes device lengths (`Collective.allGatherV`: RoCE) moves only c_r rows of rank r,
//!   any other the whole strides. The same rows behind the same entries in the same order: the same bits as dense, about half the bytes.

const std = @import("std");
const tp = @import("tp");
const DevicePtr = tp.collective.DevicePtr;
const Stream = tp.collective.Stream;

/// One selection's dense gather: [R, K] logical rows of a comp family -> this rank's owned rows into send, and the remapped tokens.
pub const DenseArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 split page tables (Slot.localTableAt): one [pages] table, or row mode's stacked [S, pts]
    table: u64,
    pts: u32,
    /// int32 [R]: each row's slot in the stacked tables (0: every row reads table row 0)
    rslot: u64,
    /// log2 of rows a page
    psh: u32,
    /// the family's local tensor and its row bytes
    base: u64,
    row_bytes: u32,
    world: u32,
    /// out: [R x K, row bytes] and int32 [R, K]
    send: u64,
    tok: u64,
};

/// One selection's packed gather (`Exchange.densePacked`): DenseArgs' fields at the same offsets, plus this rank and the lengths.
pub const PackArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 split page tables, as DenseArgs
    table: u64,
    pts: u32,
    /// int32 [R] or 0, as DenseArgs
    rslot: u64,
    psh: u32,
    /// this rank (sends the entries whose owner it is)
    rank: u32,
    base: u64,
    row_bytes: u32,
    world: u32,
    /// out: [R x K, row bytes] (this rank's owned rows in list order in the first c_rank; the rest unspecified), int32 [R, K] tokens,
    /// and int32 [world]: each owner's count x row bytes
    send: u64,
    tok: u64,
    lens: u64,

    pub fn of(a: DenseArgs, rank: u32, lens: u64) PackArgs {
        return .{ .sel = a.sel, .rows = a.rows, .k = a.k, .table = a.table, .pts = a.pts, .rslot = a.rslot, .psh = a.psh, .rank = rank, .base = a.base, .row_bytes = a.row_bytes, .world = a.world, .send = a.send, .tok = a.tok, .lens = lens };
    }

    pub fn dense(p: PackArgs) DenseArgs {
        return .{ .sel = p.sel, .rows = p.rows, .k = p.k, .table = p.table, .pts = p.pts, .rslot = p.rslot, .psh = p.psh, .base = p.base, .row_bytes = p.row_bytes, .world = p.world, .send = p.send, .tok = p.tok };
    }
};

comptime {
    std.debug.assert(@sizeOf(DenseArgs) == 80 and @sizeOf(PackArgs) == 88);
    std.debug.assert(@offsetOf(PackArgs, "rank") == 44 and @offsetOf(PackArgs, "base") == @offsetOf(DenseArgs, "base") and @offsetOf(PackArgs, "lens") == 80);
}

/// The device work around the all-gather: the dense gather + remap and a gather by row list (the kernel agent's .cu; HostKernels below).
pub const Kernels = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        dense: *const fn (ptr: *anyopaque, a: DenseArgs, stream: Stream) anyerror!void,
        /// The packed gather: tokens, this rank's owned rows into send[0:c_rank], lens (kvsplit.cu pack_kernel: one launch, any W <= 8).
        /// Null: a Kernels without it (Exchange.densePacked refuses: error.Unsupported).
        pack: ?*const fn (ptr: *anyopaque, a: PackArgs, stream: Stream) anyerror!void = null,
        /// send[i] = base row phys[i] (uint32 [n])
        gather: *const fn (ptr: *anyopaque, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64, stream: Stream) anyerror!void,
    };
};

/// The kernels on host memory (addresses are host pointers): the reference the device kernels must equal, and the host ranks' path.
pub const HostKernels = struct {
    pub fn kernels(h: *HostKernels) Kernels {
        return .{ .ptr = h, .vtable = &.{ .dense = dense, .gather = gather, .pack = pack } };
    }

    /// One pass in list order: an owner's running count is each entry's place (pack_kernel's prefix sums).
    fn pack(_: *anyopaque, a: PackArgs, _: Stream) anyerror!void {
        if (a.world < 1 or a.world > max_world or a.rank >= a.world) return error.BadWorld;
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const lens: [*]i32 = @ptrFromInt(a.lens);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        var count: [max_world]u32 = @splat(0);
        for (0..n) |i| {
            if (sel[i] < 0) {
                tok[i] = -1;
                continue;
            }
            const t: u32 = @intCast(sel[i]);
            const o = ownerOf(t, a.psh, a.world);
            const at = count[o];
            count[o] += 1;
            tok[i] = @intCast(o * n + at);
            if (o != a.rank) continue;
            const s: usize = if (rslot) |rs| @intCast(rs[i / a.k]) else 0;
            const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
            const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
            @memcpy(send[at * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
        }
        for (0..a.world) |o| lens[o] = @intCast(count[o] * a.row_bytes);
    }

    fn dense(_: *anyopaque, a: DenseArgs, _: Stream) anyerror!void {
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        for (0..n) |i| {
            const t: u32 = @intCast(@max(sel[i], 0));
            const row = i / a.k;
            const s: usize = if (rslot) |rs| @intCast(rs[row]) else 0;
            const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
            const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
            @memcpy(send[i * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
            tok[i] = if (sel[i] < 0) -1 else @intCast(ownerOf(t, a.psh, a.world) * n + i);
        }
    }

    fn gather(_: *anyopaque, base: u64, rb: u32, phys: u64, n: u32, send: u64, _: Stream) anyerror!void {
        const ph: [*]const u32 = @ptrFromInt(phys);
        const b: [*]const u8 = @ptrFromInt(base);
        const s: [*]u8 = @ptrFromInt(send);
        for (0..n) |i| @memcpy(s[i * rb ..][0..rb], b[@as(usize, ph[i]) * rb ..][0..rb]);
    }
};

/// The rank that stores logical row t of a split family: its logical page's residue.
pub fn ownerOf(t: u32, psh: u32, world: u32) u32 {
    return (t >> @intCast(psh)) % world;
}

/// Send and receive bytes of a dense exchange of R x K entries.
pub fn denseBytes(rows: u32, k: u32, row_bytes: u32, world: u32) struct { send: u64, recv: u64 } {
    const s = @as(u64, rows) * k * row_bytes;
    return .{ .send = s, .recv = s * world };
}

/// A prefill union block: rows [a, b) of the segment, m rows an owner (padded), this rank's local rows to send, the block's tokens.
pub const Block = struct {
    a: u32,
    b: u32,
    m: u32,
    phys: []u32,
    tokens: []i32,

    pub fn deinit(blk: *Block, gpa: std.mem.Allocator) void {
        gpa.free(blk.phys);
        gpa.free(blk.tokens);
    }
};

pub const UnionIn = struct {
    /// [rows, k] logical rows (-1: none), host
    sel: []const i32,
    k: u32,
    /// the slot's split table (local page by logical page), host
    table: []const u32,
    psh: u32,
    world: u32,
    rank: u32,
    row_bytes: u32,
    /// send + receive bytes a block may take (TF_DSV41_KV_SPLIT_UNION_MIB)
    cap: u64,
};

/// Ranks a split exchange plans for (the session pool's bound: sessions.pool.max_world).
pub const max_world = 8;

/// Plans a segment's union exchange: blocks of rows whose (W + 1) x M rows fit the cap; O(entries + rows), no sort.
pub fn planUnion(gpa: std.mem.Allocator, in: UnionIn, out: *std.ArrayList(Block)) !void {
    if (in.world < 1 or in.world > max_world or in.rank >= in.world) return error.BadWorld;
    const rows: u32 = @intCast(in.sel.len / in.k);
    // a row's mark, then its index + 1 in its owner's list (every row a selection names is in the table's pages)
    const seen = try gpa.alloc(u32, in.table.len << @intCast(in.psh));
    defer gpa.free(seen);
    try planRange(gpa, in, seen, 0, rows, out);
}

fn planRange(gpa: std.mem.Allocator, in: UnionIn, seen: []u32, a: u32, b: u32, out: *std.ArrayList(Block)) !void {
    const part = in.sel[@as(usize, a) * in.k .. @as(usize, b) * in.k];
    @memset(seen, 0);
    for (part) |t| if (t >= 0) {
        if (t >= seen.len) return error.RowOutsideTable;
        seen[@intCast(t)] = 1;
    };
    // each owner's rows in ascending order (the sorted unique union, partitioned by owner): a row's index is a count
    var sizes: [max_world]u32 = @splat(0);
    for (seen, 0..) |*x, t| if (x.* != 0) {
        const r = ownerOf(@intCast(t), in.psh, in.world);
        sizes[r] += 1;
        x.* = sizes[r];
    };
    var m: u32 = 1;
    for (sizes[0..in.world]) |sz| m = @max(m, sz);
    if (@as(u64, in.world + 1) * m * in.row_bytes > in.cap and b - a > 1) {
        const h = a + (b - a) / 2;
        try planRange(gpa, in, seen, a, h, out);
        return planRange(gpa, in, seen, h, b, out);
    }
    var blk: Block = .{ .a = a, .b = b, .m = m, .phys = try gpa.alloc(u32, m), .tokens = try gpa.alloc(i32, part.len) };
    errdefer blk.deinit(gpa);
    @memset(blk.phys, 0); // padding: any stored row
    const mask = (@as(u32, 1) << @intCast(in.psh)) - 1;
    for (seen, 0..) |x, t| if (x != 0 and ownerOf(@intCast(t), in.psh, in.world) == in.rank) {
        const u: u32 = @intCast(t);
        blk.phys[x - 1] = (in.table[u >> @intCast(in.psh)] << @intCast(in.psh)) | (u & mask);
    };
    for (part, blk.tokens) |t, *o| {
        if (t < 0) {
            o.* = -1;
            continue;
        }
        const u: u32 = @intCast(t);
        o.* = @intCast(ownerOf(u, in.psh, in.world) * m + seen[u] - 1);
    }
    try out.append(gpa, blk);
}

/// TF_DSV41_KV_SPLIT_COMPACT: off (0, dense), on (1: packed where every rank's transport takes device lengths and the shard fits), force
/// (packed on any transport: the bits are tested on any).
pub const Compact = enum { off, on, force };

pub const compact_env = "TF_DSV41_KV_SPLIT_COMPACT";

pub fn parseCompact(text: []const u8) error{BadCompactMode}!Compact {
    const v = std.mem.trim(u8, text, " \t");
    const eq = std.ascii.eqlIgnoreCase;
    if (v.len == 0 or eq(v, "0") or eq(v, "off") or eq(v, "no") or eq(v, "false")) return .off;
    if (eq(v, "1") or eq(v, "on") or eq(v, "yes") or eq(v, "true")) return .on;
    if (eq(v, "force")) return .force;
    return error.BadCompactMode;
}

/// The knob from the environment (unset: off).
pub fn compactMode() error{BadCompactMode}!Compact {
    const raw = std.mem.span(std.c.getenv(compact_env) orelse return .off);
    return parseCompact(raw) catch |e| {
        std.log.err("[tensorfold] {s}={s}: 0 / 1 / force", .{ compact_env, raw });
        return e;
    };
}

/// The exchange itself: kernels to fill the send buffer, the communicator's all-gather to fill the receive buffer.
pub const Exchange = struct {
    comm: tp.Collective,
    kernels: Kernels,
    compact: Compact = .off,
    /// Every rank's transport takes device lengths (`Collective.varAgreed`, asked once at `init` when compact is on).
    var_ok: bool = false,

    /// An exchange under `mode` (`compactMode()` for the knob). Every rank calls it alike, outside any graph: `on` asks the
    /// communicator's once-per-communicator agreement.
    pub fn init(comm: tp.Collective, kernels: Kernels, mode: Compact) !Exchange {
        return .{ .comm = comm, .kernels = kernels, .compact = mode, .var_ok = mode == .on and try comm.varAgreed() };
    }

    /// Whether a dense exchange of `nbytes`-byte shards (R x K x row bytes) goes packed: the same answer on every rank.
    pub fn packs(x: Exchange, nbytes: u64) bool {
        return x.compact == .force or (x.compact == .on and x.var_ok and x.comm.varFits(@intCast(nbytes)));
    }

    /// The window's exchange as the knob says: densePacked where `packs`, else dense (a.lens unused then).
    pub fn window(x: Exchange, a: PackArgs, recv: DevicePtr, stream: Stream) !void {
        if (x.packs(@as(u64, a.rows) * a.k * a.row_bytes)) return x.densePacked(a, recv, stream);
        return x.dense(a.dense(), recv, stream);
    }

    /// Packed: send [R x K, rb] (this rank's c_rank owned rows first), tokens and lens [W] by one kernel, then recv [W, R x K, rb] with
    /// owner o's c_o rows at o x R x K (persistent buffers, no host sync: graph-capturable; recv past each owner's rows unspecified).
    pub fn densePacked(x: Exchange, a: PackArgs, recv: DevicePtr, stream: Stream) !void {
        const f = x.kernels.vtable.pack orelse return error.Unsupported;
        try f(x.kernels.ptr, a, stream);
        try x.comm.allGatherV(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, a.lens, .u8, stream);
    }

    /// Dense: send [R x K, rb] gathered and tokens remapped, then recv [W, R x K, rb] (persistent buffers: graph-capturable).
    pub fn dense(x: Exchange, a: DenseArgs, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.dense(x.kernels.ptr, a, stream);
        try x.comm.allGather(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, .u8, stream);
    }

    /// One union block: this rank's M rows (phys: the block's list on the device) to send, recv [W, M, rb].
    pub fn unionBlock(x: Exchange, base: u64, row_bytes: u32, phys: DevicePtr, m: u32, send: DevicePtr, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.gather(x.kernels.ptr, base, row_bytes, phys, m, send, stream);
        try x.comm.allGather(send, recv, @as(usize, m) * row_bytes, .u8, stream);
    }
};
