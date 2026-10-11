//! A decode window on the prod kernel path (prod.env: dense3, x3seg, the CUDA mHC / attention / top-k, router GEMV, x3ld,
//! L2 prefetch), emitted as calls (calls.zig): every launch with its arguments and its tensors named by role, in the
//! Python engine's order. M1 checks the calls against the Python engine's capture launch by launch (structure on the
//! host, bits on the GPU); M2's forward issues the same calls.
//!
//! Roles: "s." persists (weights' run-time forms, KV pools, scratch), "w." lives for the window (streams, positions, mHC
//! coefficients, the exchange buffer, the index selections reuse layers read), "L." for one layer.

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const config = @import("config.zig");
const moe = @import("block_moe.zig");
const prefill = @import("block_prefill.zig");
const Config = config.Config;
const Arg = calls.Arg;
const Named = calls.Named;
const Dt = calls.Dt;
const dense = dk.dense;

/// The widths the loader picked: K2 of each dense group (by our weight prefix) and each routed layer's expert K2 range
/// (gate/up, down; the shared expert included). From the plan in a run; the M1 check reads the capture's.
pub const Widths = struct {
    dense: std.StringHashMapUnmanaged(u32) = .empty,
    experts: std.AutoHashMapUnmanaged(u32, [2][2]u32) = .empty,
    /// each routed layer's expert K2s present (bit k2 set), gate/up and down: a prefill segment's x3gm runs one launch
    /// a width present (x3gm.Ragged). From the plan in a run; the M1 check reads the capture's x3gm launches
    gm: std.AutoHashMapUnmanaged(u32, [2]u32) = .empty,

    pub fn k2(w: *const Widths, name: []const u8) !u32 {
        return w.dense.get(name) orelse error.NoWidth;
    }
};

/// Run-time sizes the Python engine fixes at load (its scratch and pools), and the prod L2 prefetch settings.
pub const Options = struct {
    /// TF_DSV41_IMAGES=native (vision_rows.zig): this prefill segment holds image positions, so Engram fuses with a keep
    /// vector (0 there; vision.window's ``keep``)
    image_keep: bool = false,
    /// TF_DSV41_BIAS_VL: the segment's image rows (their MoE call apart, routed with gate.bias_vl); 0: none; < 0: the
    /// buffer plan's (both calls at the segment's rows)
    image_rows: i64 = 0,
    /// TP ranks (a boot constant): every per-rank width below is the model's / world; the plan checked divisibility
    world: u32 = 2,
    /// KV positions a slot holds (pool rows = limit / ratio + 1)
    limit: i64 = 4096,
    /// rope table rows (cos / sin pairs of 32)
    rope_rows: i64 = 6144,
    /// rows the routed experts' scratch holds: the backbone's and a DSpark block's
    expert_rows: i64 = 1024,
    dspark_rows: i64 = 64,
    /// TF_DSV41_BRANCHES (branches.zig): the attention sublayer's indexer / compressor chains (and a non-index layer's
    /// SWA store) tagged for the side stream at branches.plan's fork / join points, windows of up to `branch_rows`
    branches: bool = false,
    branch_rows: i64 = 64,
    /// TF_DSV41_MHC_DEFER (with R1): R1's coefficient launch tagged for the deferred stream, the next mHC call joins it
    mhc_defer: bool = false,
    /// TF_DSV41_MHC_DEFER_AT=exchange (ours, with mhc_defer): each deferred launch issued at the next exchange
    /// (calls.holdDeferred); TF_DSV41_MHC_DEFER_AT_ROWS: only in programs of at most this many rows (0: every one)
    mhc_defer_hold: bool = false,
    /// TF_DSV41_COEF_LATE (ours, with MHC_DEFER; branches.zig): a deferred coefficient launch moves to just before the
    /// sublayer's exchange (still before its join), so `coef_kernel` runs beside the exchange's wait instead of on top
    /// of the one-wave router GEMV / first projection that follows its site; the same calls, arguments and data
    coef_late: bool = false,
    mhc_defer_hold_rows: i64 = 0,
    /// TF_DSV41_MHC_PFDEC (ours, 0 = off): mixing mHC sites of windows of at least this many rows (1-64) as mhc_pf's site
    /// + the normed input + mhc_cuda's coef_kernel (block_wide.zig); the same bits as the path they replace
    mhc_pf_rows: i64 = 0,
    /// TF_DSV41_L2PF_MB (bytes), _CHUNK_KB, _PACE_GBPS, _PACE_CTAS; the GPU's SMs (a segments launch's grid bound)
    l2pf_budget: u64 = 12 << 20,
    l2pf_chunk: i64 = 32768,
    l2pf_pace_gbps: f64 = 150,
    l2pf_pace_ctas: i64 = 2,
    sms: i64 = 188,
    /// TF_DSV41_EXPERT_TOPP (prod: 0.85; lossy). null: no pruning, and the router's tail (RG_TAIL) rounds the kit
    /// weights and groups the experts itself, as it does for a DSpark block (moe.py: rule None)
    expert_topp: ?f64 = 0.85,
    /// M2a: after each block, its streams with the block's last post applied out of place (Python's Forward.trace),
    /// handed to the glue step "trace" (one more launch a block; never in a served window)
    trace: bool = false,
    /// DSpark's taps (Forward(taps=True)): the boundaries entering the DSpark target layers (37-39) write the streams'
    /// mean into "w.taps" bf16 [n, 3 x hidden] (a column block a target, dspark.py), which draft/iface.zig's Taps names
    taps: bool = false,
    /// rows a prefill segment holds at most (forward.pchunk): sizes the SWA staging ring (block_prefill.zig)
    prefill_rows: i64 = 2048,
    /// TF_DSV41_PF_4K (G16 4K, pf4k.py): prefill segments up to 4,096 rows, x3gm's row block the whole segment (the
    /// expert scratch holds it, as pf4k's arena: the trellis streams once a segment, not once a 2,048 rows), the router
    /// in 2,048-row launches (router_gemv.CHUNK). Off: Python prod's 2,048 (a longer segment is refused)
    pf4k: bool = false,
    /// TF_DSV41_PF_OVERLAP (block_prefill.zig `pieces`): a prefill segment's two exchanges a layer in this many row
    /// pieces, each sent on the branches' side stream while the main stream computes the next piece's producer (wo_a /
    /// wo_b, the shared expert); 0 / 1: one exchange after its producer (Python's order). Needs `branches` and TP=2
    pf_overlap: i64 = 0,
    /// TF_DSV41_PF_OVERLAP_SITE (with pf_overlap): every exchange piece goes on the side stream (the last too), and the
    /// next mixing site (a boundary's mhc_pf.run + `_finish_k`, row-wise) runs in the same row pieces, piece p after
    /// exchange piece p only, so the site's early pieces hide the later pieces' exchanges
    pf_overlap_site: bool = false,
    /// TF_DSV41_MHC_SITE_ROWS (0 off; 256-4,096, a multiple of 16): a prefill boundary site (mhc_pf.run + `_finish_k`,
    /// row-wise) in pieces of about this many rows on the main stream, so `_finish_k` reads a piece's partials from
    /// L2; a site the exchange pieces already cut (pf_overlap_site) keeps theirs
    mhc_site_rows: i64 = 0,
    /// TF_DSV41_PF_TBO (block_prefill.emitTbo): a prompt's CED encoder segments two at a time, the second half a layer
    /// behind the first on the branches' side stream, so one's routed experts (DRAM-bound) run beside the other's
    /// attention and dense GEMMs (compute-bound). Needs `branches`; the exchange pieces (pf_overlap) are off under it
    pf_tbo: bool = false,
    /// TF_DSV41_STREAM_RB (1: `_stream_pf`, the next tile's key loads ahead; 2 / 4; 0 off): the long-context indexer's stream top-k (`_stream`, positions mode) as the
    /// Zig-own row-blocked twin `_stream_rb<N>` (tools/zig/dsv41_triton: N rows a program, each row `score_tile`'s
    /// arithmetic: the same split buffers, the same selection)
    stream_rb: i64 = 0,
    /// TF_DSV41_INDEX_BOUND (ours, prod_knobs.indexBound): a decode window's dense index scores as the Zig-own twin
    /// `_scores_b` (tools/zig/dsv41_triton/dsv41_zig_triton/scores_b.py): a (row, key tile) past the row's own visible
    /// keys stores `_scores`' -inf without its key loads and dot (a row window scores every row over the mix's bucket);
    /// and the decode window's top-k as the Zig-own twins `_dtopk_b` (tools/zig/dsv41_triton/dsv41_zig_triton/
    /// dtopk_b.py) and `tf_dsv41_topk_b_v1.topk` (zig/kernels/cuda/deepseek_v41/topk_b.cu): each row's selection
    /// stops at its own visible end (dense select: max(nvis, K); candidate blocks: the visible blocks), the same bytes
    index_bound: bool = false,
    /// R1 (prod-perf1's decode knobs, docs DSV41-DECODE-R1.md; the same bits): attention split 4 with q's RoPE in the
    /// kernel (ATTN_SPLIT, ATTN_ROPE), mHC's parallel tail + deferred coefficients (MHC_TAIL, MHC_DEFER), the router's
    /// prune + kit rounding (RG_PRUNE, PRUNE_KIT), a bf16 MoE partial (MOE_BF16), q_norm folded into its consumers'
    /// rot_in (RMS_FOLD)
    r1: bool = false,
    /// TF_DSV41_ROUTER_GROUP_ROT: wide grouping and expert input rotation in one launch (default off).
    router_group_rot: bool = false,
    /// TF_DSV41_X3LD_EPI (ours, prod_knobs.zig; default off): the decode MoE's x3ld + gateup_epilogue and x3ld +
    /// down_combine as two fused launches (x3ld_epi.cu: the epilogue in the streamer's tail behind a last-arrival
    /// ticket, the same arithmetic in the same order, so the same Xd / y / L.moe bits); ticket words "s.ex.epi" /
    /// "s.dx.epi" (zero between launches)
    x3ld_epi: bool = false,
    /// KV-only RMS feeding SWA store; separately gated, default off.
    kv_norm_store: bool = false,
    /// R1 off, but the bindings of a twin that always takes R1's trailing arguments (the midprofile twin, dba6a1f+):
    /// each call R1 extends gets them at their off values (mHC spin -1 / defer 0, the attention's split 0 / rope 0, the
    /// route's empty prune list, `_prune`'s KIT false). The same launches; the check reads it from a capture's calls
    r1_sig: bool = false,
    /// MHC_TAIL_SPIN (SM cycles a CTA waits for the last arrival)
    mhc_spin: i64 = 8000,
    /// M5: the KV families in the paged pool (kv.DevicePool, Python's serving slots) instead of the slot's contiguous
    /// buffers; null: contiguous (the gates' reference state)
    pool: ?Pool = null,
    /// TF_DSV41_PF_DENSE=fused (prod_knobs.zig): prefill's dense EXL3 GEMMs as pfdense's fused kernel; null: the W_q
    /// unpack + Triton `_gemm` (the captures' path). The same bits either way.
    pfd: ?*const @import("prod_knobs.zig").PfDense = null,
    /// TF_DSV41_GM_V2 (gm2pf.zig): prefill's routed experts on x3gm v2 (gm2_kernel); off: v1 (the captures' path). The
    /// same bits either way.
    gm_v2: @import("gm2pf.zig").Mode = .off,
    /// TF_DSV41_GM_GU2 (gm2pf.zig, with gm_v2 `one`): a 4,096-row segment's gate / up at the 128-member tile GU2 over
    /// its own 128-member plan; down keeps DN0 and the 64-member plan
    gm_gu2: bool = false,
    /// TF_DSV41_INDEX_BUDGET_MIB (bytes): the indexer's materialised scores + sort keys a launch (backend.index_budget:
    /// prefill segments and decode windows past it select in row blocks through `_keys`; the same bits)
    index_budget: i64 = 256 << 20,
    /// a row-mode program (batch.zig, rowmode.zig: Python's RowWin): every row block scores the context bucket's keys
    /// (RowWin.sub keeps `cap`) and no window streams its top-k (backend.select: `not win.rows`)
    rows: bool = false,

    /// TF_DSV41_MHC_DEFER_AT(_ROWS): a program of `rows` rows issues its deferred launches at the next exchange.
    pub fn holdsDeferred(o: Options, rows: i64) bool {
        return o.mhc_defer_hold and (o.mhc_defer_hold_rows == 0 or rows <= o.mhc_defer_hold_rows);
    }
};

/// The paged pool as a window's calls see it (pool.py, slots.py: one slot's 1-D table, PT / PSH a family).
/// - "s.kv.comp.L<i>": kv source i's comp rows, uint8 [rows, 584] (576 values then 8 scale bytes); split: this rank's
///   local pages, then the local null and discard pages;
/// - "s.kv.ik.L<i>": its index keys, uint8 [rows, 132], whole on every rank;
/// - "s.kv.pt": the slot's page table, int32 [pts]; "s.kv.ct": the split table (an owned logical page's local page,
///   any other the discard page), which the comp rows are stored through.
/// Split (TF_DSV41_KV_SPLIT): after each index layer's selection the selected rows come over an exchange (kv/split.zig:
/// dense in decode windows and short segments, a union in prefill segments) into "w.kx.recv", and the attention reads
/// them unpaged at the remapped tokens "w.kx.tok": the same row bytes in the same list order (split == replicated).
pub const Pool = struct {
    /// pages each family tensor holds (comp: split's local pages + null + discard)
    comp_pages: i64,
    ik_pages: i64,
    /// pages a slot's table holds
    pts: i64,
    split: bool = false,
    /// TF_DSV41_KV_SPLIT_EARLY: an index layer's exchange right after its selection (side stream), beside the q group
    early: bool = false,
    /// split's union cap (TF_DSV41_KV_SPLIT_UNION_MIB): the prefill receive buffer's bytes
    union_cap: i64 = 128 << 20,
    pub const page: i64 = 256;
    pub const row_bytes: i64 = 584;
    pub const ik_bytes: i64 = 132;

    /// log2 of a family's rows a page
    pub fn shift(ratio: i64) i64 {
        return std.math.log2_int(u64, @intCast(@divExact(page, ratio)));
    }
};

pub const Emitter = struct {
    a: std.mem.Allocator,
    cfg: *const Config,
    w: *const Widths,
    o: Options,
    /// rows this window and the position of its first
    n: i64,
    start: i64,
    out: std.ArrayList(calls.Call) = .empty,
    /// which of the two stream buffers holds the streams, and which coefficient set the last boundary wrote
    cur: u1 = 0,
    coef: u1 = 0,
    begin: calls.Begin = .window,
    /// the current layer's compression ratio (picks its rope table)
    layer_ratio: u32 = 0,
    /// TF_DSV41_BRANCHES: the stream the next calls go to, and the marks the next call carries
    br_side: bool = false,
    br_fork: bool = false,
    br_join: bool = false,
    br_mark: u8 = 0,
    br_wait: u8 = 0,
    /// TF_DSV41_MHC_DEFER: the next call is the coefficient launch / joins it; a coefficient launch not joined yet
    df_side: bool = false,
    df_join: bool = false,
    df_pending: bool = false,
    /// TF_DSV41_COEF_LATE: the index of a deferred coefficient launch not joined yet (moved to the next exchange)
    df_at: ?usize = null,
    /// a multi-segment prefill (block_prefill.emitMulti): the current segment's first row in the run's per-row roles
    /// (the selection's rows); 0 elsewhere
    row0: i64 = 0,
    /// TF_DSV41_PF_TBO (block_prefill.emitTbo): the call indexes where each layer's attention part and MoE part start
    marks: ?*std.ArrayList(usize) = null,
    /// its first row in the run's narrow per-row roles (the selection counts: 4 bytes a row), a multiple of 4 so every
    /// segment's slice is 16-byte aligned as Triton's specialization (and aot-needs) assume; 0 elsewhere
    cnt0: i64 = 0,

    pub fn fmt(e: *Emitter, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(e.a, f, args);
    }

    /// A contiguous tensor of `role`.
    pub fn t(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64) !Arg {
        const s = try e.a.dupe(i64, shape);
        return .{ .t = .{ .role = role, .dt = dt, .shape = s, .stride = try calls.contiguous(e.a, s) } };
    }

    /// A view of `role`: its own strides and a byte offset.
    pub fn view(e: *Emitter, role: calls.Role, dt: Dt, shape: []const i64, stride: []const i64, offset: i64) !Arg {
        return .{ .t = .{ .role = role, .dt = dt, .shape = try e.a.dupe(i64, shape), .stride = try e.a.dupe(i64, stride), .offset = offset } };
    }

    pub fn buf(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .buf = try e.fmt(f, args) }, dt, shape);
    }

    pub fn weight(e: *Emitter, comptime f: []const u8, args: anytype, dt: Dt, shape: []const i64) !Arg {
        return e.t(.{ .weight = try e.fmt(f, args) }, dt, shape);
    }

    pub fn empty(e: *Emitter, dt: Dt) !Arg {
        return e.t(.empty, dt, &.{0});
    }

    pub fn ints(e: *Emitter, xs: []const i64) !Arg {
        const l = try e.a.alloc(Arg, xs.len);
        for (xs, l) |x, *y| y.* = .{ .i = x };
        return .{ .list = l };
    }

    fn push(e: *Emitter, c: calls.Call) !void {
        var x = c;
        x.begin = e.begin;
        e.begin = .none;
        x.side = e.br_side;
        x.fork = e.br_fork;
        x.join = e.br_join;
        x.mark = e.br_mark;
        x.wait = e.br_wait;
        e.br_fork = false;
        e.br_join = false;
        e.br_mark = 0;
        e.br_wait = 0;
        x.defer_side = e.df_side;
        x.defer_join = e.df_join;
        e.df_side = false;
        e.df_join = false;
        if (e.o.coef_late and e.marks == null) try e.lateCoef(&x);
        try e.out.append(e.a, x);
    }

    /// TF_DSV41_COEF_LATE: a deferred coefficient launch (`coef_kernel`, read only by the next mHC call, which joins
    /// it) leaves its place after the site and goes just before the next exchange of the same fork region: the main
    /// calls between them touch none of its roles (L.mhc.part, the weights, the next coefficient set), so every launch
    /// sees the same data. A join before any exchange leaves it where it was. The exchange's spin leaves the SMs the
    /// one-wave router GEMV (or the first projection) would otherwise share with it.
    fn lateCoef(e: *Emitter, x: *const calls.Call) !void {
        if (x.defer_join) e.df_at = null;
        if (x.defer_side) {
            e.df_at = e.out.items.len;
            return;
        }
        const at = e.df_at orelse return;
        if (!(x.glue and std.mem.startsWith(u8, x.name, "glue.exchange"))) return;
        e.df_at = null;
        const c = e.out.orderedRemove(at);
        std.debug.assert(c.defer_side and c.begin == .none);
        try e.out.append(e.a, c);
    }

    /// Appends positional arguments to the last emitted call (R1's trailing arguments).
    pub fn appendLast(e: *Emitter, args: []const Arg) !void {
        const last = &e.out.items[e.out.items.len - 1];
        const more = try e.a.alloc(calls.Named, last.args.len + args.len);
        @memcpy(more[0..last.args.len], last.args);
        for (more[last.args.len..], args) |*m, x| m.* = .{ .arg = x };
        last.args = more;
    }

    pub fn ext(e: *Emitter, name: []const u8, args: []const Arg) !void {
        const l = try e.a.alloc(Named, args.len);
        for (args, l) |x, *y| y.* = .{ .arg = x };
        try e.push(.{ .triton = false, .name = name, .args = l });
    }

    /// A glue step ("glue.<name>"): the roles it reads / writes and its scalars, for the forward to run.
    pub fn glue(e: *Emitter, name: []const u8, args: []const Arg) !void {
        const l = try e.a.alloc(Named, args.len);
        for (args, l) |x, *y| y.* = .{ .arg = x };
        try e.push(.{ .triton = false, .glue = true, .name = try e.fmt("glue.{s}", .{name}), .args = l });
    }

    pub fn triton(e: *Emitter, name: []const u8, grid: [3]i64, args: []const Named) !void {
        try e.push(.{ .triton = true, .name = name, .grid = grid, .args = try e.a.dupe(Named, args) });
    }

    /// The Python engine's float of a config value: the JSON number (f64), not the f32 we keep.
    pub fn dec(x: f32) f64 {
        var b: [64]u8 = undefined;
        const s = std.fmt.bufPrint(&b, "{e}", .{x}) catch return x;
        return std.fmt.parseFloat(f64, s) catch x;
    }

    /// Positions after this window: the last row's + 1.
    pub fn end(e: *const Emitter) i64 {
        return e.start + e.n;
    }

    pub fn streams(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    pub fn spare(e: *Emitter) !Arg {
        return e.buf("w.x{d}", .{1 - e.cur}, .bf16, &.{ e.n, 4 * @as(i64, e.cfg.hidden) });
    }

    // -----------------------------------------------------------------------------------------------------------
    // dense EXL3 projections

    /// A trellis's run-time words (int32 [N/128, K/16, 8, 4 K2]): upstream's strips "s.<weight>.T", or dense3's lanes
    /// layout of them "s.<weight>.lanes" (prepare.strips, prepare.lanes).
    pub fn strips(e: *Emitter, w: []const u8, k: i64, n: i64) !Arg {
        return e.words(w, k, n, "T");
    }

    fn words(e: *Emitter, w: []const u8, k: i64, n: i64, form: []const u8) !Arg {
        const k2: i64 = try e.w.k2(w);
        return e.t(.{ .buf = try e.fmt("s.{s}.{s}", .{ w, form }) }, .i32, &.{ @divExact(n, 128), @divExact(k, 16), 8, 4 * k2 });
    }

    /// Bytes of a group's trellis (= its strips).
    pub fn trellisBytes(e: *Emitter, w: []const u8, k: i64, n: i64) !u64 {
        const k2: u64 = try e.w.k2(w);
        return @as(u64, @intCast(@divExact(k, 16) * @divExact(n, 16))) * 8 * k2 * 2;
    }

    /// One projection of a fused group: weight prefix, input, output (a whole buffer; `col` its first column), K x N.
    pub const Proj = struct { w: []const u8, x: Arg, y: Arg, col: usize = 0, k: usize, n: usize };

    /// R1's RMS_FOLD: rot_in norms its rows itself (rms(x, w, eps), rmsnorm's order) before the rotation.
    pub const Norm = struct { w: Arg, eps: f64 };

    /// x3seg rot_in, then dense3 a width (fused_proj.run): the group's projections in one rot_in launch, one linear
    /// launch for each K2 in first-appearance order. `tag` names the group's scratch.
    pub fn fused(e: *Emitter, tag: []const u8, ps: []const Proj) !void {
        return e.fusedNorm(tag, ps, null);
    }

    pub fn fusedNorm(e: *Emitter, tag: []const u8, ps: []const Proj, norm: ?Norm) !void {
        const m: usize = @intCast(e.n);
        const segs = try e.a.alloc(dense.Seg, ps.len);
        for (ps, segs, 0..) |p, *s, i| {
            const pl = dense.plan(p.k, p.n);
            s.* = .{ .k = p.k, .n = p.n, .k2 = try e.w.k2(p.w), .sk = pl[0], .wk = pl[1], .out = i, .col = p.col };
        }
        const lay = try dense.layout(segs, m);
        const xs = try e.a.alloc(Arg, ps.len);
        const suh = try e.a.alloc(Arg, ps.len);
        const offs = try e.a.alloc(i64, ps.len);
        for (ps, xs, suh, offs, 0..) |p, *x, *s, *o, i| {
            x.* = p.x;
            s.* = try e.weight("{s}.suh", .{p.w}, .f16, &.{@intCast(p.k)});
            o.* = @intCast(lay.xh_off[i]);
        }
        const xh = try e.buf("L.{s}.xh", .{tag}, .f16, &.{@intCast(lay.xh_total)});
        if (norm) |nm| {
            const ws = try e.a.alloc(Arg, ps.len);
            @memset(ws, nm.w);
            try e.ext("tf_dsv41_x3seg_v3.rot_in", &.{ .{ .list = xs }, .{ .list = suh }, xh, try e.ints(offs), .{ .b = false }, .{ .list = ws }, .{ .f = nm.eps } });
        } else try e.ext("tf_dsv41_x3seg_v3.rot_in", &.{ .{ .list = xs }, .{ .list = suh }, xh, try e.ints(offs), .{ .b = false } });
        const z: Arg = if (lay.z_total > 0) try e.buf("L.{s}.z", .{tag}, .f32, &.{@intCast(lay.z_total)}) else .none;
        const counters = try e.buf("L.{s}.cnt", .{tag}, .i32, &.{@intCast(lay.c_total)});
        for (lay.launches[0..lay.n_launches]) |l| {
            const T = try e.a.alloc(Arg, l.count);
            const svh = try e.a.alloc(Arg, l.count);
            const ys = try e.a.alloc(Arg, l.count);
            var meta: std.ArrayList(i64) = .empty;
            for (l.segs[0..l.count], 0..) |si, j| {
                const s = segs[si];
                const p = ps[si];
                T[j] = try e.words(p.w, @intCast(p.k), @intCast(p.n), "lanes");
                svh[j] = try e.weight("{s}.svh", .{p.w}, .f16, &.{@intCast(p.n)});
                ys[j] = p.y;
                const st = dense.strides(s.k, s.k2);
                try meta.appendSlice(e.a, &.{ @intCast(s.k), @intCast(s.n), @intCast(s.sk), @intCast(s.wk), st[0], st[1], @intCast(lay.xh_off[si]), @intCast(lay.z_off[si]), @intCast(lay.c_off[si]), @intCast(s.col), @intCast(l.p0[j]) });
            }
            try e.ext("tf_dsv41_dense3_v1.linear", &.{
                xh,                    .{ .list = T },                 .{ .list = svh },          .{ .list = ys },
                z,                     counters,                       try e.ints(meta.items),    .{ .i = @intCast(l.programs) },
                .{ .i = l.k2 },        .{ .i = @intCast(l.wk) },       .{ .i = if (l.wk == 8) 1 else 3 }, .{ .b = false },
            });
        }
    }

    /// Upstream's rot_in + linear (exl3 linear_v4): one projection, its trellis as strips unless one block of 128
    /// columns (then the stored trellis is the strips).
    pub fn upstream(e: *Emitter, w: []const u8, x: Arg, k: i64, n: i64, y: Arg) !void {
        const k2: i64 = try e.w.k2(w);
        const xh = try e.buf("L.{s}.xh", .{w}, .f16, &.{ e.n, k });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
        const pl = dense.plan(@intCast(k), @intCast(n));
        const st = dense.strides(@intCast(k), @intCast(k2));
        const T = if (n == 128)
            try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, 4 * k2 }, &.{ 32 * k2, 32 * k2, 4 * k2, 1 }, 0)
        else
            try e.strips(w, k, n);
        if (pl[0] != 1) return error.Unsupported; // no upstream linear on the prod decode path splits K
        try e.ext("tensorfold_exl3_linear_v4.linear", &.{
            xh, T, .{ .i = st[0] }, .{ .i = st[1] }, try e.weight("{s}.svh", .{w}, .f16, &.{n}), .none, y, .none,
            try e.buf("L.{s}.cnt", .{w}, .i32, &.{8 * @divExact(n, 128)}),
            .{ .i = k2 }, .{ .i = 2 }, .{ .i = @intCast(pl[0]) }, .{ .i = @intCast(pl[1]) },
        });
    }

    // -----------------------------------------------------------------------------------------------------------
    // L2 prefetch (l2pf.py): the table rows; l2pf.zig fills the tables with the addresses of `l2pfSources` at boot
    // (TF_DSV41_L2PF) and the runner issues these launches on its side stream, else skips them

    /// What a site prefetches, in the order the launches after it read it (l2pf.site_tensors): a weight, a projection's
    /// trellis words as its linear reads them ("s.<w>.lanes", else the stored "<w>.trellis"), or the trellis an
    /// expert table's entry `index` points at (the shared expert's, the last).
    pub const L2Src = struct { kind: enum { weight, words, expert }, name: []const u8, index: u32 = 0, bytes: u64 };
    pub const L2Site = enum { o, f, x };

    /// l2pf._lin_tensors: every projection's suh and svh, then their trellis words.
    fn linSrcs(e: *Emitter, out: *std.ArrayList(L2Src), ws: []const []const u8, ks: []const i64, ns: []const i64) !void {
        for (ws, ks, ns) |w, k, n| try out.appendSlice(e.a, &.{
            .{ .kind = .weight, .name = try e.fmt("{s}.suh", .{w}), .bytes = @intCast(2 * k) },
            .{ .kind = .weight, .name = try e.fmt("{s}.svh", .{w}), .bytes = @intCast(2 * n) },
        });
        for (ws, ks, ns) |w, k, n| try out.append(e.a, .{ .kind = .words, .name = w, .bytes = try e.trellisBytes(w, k, n) });
    }

    /// Site `site`'s sources: "o" and "f" of block `L`, "x" of the block before `L` (its table is "s.L<L>.pf.x").
    pub fn l2pfSources(e: *Emitter, L: u32, site: L2Site) ![]const L2Src {
        var out: std.ArrayList(L2Src) = .empty;
        switch (site) {
            .o => {
                const D: i64 = e.cfg.hidden;
                const g = e.cfg.o_groups / e.o.world;
                const ki: i64 = @divExact(@as(i64, e.cfg.heads) * e.cfg.head_dim, @as(i64, e.cfg.o_groups));
                const ws = try e.a.alloc([]const u8, g);
                const ks = try e.a.alloc(i64, g);
                const ns = try e.a.alloc(i64, g);
                for (ws, ks, ns, 0..) |*w, *k, *n, i| {
                    w.* = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i });
                    k.* = ki;
                    n.* = e.cfg.o_lora;
                }
                try e.linSrcs(&out, ws, ks, ns);
                try e.linSrcs(&out, &.{try e.fmt("L{d}.attn.wo_b", .{L})}, &.{@as(i64, g) * e.cfg.o_lora}, &.{D});
            },
            .f => {
                const D: i64 = e.cfg.hidden;
                const I: i64 = e.cfg.expert_width / e.o.world;
                const ex = e.cfg.expertsOf(L);
                const sh = try e.fmt("L{d}.moe.shared.0.w1", .{L});
                const k2: u64 = try e.w.k2(sh);
                const shared: u64 = @as(u64, @intCast(@divExact(D, 16) * @divExact(I, 16))) * 8 * k2 * 2;
                try out.appendSlice(e.a, &.{
                    .{ .kind = .weight, .name = try e.fmt("L{d}.moe.gate", .{L}), .bytes = @as(u64, ex.count) * @as(u64, @intCast(D)) * 2 },
                    .{ .kind = .expert, .name = try e.fmt("s.L{d}.ex.tp_g", .{L}), .index = ex.count, .bytes = shared },
                    .{ .kind = .expert, .name = try e.fmt("s.L{d}.ex.tp_u", .{L}), .index = ex.count, .bytes = shared },
                });
            },
            .x => {
                const xg = try e.xGroup(L);
                try e.linSrcs(&out, xg.ws, xg.ks, xg.ns);
                const qg = try e.qGroup(L);
                try e.linSrcs(&out, qg.ws, qg.ks, qg.ns);
            },
        }
        return out.items;
    }

    fn srcSizes(e: *Emitter, srcs: []const L2Src) ![]const u64 {
        const out = try e.a.alloc(u64, srcs.len);
        for (srcs, out) |x, *y| y.* = x.bytes;
        return out;
    }

    /// l2pf.take: spans of `sizes` in order within the budget, the last one's head.
    fn take(e: *const Emitter, sizes: []const u64) i64 {
        var left = e.o.l2pf_budget;
        var rows: i64 = 0;
        for (sizes) |s| {
            if (left < 16) break;
            const n = @min(left, s) / 16 * 16;
            if (n == 0) continue;
            rows += 1;
            left -= n;
        }
        return rows;
    }


    /// Site "o" (after wq_b): wo_a's groups, then wo_b; l2pf segments.
    pub fn prefetchO(e: *Emitter, L: u32) !void {
        const sizes = try e.srcSizes(try e.l2pfSources(L, .o));
        var total: u64 = 0;
        var left = e.o.l2pf_budget;
        for (sizes) |s| {
            if (left < 16) break;
            const n = @min(left, s) / 16 * 16;
            total += n;
            left -= n;
        }
        const pieces = std.math.divCeil(u64, total, @intCast(e.o.l2pf_chunk)) catch 1;
        const grid: i64 = @max(1, @min(e.o.sms, @as(i64, @intCast(std.math.divCeil(u64, pieces, 128) catch 1))));
        const rows = e.take(sizes);
        try e.ext("tf_dsv41_l2pf_v1.segments", &.{ .{ .opaque_table = (try e.buf("s.L{d}.pf.o", .{L}, .i64, &.{ rows, 2 })).t }, .{ .i = grid }, .{ .i = 128 }, .{ .i = e.o.l2pf_chunk }, .{ .b = true } });
    }

    fn paced(e: *Emitter, role: []const u8, rows: i64) !void {
        const ns: i64 = @max(1, @as(i64, @intFromFloat(@round(@as(f64, @floatFromInt(e.o.l2pf_chunk)) / e.o.l2pf_pace_gbps))));
        try e.ext("tf_dsv41_l2pace_v1.paced", &.{ .{ .opaque_table = (try e.buf("{s}", .{role}, .i64, &.{ rows, 2 })).t }, .{ .i = e.o.l2pf_pace_ctas }, .{ .i = e.o.l2pf_chunk }, .{ .i = ns }, .{ .i = 0 } });
    }

    /// Site "f" (after the attention sublayer): the router's gate, then the shared expert's gate and up trellises.
    pub fn prefetchF(e: *Emitter, L: u32) !void {
        try e.paced(try e.fmt("s.L{d}.pf.f", .{L}), e.take(try e.srcSizes(try e.l2pfSources(L, .f))));
    }

    /// Site "x" of the block before `nxt` (after its MoE): `nxt`'s x group, then its q group.
    pub fn prefetchX(e: *Emitter, nxt: u32) !void {
        try e.paced(try e.fmt("s.L{d}.pf.x", .{nxt}), e.take(try e.srcSizes(try e.l2pfSources(nxt, .x))));
    }

    const Group = struct { ws: []const []const u8, ks: []const i64, ns: []const i64 };

    /// The x group (fused_proj): wq_a, wkv, then a KV source's compressor projections.
    fn xGroup(e: *Emitter, L: u32) !Group {
        const D: i64 = e.cfg.hidden;
        const ratio = e.cfg.compressRatio(L);
        const comp: usize = if (e.cfg.isKvSource(L)) (if (ratio == 2) 2 else 1) else 0;
        const ws = try e.a.alloc([]const u8, 2 + comp);
        const ns = try e.a.alloc(i64, 2 + comp);
        const names = [_][]const u8{ "wq_a", "wkv", "comp_wkv", "comp_wgate" };
        for (ws, ns, 0..) |*w, *n, i| {
            w.* = try e.fmt("L{d}.attn.{s}", .{ L, names[i] });
            n.* = if (i == 0) e.cfg.q_lora else e.cfg.head_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, D);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    /// The q group: wq_b, and an indexer's ix_wq_b.
    fn qGroup(e: *Emitter, L: u32) !Group {
        const ix = switch (e.cfg.mode(L)) {
            .full, .reindex => true,
            else => false,
        };
        const ws = try e.a.alloc([]const u8, if (ix) 2 else 1);
        const ns = try e.a.alloc(i64, ws.len);
        ws[0] = try e.fmt("L{d}.attn.wq_b", .{L});
        ns[0] = @as(i64, e.cfg.heads / e.o.world) * e.cfg.head_dim;
        if (ix) {
            ws[1] = try e.fmt("L{d}.attn.ix_wq_b", .{L});
            ns[1] = @as(i64, e.cfg.index_heads) * e.cfg.index_dim;
        }
        const ks = try e.a.alloc(i64, ws.len);
        @memset(ks, e.cfg.q_lora);
        return .{ .ws = ws, .ks = ks, .ns = ns };
    }

    // -----------------------------------------------------------------------------------------------------------
    // attention

    fn rope(e: *Emitter, x: Arg, heads: i64, dim: i64, nope: i64) !void {
        const n = e.n;
        const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
        try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
            .{ .name = "X", .arg = xv },                               .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
            .{ .name = "x_hs", .arg = .{ .i = dim } },                 .{ .name = "OUT", .arg = xv },
            .{ .name = "o_rs", .arg = .{ .i = heads * dim } },         .{ .name = "o_hs", .arg = .{ .i = dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },              .{ .name = "cs_rs", .arg = .{ .i = 64 } },
            .{ .name = "POS", .arg = try e.buf("w.pos64", .{}, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
            .{ .name = "NOPE", .arg = .{ .i = nope } },                .{ .name = "NB", .arg = .{ .i = 64 } },
            .{ .name = "HALF", .arg = .{ .i = 32 } },                  .{ .name = "BH", .arg = .{ .i = 4 } },
            .{ .name = "INV", .arg = .{ .b = false } },                .{ .name = "COPY", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The rope table (cos / sin) a layer uses: the compressed layers' (compress_theta) or the SWA layers'.
    pub fn ropeTable(e: *Emitter) !Arg {
        return e.buf("s.rope.{s}", .{if (e.layer_ratio > 0) "comp" else "main"}, .f32, &.{ e.o.rope_rows, 64 });
    }

    fn pos(e: *Emitter) !Arg {
        return e.buf("w.pos", .{}, .i32, &.{1});
    }

    /// The KV store (_kv_store): the SWA ring (ratio 0) or a compressed family (contiguous, or the pool's rows through
    /// the slot's table: the split table under split KV, so a rank keeps only the rows it owns).
    pub fn kvStore(e: *Emitter, lat: Arg, src: u32, ratio: i64) !void {
        const rows = try e.pool(src, ratio);
        const pg = try e.paging(ratio, true);
        try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
            .{ .name = "LAT", .arg = lat },                          .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },             .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "V", .arg = rows[0] },                        .{ .name = "S", .arg = rows[1] },
            .{ .name = "v_stride", .arg = .{ .i = pg.vs } },         .{ .name = "s_stride", .arg = .{ .i = pg.ss } },
            .{ .name = "POS", .arg = try e.pos() },                  .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "RING", .arg = .{ .i = if (ratio == 0) 2 * @as(i64, e.cfg.window) else 1 } }, .{ .name = "PT", .arg = pg.pt },
            .{ .name = "PSH", .arg = .{ .i = pg.psh } },             .{ .name = "SL", .arg = .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                  .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// A family's paging: the slot's table and rows-a-page shift (pool mode), with the row strides (pool rows are 584 B
    /// records); none for the SWA ring and the contiguous slot. `store`: the comp rows' writes (the split table).
    pub const Paging = struct { pt: Arg, psh: i64, vs: i64, ss: i64 };

    pub fn paging(e: *Emitter, ratio: i64, store: bool) !Paging {
        const p = e.o.pool orelse return .{ .pt = .none, .psh = 0, .vs = 576, .ss = 8 };
        if (ratio == 0) return .{ .pt = .none, .psh = 0, .vs = 576, .ss = 8 };
        const tab: []const u8 = if (store and p.split) "s.kv.ct" else "s.kv.pt";
        return .{ .pt = try e.buf("{s}", .{tab}, .i32, &.{p.pts}), .psh = Pool.shift(ratio), .vs = Pool.row_bytes, .ss = Pool.row_bytes };
    }

    /// A layer's KV rows (fp8 latent + rope, 576 B) and scales (8 B): the SWA ring (ratio 0), the slot's contiguous
    /// compressed rows, or the pool's family tensor (584 B records, paged).
    pub fn pool(e: *Emitter, L: u32, ratio: i64) ![2]Arg {
        if (ratio > 0) if (e.o.pool) |p| {
            const rows = p.comp_pages * @divExact(Pool.page, ratio);
            const role: calls.Role = .{ .buf = try e.fmt("s.kv.comp.L{d}", .{L}) };
            return .{ try e.view(role, .u8, &.{ rows, 576 }, &.{ Pool.row_bytes, 1 }, 0), try e.view(role, .u8, &.{ rows, 8 }, &.{ Pool.row_bytes, 1 }, 576) };
        };
        const rows: i64 = if (ratio == 0) 2 * @as(i64, e.cfg.window) else @divFloor(e.o.limit, ratio) + 1;
        const kind = if (ratio == 0) "swa" else "comp";
        return .{ try e.buf("s.L{d}.{s}.v", .{ L, kind }, .u8, &.{ rows, 576 }), try e.buf("s.L{d}.{s}.s", .{ L, kind }, .u8, &.{ rows, 8 }) };
    }

    pub fn indexKeys(e: *Emitter, L: u32) !Arg {
        const r: i64 = e.cfg.compressRatio(L);
        if (e.o.pool) |p| return e.buf("s.kv.ik.L{d}", .{L}, .u8, &.{ p.ik_pages * @divExact(Pool.page, r), Pool.ik_bytes });
        return e.buf("s.L{d}.ik", .{L}, .u8, &.{ @divFloor(e.o.limit, r) + 1, @as(i64, e.cfg.index_dim) + 4 });
    }

    /// Split KV: kv source `src`'s rows of selection `sel` [n, K] from their owners (kv/split.zig, forward's glue): dense
    /// (fixed shapes) up to 16 rows, else the segment's union (host-planned, under the union cap). The attention reads
    /// the received rows unpaged at the remapped tokens.
    pub const Received = struct { cv: Arg, cs: Arg, tok: Arg };

    /// The rows and tokens of the last exchange (a reuse layer: the same selection and kv source).
    pub fn received(e: *Emitter, src: u32, ratio: i64) !Received {
        _ = src;
        _ = ratio;
        const p = e.o.pool.?;
        const n = e.n;
        const K: i64 = e.cfg.index_topk;
        const W: i64 = e.o.world;
        const sent: i64 = if (n <= 16) n * K else @divFloor(p.union_cap, (W + 1) * Pool.row_bytes);
        const recv: calls.Role = .{ .buf = "w.kx.recv" };
        return .{
            .cv = try e.view(recv, .u8, &.{ W * sent, 576 }, &.{ Pool.row_bytes, 1 }, 0),
            .cs = try e.view(recv, .u8, &.{ W * sent, 8 }, &.{ Pool.row_bytes, 1 }, 576),
            .tok = try e.buf("w.kx.tok", .{}, .i32, &.{ n, K }),
        };
    }

    pub fn exchange(e: *Emitter, src: u32, ratio: i64, sel: Arg) !Received {
        const p = e.o.pool.?;
        const n = e.n;
        const K: i64 = e.cfg.index_topk;
        const W: i64 = e.o.world;
        // row windows exchange dense at any rows (kvsplit.py attend_rows: `win.rows or win.n <= dense_max`)
        const small = n <= 16 or e.o.rows;
        // dense: every rank sends R x K rows; union: (W + 1) x M rows under the cap
        const sent: i64 = if (small) n * K else @divFloor(p.union_cap, (W + 1) * Pool.row_bytes);
        const recv = try e.buf("w.kx.recv", .{}, .u8, &.{ W * sent, Pool.row_bytes });
        const tok = try e.buf("w.kx.tok", .{}, .i32, &.{ n, K });
        const comp = try e.buf("s.kv.comp.L{d}", .{src}, .u8, &.{ p.comp_pages * @divExact(Pool.page, ratio), Pool.row_bytes });
        try e.glue(if (small) "kx_dense" else "kx_union", &.{
            sel,                                                 comp,
            try e.buf("s.kv.ct", .{}, .i32, &.{p.pts}),          try e.buf("w.kx.send", .{}, .u8, &.{ sent, Pool.row_bytes }),
            recv,                                                tok,
            .{ .i = Pool.shift(ratio) },
        });
        return .{
            .cv = try e.view(recv.t.role, .u8, &.{ W * sent, 576 }, &.{ Pool.row_bytes, 1 }, 0),
            .cs = try e.view(recv.t.role, .u8, &.{ W * sent, 8 }, &.{ Pool.row_bytes, 1 }, 576),
            .tok = tok,
        };
    }

    /// attn_cuda's MAX_EPT: past 8 CTAs x 512 threads x 31 entries a selection runs Python's Triton dtopk instead.
    const topk_max_ept: i64 = 31;

    /// topk.plan: (cluster CTAs, entries a thread) for `nk` entries.
    fn topkPlan(nk: i64, cl_in: ?i64) [2]i64 {
        const cl = cl_in orelse @max(1, @min(8, std.math.divCeil(i64, nk, 4096) catch 1));
        var ept = @max(1, std.math.divCeil(i64, nk, cl * 512) catch 1);
        ept += 1 - @mod(ept, 2);
        return .{ cl, ept };
    }

    /// The window's index selection of layer `src` (reuse layers read the source's).
    fn selection(e: *Emitter, src: u32) ![2]Arg {
        const k: i64 = e.cfg.index_topk;
        return .{
            try e.view(.{ .buf = try e.fmt("w.sel{d}", .{src}) }, .i32, &.{ e.n, k }, &.{ k, 1 }, e.row0 * k * 4),
            // the counts' 4-byte rows: a run's segment from a row a multiple of 4 (cnt0), so the slice is 16-aligned
            try e.view(.{ .buf = try e.fmt("w.cnt{d}", .{src}) }, .i32, &.{e.n}, &.{1}, e.cnt0 * 4),
        };
    }

    /// attn_cuda's ticket words: a (row, head tile)'s ranks, 4 with R1's split kernel (MAX_SPLIT); the twins with
    /// R1's bindings size it for the split kernel whatever the knob (`r1_sig`)
    pub fn ticketLen(e: *const Emitter) i64 {
        return if (e.o.r1 or e.o.r1_sig) 1024 else 256;
    }

    /// attn_cuda.split: CTAs a chunk with R1 (ATTN_SPLIT 4 up to ATTN_SPLIT_ROWS = 2 rows, 1 above)
    pub fn attnSplit(n: i64) i64 {
        return if (n <= 2) 4 else 1;
    }

    /// Triton `_rms` (rmsnorm.rms: a weight, bf16 in and out): R1's kv_norm beside the SWA store.
    pub fn rms(e: *Emitter, x: Arg, w: Arg, out: Arg, k: i64) !void {
        try e.triton("_rms", .{ e.n, 1, 1 }, &.{
            .{ .name = "X", .arg = x },                                                .{ .name = "x_rs", .arg = .{ .i = k } },
            .{ .name = "W", .arg = w },                                                .{ .name = "OUT", .arg = out },
            .{ .name = "o_rs", .arg = .{ .i = k } },                                   .{ .name = "K", .arg = .{ .i = k } },
            .{ .name = "inv_k", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(k)) } }, .{ .name = "eps", .arg = .{ .f = dec(e.cfg.eps) } },
            .{ .name = "BK", .arg = .{ .i = 512 } },                                   .{ .name = "HAS_W", .arg = .{ .b = true } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                              .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// Only the 512-wide KV producer belongs to this fusion; other RMS norms keep their calls.
    pub fn kvNormStoreOn(e: *const Emitter) bool {
        return e.o.kv_norm_store and e.cfg.head_dim == 512 and e.n >= 1 and e.n <= 64;
    }

    pub fn normStore(e: *Emitter, x: Arg, w: Arg, v: Arg, s: Arg, p: Arg, sl: Arg, ring: i64, rows: bool) !void {
        try e.ext("tf_dsv41_kv_glue_v1.norm_store", &.{
            x, w, try e.ropeTable(), v, s, p, sl,
            .{ .i = 512 }, .{ .i = 64 }, .{ .i = 576 }, .{ .i = 8 },
            .{ .f = 1.0 / 512.0 }, .{ .f = dec(e.cfg.eps) }, .{ .i = ring }, .{ .b = rows },
        });
    }

    /// The SWA store of the window's kv rows (R1: kv_norm runs here, RMS_FOLD's `Normed.tensor()`).
    fn swaStore(e: *Emitter, L: u32, kv: Arg, kvn: Arg) !void {
        if (e.o.r1 and e.kvNormStoreOn()) {
            const rows = try e.pool(L, 0);
            try e.normStore(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{e.cfg.head_dim}), rows[0], rows[1], try e.pos(), .none, 2 * @as(i64, e.cfg.window), false);
        } else {
            if (e.o.r1) try e.rms(kv, try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{e.cfg.head_dim}), kvn, e.cfg.head_dim);
            try e.kvStore(kvn, L, 0);
        }
    }

    /// branches.plan's marks: `fork` (the next calls on the side stream, after everything main issued), `onMain`,
    /// `join` (the next call on main after everything the side issued); nothing unless TF_DSV41_BRANCHES branches this
    /// window
    fn branching(e: *const Emitter) bool {
        return e.o.branches and e.n <= e.o.branch_rows;
    }
    fn fork(e: *Emitter) void {
        if (!e.branching()) return;
        e.br_fork = true;
        e.br_side = true;
    }
    fn onMain(e: *Emitter) void {
        e.br_side = false;
    }
    fn join(e: *Emitter) void {
        if (!e.branching()) return;
        e.br_join = true;
        e.br_side = false;
    }

    pub fn attention(e: *Emitter, L: u32) !void {
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const md = e.cfg.mode(L);
        const ratio: i64 = e.cfg.compressRatio(L);
        const kv_src = e.cfg.kvSource(L);
        const H: i64 = e.cfg.heads / e.o.world;
        const hd: i64 = e.cfg.head_dim;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const indexer = md == .full or md == .reindex;
        const own_kv = md == .full;
        // the indexer's head weights (a bf16 GEMV over the normed input)
        const iw = try e.buf("L.ix_w", .{}, .bf16, &.{ n, IH });
        // branches.plan (index layer): fork, side ix_wp, main xproj (+ rms2), fork, side [compress] ix_wq_b rope_qi select,
        // main wq_b l2pf_o rope_q kv_swa, join, attn; other layers: xproj, rms2, fork, side kv_swa, main wq_b l2pf_o
        // rope_q, join, attn
        if (indexer) e.fork();
        if (indexer) try e.triton("_plain", .{ n, IH, 1 }, &.{
            .{ .name = "X", .arg = x },                                                      .{ .name = "x_rs", .arg = .{ .i = D } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
            .{ .name = "OUT", .arg = iw },                                                   .{ .name = "K", .arg = .{ .i = D } },
            .{ .name = "N", .arg = .{ .i = IH } },                                           .{ .name = "BK", .arg = .{ .i = 256 } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                      .{ .name = "BF16", .arg = .{ .b = true } },
        });
        e.onMain();
        // x group: wq_a, wkv (+ the compressor's on a KV source)
        const xg = try e.xGroup(L);
        const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
        const cw: i64 = if (ratio == 2) 2 * hd else hd;
        // a KV source's compressor projection outlives its layer: keep() carries the accepted row (ratio 2)
        const comp = if (ratio == 2) try e.buf("w.L{d}.comp", .{L}, .bf16, &.{ n, cw }) else try e.buf("L.comp", .{}, .bf16, &.{ n, cw });
        var ps: [4]Proj = undefined;
        for (xg.ws, 0..) |w, i| ps[i] = .{ .w = w, .x = x, .k = @intCast(D), .n = @intCast(xg.ns[i]), .y = switch (i) {
            0 => qa,
            1 => kv,
            else => comp,
        }, .col = if (i == 3) @intCast(hd) else 0 };
        try e.fused("x", ps[0..xg.ws.len]);
        const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kvn = try e.buf("L.kvn", .{}, .bf16, &.{ n, hd });
        // R1 (RMS_FOLD): no launch here; q_norm runs in its consumers' rot_in, kv_norm beside the SWA store
        const qnorm: ?Norm = if (e.o.r1) .{ .w = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}), .eps = dec(e.cfg.eps) } else null;
        if (!e.o.r1) try e.triton("_rms2", .{ n, 2, 1 }, &.{
            .{ .name = "XA", .arg = qa },                                                      .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
            .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                              .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } }, .{ .name = "XB", .arg = kv },
            .{ .name = "xb_rs", .arg = .{ .i = hd } },                                        .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
            .{ .name = "OB", .arg = kvn },                                                    .{ .name = "ob_rs", .arg = .{ .i = hd } },
            .{ .name = "KB", .arg = .{ .i = hd } },                                           .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
            .{ .name = "eps", .arg = .{ .f = dec(e.cfg.eps) } },                              .{ .name = "BK", .arg = .{ .i = 512 } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
        if (!indexer) {
            e.fork();
            try e.swaStore(L, kv, kvn); // R1 (RMS_FOLD): its kv_norm with it, on the side
            e.onMain();
        }
        if (indexer) e.fork();
        if (own_kv) {
            // the compressor: pooled rows, normed, into the pool; the indexer's keys from the same rows
            const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
            try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
                .{ .name = "BUF", .arg = comp },                                                  .{ .name = "b_stride", .arg = .{ .i = cw } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
                .{ .name = "POS", .arg = try e.pos() },                                           .{ .name = "n", .arg = .{ .i = n } },
                .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },                              .{ .name = "RATIO", .arg = .{ .i = ratio } },
                .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
                .{ .name = "OFF", .arg = .{ .i = if (ratio == 2) 1 else 0 } },                    .{ .name = "ROWS", .arg = .{ .b = false } },
                .{ .name = "PDL", .arg = .{ .b = false } },
                .{ .name = "CARRY", .arg = if (ratio == 2) try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, cw }) else .none },
                .{ .name = "c_stride", .arg = .{ .i = if (ratio == 2) cw else 0 } },              .{ .name = "SPLIT", .arg = .{ .b = ratio == 2 } },
            });
            try e.kvStore(lat, L, ratio);
            const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
            try e.upstream(try e.fmt("L{d}.attn.ix_wk", .{L}), lat, hd, ID, kp);
            const kpg = try e.paging(ratio, false);
            try e.triton("_index_k", .{ n, 1, 1 }, &.{
                .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
                .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try e.indexKeys(L) },
                .{ .name = "POS", .arg = try e.pos() },                                          .{ .name = "EPS", .arg = .{ .f = dec(e.cfg.eps) } },
                .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = kpg.pt },
                .{ .name = "PSH", .arg = .{ .i = kpg.psh } },                                    .{ .name = "KFP8", .arg = .{ .b = true } },
                .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
                .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
            });
        }
        if (indexer) try e.index(L, if (e.o.r1) qa else qn, iw, qnorm);
        const early: ?Received = if (indexer and e.o.pool != null and e.o.pool.?.split and e.o.pool.?.early and (n <= 16 or e.o.rows))
            try e.exchange(kv_src.?, e.cfg.compressRatio(kv_src.?), (try e.selection(L))[0])
        else
            null;
        e.onMain();
        // q group: wq_b
        const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
        try e.fusedNorm("q", &.{.{ .w = try e.fmt("L{d}.attn.wq_b", .{L}), .x = if (e.o.r1) qa else qn, .y = q, .k = e.cfg.q_lora, .n = @intCast(H * hd) }}, qnorm);
        try e.prefetchO(L);
        if (!e.o.r1) try e.rope(q, H, hd, hd - 64); // R1 (ATTN_ROPE): the attention kernel rotates q as it loads it
        if (indexer) try e.swaStore(L, kv, kvn);
        e.join(); // the attention core reads both streams' values
        // attention over the SWA ring and (a compressed layer) the source's pool at the source's selection
        const o = try e.buf("L.o", .{}, .bf16, &.{ n, H, hd });
        const swa = try e.pool(L, 0);
        var cv = try e.empty(.u8);
        var cs = try e.empty(.u8);
        var tok = try e.empty(.i32);
        var cnt = try e.empty(.i32);
        var apg: Paging = .{ .pt = try e.empty(.i32), .psh = 0, .vs = 0, .ss = 0 };
        if (ratio > 0) {
            const src_ratio = e.cfg.compressRatio(kv_src.?);
            const sel = try e.selection(e.cfg.indexSource(L).?);
            cnt = sel[1];
            if (e.o.pool != null and e.o.pool.?.split) {
                // split: an index layer's selection brings its rows over the exchange; its reuse layers read them too
                const got = early orelse if (indexer) try e.exchange(kv_src.?, src_ratio, sel[0]) else try e.received(kv_src.?, src_ratio);
                cv = got.cv;
                cs = got.cs;
                tok = got.tok;
            } else {
                const p = try e.pool(kv_src.?, src_ratio);
                cv = p[0];
                cs = p[1];
                tok = sel[0];
                if (e.o.pool != null) {
                    const pg = try e.paging(src_ratio, false);
                    apg = .{ .pt = pg.pt, .psh = pg.psh, .vs = 0, .ss = 0 };
                }
            }
        }
        const rows_scr = n * 5 * H;
        try e.ext("tf_dsv41_attn_cuda_v1.attn", &.{
            try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, 0), cv, cs, tok, cnt, swa[0], swa[1],
            try e.buf("w.lo", .{}, .i32, &.{n}),                                 try e.empty(.i32),
            try e.pos(),                              try e.empty(.i64),
            apg.pt,                                                             try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}),
            try e.ropeTable(),                                                  try e.buf("s.attn.po", .{}, .f32, &.{rows_scr * 512}),
            try e.buf("s.attn.pm", .{}, .f32, &.{rows_scr}),                    try e.buf("s.attn.pl", .{}, .f32, &.{rows_scr}),
            o,                                                                  try e.buf("s.attn.ticket", .{}, .i32, &.{e.ticketLen()}),
            .{ .i = 2 * @as(i64, e.cfg.window) },                               .{ .i = apg.psh },
            .{ .i = 0 },                                                        .{ .i = 4 },
        });
        if (e.o.r1) {
            // ATTN_SPLIT 4 (a cluster of 4 a chunk up to ATTN_SPLIT_ROWS = 2 rows, 1 above) and ATTN_ROPE
            const last = &e.out.items[e.out.items.len - 1];
            const more = try e.a.alloc(calls.Named, last.args.len + 2);
            @memcpy(more[0..last.args.len], last.args);
            more[last.args.len] = .{ .arg = .{ .i = attnSplit(n) } };
            more[last.args.len + 1] = .{ .arg = .{ .i = 1 } };
            last.args = more;
        } else if (e.o.r1_sig) try e.appendLast(&.{ .{ .i = 0 }, .{ .i = 0 } });
        // o: wo_a's groups over the heads' columns, then wo_b (this rank's partial)
        const g: usize = e.cfg.o_groups / e.o.world;
        const gk: i64 = @divExact(H * hd, @as(i64, @intCast(g)));
        const oa = try e.buf("L.oa", .{}, .bf16, &.{ n, @as(i64, @intCast(g)) * e.cfg.o_lora });
        var po: [8]Proj = undefined;
        for (po[0..g], 0..) |*p, i| p.* = .{
            .w = try e.fmt("L{d}.attn.wo_a.{d}", .{ L, i }),
            .x = try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ H * hd, 1 }, @as(i64, @intCast(i)) * gk * 2),
            .y = oa,
            .col = i * e.cfg.o_lora,
            .k = @intCast(gk),
            .n = e.cfg.o_lora,
        };
        try e.fused("o", po[0..g]);
        try e.fused("wo_b", &.{.{ .w = try e.fmt("L{d}.attn.wo_b", .{L}), .x = oa, .y = try e.buf("L.part", .{}, .bf16, &.{ n, D }), .k = g * e.cfg.o_lora, .n = @intCast(D) }});
    }

    /// The indexer: query projection and rope, scores over the keys, top-k (a candidate source also picks candidate
    /// blocks for the Reindex layers; a Reindex layer scores only those).
    fn index(e: *Emitter, L: u32, qn: Arg, iw: Arg, qnorm: ?Norm) !void {
        const n = e.n;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
        try e.fusedNorm("qi", &.{.{ .w = try e.fmt("L{d}.attn.ix_wq_b", .{L}), .x = qn, .y = qi, .k = e.cfg.q_lora, .n = @intCast(IH * ID) }}, qnorm);
        try e.rope(qi, IH, ID, ID - 64);
        const reindex = e.cfg.mode(L) == .reindex;
        const src = if (reindex) e.cfg.kvSource(L).? else L;
        const ratio: i64 = e.cfg.compressRatio(src);
        const cbs: i64 = e.cfg.candidate_block_size;
        const nk: i64 = if (reindex) @as(i64, e.cfg.candidate_blocks) * cbs else @divFloor(e.end(), ratio);
        const cand = try e.buf("w.cand", .{}, .i32, &.{ n, e.cfg.candidate_blocks });
        // backend.select / select_cand / reindex: the materialised scores + sort keys of the whole window within the
        // index budget, else row blocks through `_keys` (_blocked); a one-slot window over 16 rows past the stream
        // threshold would stream (backend._stream_select), which no decode window here is (the plans stop at 16 rows)
        if (reindex) {
            if (n * nk * 16 > e.o.index_budget) return error.Unsupported; // index.reindex in row blocks: > 256 rows at 64 MiB
        } else {
            const nvis = @max(nk, 1);
            if (!e.o.rows and n > 16 and nvis >= prefill.stream_min and !e.cfg.isCandidateSource(L) and @popCount(e.cfg.index_topk) == 1) return error.Unsupported;
            if (n * nvis * 12 > e.o.index_budget) return e.blocked(L, qi, iw, src, ratio, nvis);
        }
        const sc = try e.buf("L.scores", .{}, .f32, &.{ n, nk });
        try e.scoreRows(qi, iw, src, ratio, 0, n, try e.pos(), nk, sc, if (reindex) cand else null);
        try e.topkOf(L, sc, cand, nk, ratio, e.o.index_bound);
    }

    /// index.scores over rows [a, a + rows) of the window into `out` [rows, nk]: dense (key i = position i), or over
    /// the candidate blocks `cand` (a Reindex layer: candidate_keys unmaterialised, CBS).
    fn scoreRows(e: *Emitter, qi: Arg, iw: Arg, src: u32, ratio: i64, a: i64, rows: i64, p: Arg, nk: i64, out: Arg, cand: ?Arg) !void {
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const cbs: i64 = e.cfg.candidate_block_size;
        const spg = try e.paging(ratio, false);
        try e.triton(if (e.o.index_bound and cand == null) "_scores_b" else "_scores", .{ rows, std.math.divCeil(i64, nk, 64) catch unreachable, 1 }, &.{
            .{ .name = "QI", .arg = try e.view(qi.t.role, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, a * IH * ID * 2) },
            .{ .name = "W", .arg = try e.view(iw.t.role, .bf16, &.{ rows, IH }, &.{ IH, 1 }, a * IH * 2) },
            .{ .name = "w_stride", .arg = .{ .i = IH } },               .{ .name = "IK", .arg = try e.indexKeys(src) },
            .{ .name = "OUT", .arg = out },                             .{ .name = "POS", .arg = p },
            .{ .name = "KEYS", .arg = cand orelse out },                .{ .name = "k_stride", .arg = .{ .i = if (cand != null) e.cfg.candidate_blocks else nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },                     .{ .name = "o_stride", .arg = .{ .i = nk } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },               .{ .name = "H", .arg = .{ .i = IH } },
            .{ .name = "D", .arg = .{ .i = ID } },                      .{ .name = "BP", .arg = .{ .i = 64 } },
            .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
            .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
            .{ .name = "GATHER", .arg = .{ .b = cand != null } },       .{ .name = "PT", .arg = spg.pt },
            .{ .name = "PSH", .arg = .{ .i = spg.psh } },               .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                            .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                .{ .name = "CBS", .arg = .{ .i = if (cand != null) cbs else 0 } },
        });
    }

    /// backend._blocked (a window whose scores + sort keys pass the index budget: decode windows at long positions,
    /// prod's 64 MiB holds 5 rows at 1M keys): row blocks of backend._row_blocks, each `_scores`, `_keys` and
    /// index.top_positions (glue "topk"); a candidate source also its blocks (index.candidate_blocks: `_block_keys` +
    /// top_positions); then the counts from the selection (glue "counts", backend.visible_counts). A block's first
    /// position is Win.sub's (glue "block_pos", which row mode turns into the block's rows of the table); a one-slot
    /// block scores to its own last row, a row-mode block (RowWin.sub) the bucket's keys.
    fn blocked(e: *Emitter, L: u32, qi: Arg, iw: Arg, src: u32, ratio: i64, nvis: i64) !void {
        const n = e.n;
        const k: i64 = e.cfg.index_topk;
        const cb: i64 = e.cfg.candidate_blocks;
        const cbs: i64 = e.cfg.candidate_block_size;
        const sel = try e.selection(L);
        const cand_src = e.cfg.isCandidateSource(L);
        const cand = try e.buf("w.cand", .{}, .i32, &.{ n, cb });
        const st: i64 = if (n > 16) @max(16, @divFloor(e.o.index_budget, nvis * 12)) else n;
        var a: i64 = 0;
        var j: usize = 0;
        while (a < n) : ({
            a += st;
            j += 1;
        }) {
            const rows = @min(st, n - a);
            const p = if (st >= n) try e.pos() else try e.blockPos(j, a, rows);
            const nk = if (st >= n or e.o.rows) nvis else @max(@divFloor(e.start + a + rows, ratio), 1);
            const sc = try e.buf("L.scores", .{}, .f32, &.{ rows, nk });
            try e.scoreRows(qi, iw, src, ratio, a, rows, p, nk, sc, null);
            const keys = try e.buf("L.ix.keys", .{}, .i64, &.{ rows, nk });
            try e.triton("_keys", .{ rows, std.math.divCeil(i64, nk, 1024) catch unreachable, 1 }, &.{
                .{ .name = "S", .arg = sc },                .{ .name = "s_stride", .arg = .{ .i = nk } },
                .{ .name = "K", .arg = keys },              .{ .name = "k_stride", .arg = .{ .i = nk } },
                .{ .name = "POSN", .arg = sc },             .{ .name = "p_stride", .arg = .{ .i = nk } },
                .{ .name = "NK", .arg = .{ .i = nk } },     .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
                .{ .name = "GATHER", .arg = .{ .b = false } },
            });
            try e.glue("topk", &.{ keys, try e.view(sel[0].t.role, .i32, &.{ rows, k }, &.{ k, 1 }, a * k * 4), .{ .i = k } });
            if (!cand_src) continue;
            const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
            const bk = try e.buf("L.ix.bkeys", .{}, .i64, &.{ rows, nb });
            try e.triton("_block_keys", .{ rows, std.math.divCeil(i64, nb, 256) catch unreachable, 1 }, &.{
                .{ .name = "S", .arg = sc },                  .{ .name = "s_stride", .arg = .{ .i = nk } },
                .{ .name = "K", .arg = bk },                  .{ .name = "k_stride", .arg = .{ .i = nb } },
                .{ .name = "POS", .arg = p },                 .{ .name = "NB", .arg = .{ .i = nb } },
                .{ .name = "RATIO", .arg = .{ .i = ratio } }, .{ .name = "BS", .arg = .{ .i = cbs } },
                .{ .name = "TB", .arg = .{ .i = 256 } },      .{ .name = "ROWS", .arg = .{ .b = false } },
            });
            try e.glue("topk", &.{ bk, try e.view(cand.t.role, .i32, &.{ rows, cb }, &.{ cb, 1 }, a * cb * 4), .{ .i = cb } });
        }
        try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
    }

    /// Win.sub's first position (int32 [1]) of the row block [a, a + rows): the glue writes start + a; row mode reads
    /// the block's rows of the table instead (rowmode.zig keys on the glue's `a` and `rows`).
    fn blockPos(e: *Emitter, j: usize, a: i64, rows: i64) !Arg {
        const p = try e.buf("L.ix.pos{d}", .{j}, .i32, &.{1});
        try e.glue("block_pos", &.{ p, .{ .i = e.start + a }, .{ .i = a }, .{ .i = rows } });
        return p;
    }

    /// attn_cuda topk over a window's `scores` [n, nk] of index layer L: up to two jobs in one launch (the list repeats a
    /// single job). A candidate source also picks the candidate blocks into `cand`; a Reindex layer's scores are over
    /// `cand`'s blocks. Python's short prefill segments take it too (block_prefill.zig).
    pub fn topk(e: *Emitter, L: u32, scores: Arg, cand: Arg, nk: i64, ratio: i64) !void {
        return e.topkOf(L, scores, cand, nk, ratio, false);
    }

    /// `topk`; `bound` (TF_DSV41_INDEX_BOUND, decode windows): the dense select and candidate-block launches as the
    /// row-bounded twins `_dtopk_b` / `tf_dsv41_topk_b_v1.topk` (the same grids, arguments and bytes); a Reindex
    /// layer's selection over its candidates (mode 3: not in position order) keeps attn_cuda's.
    fn topkOf(e: *Emitter, L: u32, scores: Arg, cand: Arg, nk: i64, ratio: i64, bound: bool) !void {
        const n = e.n;
        const dtopk = if (bound) "_dtopk_b" else "_dtopk";
        const topk_ext = if (bound and e.cfg.mode(L) != .reindex) "tf_dsv41_topk_b_v1.topk" else "tf_dsv41_attn_cuda_v1.topk";
        const reindex = e.cfg.mode(L) == .reindex;
        const cbs: i64 = e.cfg.candidate_block_size;
        const sel = try e.selection(L);
        const k: i64 = e.cfg.index_topk;
        const pl = topkPlan(nk, null);
        if (pl[1] > topk_max_ept) {
            // attn_cuda.plan is None past 8 x 512 x 31 entries: Python's backend runs dtopk.select (Triton _dtopk,
            // the counts in the same launch), then a candidate source's blocks alone (attn_cuda.blocks: one job over
            // the cdiv(nk, 8) blocks). A Reindex layer's scores are over the candidate blocks (16,384 keys): never here.
            if (reindex) return error.Unsupported;
            try e.triton(dtopk, .{ n, 1, 1 }, &.{
                .{ .name = "S", .arg = scores },                 .{ .name = "s_stride", .arg = .{ .i = scores.t.stride[0] } },
                .{ .name = "P", .arg = scores },                 .{ .name = "p_stride", .arg = .{ .i = 0 } },
                .{ .name = "POS", .arg = try e.pos() },          .{ .name = "NK", .arg = .{ .i = nk } },
                .{ .name = "OUT", .arg = sel[0] },               .{ .name = "o_stride", .arg = .{ .i = k } },
                .{ .name = "CNT", .arg = sel[1] },               .{ .name = "RATIO", .arg = .{ .i = ratio } },
                .{ .name = "K", .arg = .{ .i = k } },            .{ .name = "MODE", .arg = .{ .i = 0 } },
                .{ .name = "ROWS", .arg = .{ .b = false } },     .{ .name = "T", .arg = .{ .i = 4096 } },
                .{ .name = "BS", .arg = .{ .i = 8 } },           .{ .name = "RB", .arg = .{ .i = 8 } },
                .{ .name = "NPASS", .arg = .{ .i = 8 } },        .{ .name = "SORT", .arg = .{ .b = false } },
                .{ .name = "COUNT", .arg = .{ .b = true } },     .{ .name = "KP", .arg = .{ .i = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(k)) catch unreachable) } },
            });
            if (!e.cfg.isCandidateSource(L)) return;
            const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
            const bp = topkPlan(nb, null);
            if (bp[1] > topk_max_ept) {
                // past 8 x 512 x 31 blocks too (positions past ~1,015,808): dtopk.blocks (_dtopk mode 2, no counts)
                const cb: i64 = e.cfg.candidate_blocks;
                return e.triton(dtopk, .{ n, 1, 1 }, &.{
                    .{ .name = "S", .arg = scores },             .{ .name = "s_stride", .arg = .{ .i = scores.t.stride[0] } },
                    .{ .name = "P", .arg = scores },             .{ .name = "p_stride", .arg = .{ .i = 0 } },
                    .{ .name = "POS", .arg = try e.pos() },      .{ .name = "NK", .arg = .{ .i = nb } },
                    .{ .name = "OUT", .arg = cand },             .{ .name = "o_stride", .arg = .{ .i = cand.t.stride[0] } },
                    .{ .name = "CNT", .arg = cand },             .{ .name = "RATIO", .arg = .{ .i = ratio } },
                    .{ .name = "K", .arg = .{ .i = cb } },       .{ .name = "MODE", .arg = .{ .i = 2 } },
                    .{ .name = "ROWS", .arg = .{ .b = false } }, .{ .name = "T", .arg = .{ .i = 4096 } },
                    .{ .name = "BS", .arg = .{ .i = cbs } },     .{ .name = "RB", .arg = .{ .i = 8 } },
                    .{ .name = "NPASS", .arg = .{ .i = 8 } },    .{ .name = "SORT", .arg = .{ .b = false } },
                    .{ .name = "COUNT", .arg = .{ .b = false } }, .{ .name = "KP", .arg = .{ .i = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(cb)) catch unreachable) } },
                });
            }
            const job: [8]Arg = .{ scores, try e.empty(.i32), cand, try e.view(cand.t.role, .i32, &.{0}, &.{e.cfg.candidate_blocks}, 0), .{ .i = nb }, .{ .i = e.cfg.candidate_blocks }, .{ .i = 2 }, .{ .i = bp[1] } };
            var one: std.ArrayList(Arg) = .empty;
            try one.appendSlice(e.a, &job);
            try one.appendSlice(e.a, &job);
            try one.appendSlice(e.a, &.{ .{ .i = 1 }, try e.pos(), .{ .i = ratio }, .{ .i = cbs }, .{ .i = n }, .{ .i = bp[0] } });
            return e.ext(topk_ext, one.items);
        }
        var jobs: [2][8]Arg = undefined;
        jobs[0] = .{ scores, if (reindex) cand else try e.empty(.i32), sel[0], sel[1], .{ .i = nk }, .{ .i = k }, .{ .i = if (reindex) 3 else 0 }, .{ .i = pl[1] } };
        const blocks = e.cfg.isCandidateSource(L);
        jobs[1] = jobs[0];
        if (blocks) {
            const nb = std.math.divCeil(i64, nk, cbs) catch unreachable;
            jobs[1] = .{ scores, try e.empty(.i32), cand, try e.view(cand.t.role, .i32, &.{0}, &.{e.cfg.candidate_blocks}, 0), .{ .i = nb }, .{ .i = e.cfg.candidate_blocks }, .{ .i = 2 }, .{ .i = topkPlan(nb, pl[0])[1] } };
        }
        var args: std.ArrayList(Arg) = .empty;
        try args.appendSlice(e.a, &jobs[0]);
        try args.appendSlice(e.a, &jobs[1]);
        try args.appendSlice(e.a, &.{ .{ .i = if (blocks) 2 else 1 }, try e.pos(), .{ .i = ratio }, .{ .i = if (blocks or reindex) cbs else 1 }, .{ .i = n }, .{ .i = pl[0] } });
        try e.ext(topk_ext, args.items);
    }

    // -----------------------------------------------------------------------------------------------------------
    // the window

    /// Every launch of a decode window of `n` rows from position `start` over `layers` (ascending, consecutive), and
    /// the head when `head` (the window ends the model).
    pub fn window(e: *Emitter, layers: []const u32, head: bool) !void {
        // the window's positions and streams (this rank's embedding rows, their all-sum: the first exchange)
        try e.glue("positions", &.{ try e.pos(), try e.buf("w.pos64", .{}, .i64, &.{e.n}), try e.buf("w.lo", .{}, .i32, &.{e.n}) });
        try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, e.cfg.hidden }) });
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            e.layer_ratio = e.cfg.compressRatio(L);
            if (e.cfg.isEngram(L)) {
                // Python's forward: an Engram block after the first runs the last exchange's post alone, Engram on the
                // streams, then a site that collapses with the previous coefficients' pre (site2)
                if (i > 0) try moe.postOnly(e, null);
                try moe.engram(e, L);
                try moe.mhc(e, L, "hc_attn", if (i == 0) .site1 else .site2);
            } else try moe.mhc(e, L, "hc_attn", if (i == 0) .site1 else .boundary);
            try e.attention(L);
            try e.prefetchF(L);
            const D: i64 = e.cfg.hidden;
            try e.glue("exchange", &.{ try e.buf("L.part", .{}, .bf16, &.{ e.n, D }), try e.buf("w.recv.attn", .{}, .bf16, &.{ e.o.world, e.n, D }) });
            try moe.mhc(e, L, "hc_ffn", .boundary);
            try moe.moe(e, L);
            if (i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
            // R1 (MOE_BF16): down_combine stores the partial as bf16, the exchange takes it as is
            if (e.o.r1)
                try e.glue("exchange", &.{ try e.buf("L.moe", .{}, .bf16, &.{ e.n, D }), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D }) })
            else
                try e.glue("exchange_f32", &.{ try e.buf("L.moe", .{}, .f32, &.{ e.n, D }), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D }) });
            if (e.o.trace) {
                const y = try e.buf("w.trace", .{}, .bf16, &.{ e.n, 4 * D });
                try moe.postOnly(e, y);
                try e.glue("trace", &.{ y, .{ .i = L } });
            }
        }
        if (head) try moe.finish(e);
    }
};

/// The window's calls (arena-owned by `a`).
pub fn emit(a: std.mem.Allocator, cfg: *const Config, w: *const Widths, o: Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    var e: Emitter = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start };
    try e.window(layers, head);
    if (o.holdsDeferred(n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}

/// exl3 expert_loads.k2_range: the instance covering a layer's half-bit widths [lo, hi].
pub fn k2Range(lo: u32, hi: u32) ![2]i64 {
    const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 12 } };
    for (ranges) |r| if (r[0] <= lo and hi <= r[1]) return .{ r[0], r[1] };
    return error.NoInstance;
}
