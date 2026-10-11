//! DeepSeek-V4.1's other prod .cu kernels: CSA2 attention (attn_cuda) and its top-k (topk_cuda), the mHC boundary
//! (mhc_cuda, decode) and site (mhc_pf, prefill), the router GEMV (router_gemv), the fused prefill dense GEMM
//! (pfdense), the Engram gate copy and the L2 prefetchers. Args structs are the kernels' own (passed by value);
//! each launch mirrors its C++ entry's dispatch (instance, grid, block, shared memory, cluster) and checks.

const std = @import("std");
const cuda = @import("cuda");
const fitShared = @import("kernels_exl3.zig").fitShared;

fn dim(x: usize) u32 {
    return @intCast(x);
}

pub const Error = error{ Shape, Unsupported };

// ---------------------------------------------------------------------------------------------------------------
// attn_cuda.cu: dsv41_attn

pub const attn = struct {
    pub const NCH = 5;
    pub const BMQ = 16;
    pub const WINDOW = 128;
    pub const NT = 256;
    pub const SMEM: u32 = 98816;

    pub const Args = extern struct {
        q: u64 = 0,
        cv: u64 = 0,
        csc: u64 = 0,
        cvs: i64 = 0,
        css: i64 = 0,
        tok: u64 = 0,
        ts: i64 = 0,
        cnt: u64 = 0,
        sv: u64 = 0,
        ssc: u64 = 0,
        svs: i64 = 0,
        sss: i64 = 0,
        lo: u64 = 0,
        hi: u64 = 0,
        pos: u64 = 0,
        sl: u64 = 0,
        pt: u64 = 0,
        pts: i64 = 0,
        psh: c_int = 0,
        sink: u64 = 0,
        cs: u64 = 0,
        csst: i64 = 0,
        po: u64 = 0,
        pm: u64 = 0,
        pl: u64 = 0,
        out: u64 = 0,
        ticket: u64 = 0,
        R: c_int = 0,
        H: c_int = 0,
        ring: c_int = 0,
        has_comp: c_int = 0,
        qrope: c_int = 0, // R1c (split_kernel only): q arrives un-rotated and is RoPE'd as it loads
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "psh") == 144 and @offsetOf(Args, "sink") == 152 and @offsetOf(Args, "qrope") == 232 and @sizeOf(Args) == 240);
    }

    pub fn symbol(comptime kw: u32, comptime rows: bool, comptime hi: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10dsv41_attn11attn_kernelILi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ kw, @intFromBool(rows), @intFromBool(hi) });
    }

    /// R1 split_kernel<KW, ROWS, HAS_HI, P>: P CTAs a chunk, a cluster of P (P 4); the same bits as attn_kernel.
    pub fn splitSymbol(comptime kw: u32, comptime rows: bool, comptime hi: bool, comptime p: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10dsv41_attn12split_kernelILi{d}ELb{d}ELb{d}ELi{d}EEEvNS_4ArgsE", .{ kw, @intFromBool(rows), @intFromBool(hi), p });
    }

    /// Split<P>::SMEM: P 1 the same as attn_kernel's, P 4 the S rows' B pairs beside the PV slice.
    pub fn splitSmem(p: u32) u32 {
        return if (p == 4) 100864 else SMEM;
    }

    /// The ticket's ranks a (row, head tile) once any launch splits: the ticket buffer must hold R H / 16 * MAXP.
    pub const MAXP = 4;

    /// attn_cuda.split(R) (TF_DSV41_ATTN_SPLIT on): P 4 up to `split_rows` rows (2), P 1 above.
    pub fn splitFor(rows: usize, split_rows: usize) u32 {
        return if (rows <= split_rows) 4 else 1;
    }

    /// dispatch(): the instance index (kw 4 | rows | has_hi), after its checks.
    pub fn instance(a: Args, kw: u32, rows: bool, has_hi: bool) Error!usize {
        if (a.R < 1 or @rem(a.H, BMQ) != 0 or a.ring < WINDOW or (a.ring & (a.ring - 1)) != 0) return error.Shape;
        return @as(usize, if (kw == 4) 4 else 0) + @as(usize, if (rows) 2 else 0) + @intFromBool(has_hi);
    }
};

// topk_cuda.cu: dsv41_topk
pub const topk = struct {
    pub const NT = 512;
    pub const MAXCL = 8;
    pub const MAXEPT = 31;

    pub const Job = extern struct { s: u64 = 0, ss: i64 = 0, cand: u64 = 0, cs: i64 = 0, out: u64 = 0, os: i64 = 0, cnt: u64 = 0, nk: c_int = 0, k: c_int = 0, mode: c_int = 0, ept: c_int = 0 };
    pub const Args = extern struct { job: [2]Job, pos: u64 = 0, pos64: c_int = 0, ratio: c_int = 0, bs: c_int = 0 };

    comptime {
        std.debug.assert(@sizeOf(Job) == 72 and @offsetOf(Args, "pos") == 144 and @sizeOf(Args) == 168);
    }

    pub const symbol = "_ZN10dsv41_topk11topk_kernelENS_4ArgsE";
    /// topk_b.cu (ours, TF_DSV41_INDEX_BOUND): topk_kernel with each row stopped at its own visible end, the same Args
    pub const symbol_b = "_ZN10dsv41_topk13topk_b_kernelENS_4ArgsE";

    pub fn smem(ept: usize) u32 {
        return @intCast(NT * ept * 4);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// mhc_cuda.cu: dsv41_mhc (decode boundary, <= 16 rows)

pub const mhc = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const MAXR = 16;
    pub const ROWB = 128;
    /// coef_kernel is a CTA a row with nothing sized by R: TF_DSV41_MHC_PFDEC launches it past MAXR, up to a decode
    /// window's 64 rows (the Python binding stops at MAXR, the only rows it launches)
    pub const COEF_ROWS = 64;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        base: u64 = 0,
        scale: u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        nw: u64 = 0,
        nwbf16: c_int = 0,
        out: u64 = 0,
        opre: u64 = 0,
        opost: u64 = 0,
        ocomb: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        eps: f32 = 0,
        hc_eps: f32 = 0,
        post_alpha: f32 = 0,
        iters: c_int = 0,
        spin: i64 = -1, // R1 tail: SM cycles a CTA waits for the last arrival (the binding's -1: today's finish)
        @"defer": c_int = 0, // R1 tail: the coefficient items run in coef_kernel on a side stream
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "post") == 48 and @offsetOf(Args, "nwbf16") == 136 and @offsetOf(Args, "R") == 184 and @offsetOf(Args, "spin") == 208 and @sizeOf(Args) == 224);
    }

    pub const coef_symbol = "_ZN9dsv41_mhc11coef_kernelENS_4ArgsE";

    /// (POST, COLL, MIX, ALL4, FN32) of dispatch's cases, by mode * 2 + fn32 (mode 3 ignores fn32).
    pub const instances = [_]struct { post: bool, coll: u32, mix: bool, all4: bool, fn32: bool }{
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = true, .coll = 2, .mix = false, .all4 = true, .fn32 = false },
    };

    /// boundary_kernel<POST, COLL, MIX, ALL4, FN32, TAIL> (TAIL: R1's parallel tail, default false: still mangled)
    pub fn symbol(comptime i: usize, comptime tail: bool) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN9dsv41_mhc15boundary_kernelILb{d}ELi{d}ELb{d}ELb{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.mix), @intFromBool(s.all4), @intFromBool(s.fn32), @intFromBool(tail) });
    }

    /// dispatch(): mode 0 boundary, 1 site from stream 0, 2 site with a carried pre-mix, 3 final.
    pub fn instance(a: Args, mode: u32, fn32: bool) Error!usize {
        if (a.R < 1 or a.R > MAXR or a.world < 1 or a.world > 8) return error.Shape;
        if (mode == 3) return 6;
        if (mode > 2) return error.Unsupported;
        return mode * 2 + @intFromBool(fn32);
    }
};

// mhc_pf.cu: dsv41_mhc_pf (prefill site, 16 rows a CTA)
pub const mhc_pf = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const BM = 16;
    pub const MAXW = 8;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        R: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "R") == 112 and @sizeOf(Args) == 120);
    }

    /// Lay<FN32>::BYTES: staged streams, the collapsed rows and fn.
    pub fn smem(fn32: bool) u32 {
        const pb = 128 + 8;
        const pf = 128 + 4;
        const xs = 4 * BM * pb * 2;
        const cs = BM * pb * 2;
        const fnb: u32 = if (fn32) 4 * 24 * pf * 4 else 4 * 24 * pb * 2;
        return xs + cs + fnb;
    }

    /// (POST, COLL, FN32, TAP) by dispatch's case (mode * 4 + fn32 * 2 + tap).
    pub const instances = [_]struct { case: u32, post: bool, coll: u32, fn32: bool, tap: bool }{
        .{ .case = 0, .post = true, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 1, .post = true, .coll = 2, .fn32 = false, .tap = true },
        .{ .case = 2, .post = true, .coll = 2, .fn32 = true, .tap = false },
        .{ .case = 3, .post = true, .coll = 2, .fn32 = true, .tap = true },
        .{ .case = 4, .post = false, .coll = 1, .fn32 = false, .tap = false },
        .{ .case = 6, .post = false, .coll = 1, .fn32 = true, .tap = false },
        .{ .case = 8, .post = false, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 10, .post = false, .coll = 2, .fn32 = true, .tap = false },
    };

    pub fn symbol(comptime i: usize) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN12dsv41_mhc_pf11site_kernelILb{d}ELi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.fn32), @intFromBool(s.tap) });
    }

    pub fn instance(a: Args, mode: u32, fn32: bool, tap: bool) Error!usize {
        if (a.R < 1 or a.world < 1 or a.world > MAXW or (tap and mode != 0)) return error.Shape;
        const case = mode * 4 + @as(u32, if (fn32) 2 else 0) + @intFromBool(tap);
        for (instances, 0..) |s, i| if (s.case == case) return i;
        return error.Unsupported;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// router_gemv.cu: dsv41_rg

pub const rg = struct {
    pub const D = 5120;
    pub const EPL = 16;
    pub const KMAX = 8;
    pub const GE_MAX = 512;
    pub const rts = [_]u32{ 1, 2, 4, 8, 16 };
    pub const wide = [_][2]u32{ .{ 8, 1 }, .{ 4, 1 }, .{ 8, 2 }, .{ 4, 2 } };
    pub const narrow = [_]u32{ 1, 2, 4, 8 };

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        w: u64 = 0,
        bias: u64 = 0,
        pick: u64 = 0,
        wts: u64 = 0,
        lg: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        E: c_int = 0,
        K: c_int = 0,
        slots: c_int = 0,
        select: c_int = 0,
        scale: f32 = 0,
        kit: c_int = 0,
        gid: u64 = 0,
        gcnt: u64 = 0,
        gmem: u64 = 0,
        GE: c_int = 0,
        maxm: c_int = 0,
        group: c_int = 0,
        prune: Prune = .{}, // R1c: prune.py's rule folded into the selection (on 0: off)
    };

    /// router_gemv.cu Prune: [min_w, topp, scale, min_k, orig, drop, dup] (prune.fold_args) in fp32 / int, on = 1.
    pub const Prune = extern struct { min_w: f32 = 0, topp: f32 = 0, scale: f32 = 0, min_k: c_int = 0, orig: c_int = 0, drop: c_int = 0, dup: c_int = 0, on: c_int = 0 };

    /// The binding's prune vector (7 doubles, empty: off) as the kernel's struct (fp32 thresholds, as it rounds them).
    pub fn pruneOf(v: []const f64) error{Shape}!Prune {
        if (v.len == 0) return .{};
        if (v.len != 7) return error.Shape;
        return .{ .min_w = @floatCast(v[0]), .topp = @floatCast(v[1]), .scale = @floatCast(v[2]), .min_k = @intFromFloat(v[3]), .orig = @intFromFloat(v[4]), .drop = @intFromFloat(v[5]), .dup = @intFromFloat(v[6]), .on = 1 };
    }

    comptime {
        std.debug.assert(@sizeOf(Prune) == 32);
        std.debug.assert(@offsetOf(Args, "gid") == 96 and @offsetOf(Args, "group") == 128 and @offsetOf(Args, "prune") == 132 and @sizeOf(Args) == 168);
    }

    pub fn gemvSymbol(comptime rt: u32, comptime nw: u32, comptime ew: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg11gemv_kernelILi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw, ew });
    }

    pub fn narrowSymbol(comptime rt: u32, comptime nw: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg13narrow_kernelILi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw });
    }

    /// dsv41_rg_route_cuda's checks (row stride / alignment of x and the buffer sizes are the caller's).
    pub fn check(a: Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) Error!void {
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        if (E < 1 or E > 32 * EPL or a.K < 1 or a.K > KMAX or a.K > a.E or a.slots < a.K or a.slots > a.K + 1) return error.Shape;
        if (lg_len < R * E) return error.Shape;
        if (a.select != 0 and cnt_len < (R + rt - 1) / rt) return error.Shape;
        const is_wide = (nw == 4 or nw == 8) and (ew == 1 or ew == 2);
        const is_narrow = ew == 0 and (nw == 1 or nw == 2 or nw == 4 or nw == 8);
        if (std.mem.indexOfScalar(u32, &rts, rt) == null or !(is_wide or is_narrow)) return error.Unsupported;
        if (a.group != 0 and (a.select == 0 or !is_narrow or R > rt or a.GE < 1 or a.GE > GE_MAX)) return error.Shape;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// pfdense.cuh: dsv41_pfd (fused prefill dense GEMM, one fatbin a width)

pub const pfd = struct {
    pub const k2s = [_]u32{ 8, 10, 12, 16 };
    /// DSV41_PFD_CFGS: (BM, WM, WN, KS, NST) by id.
    pub const cfgs = [_][5]u32{ .{ 128, 2, 4, 2, 3 }, .{ 128, 2, 4, 2, 4 }, .{ 128, 4, 2, 2, 3 }, .{ 64, 2, 2, 2, 4 }, .{ 64, 1, 4, 2, 4 }, .{ 128, 2, 4, 2, 2 }, .{ 32, 1, 4, 2, 4 }, .{ 256, 4, 2, 2, 3 } };
    pub const OutType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

    pub const Args = extern struct {
        xh: u64 = 0,
        T: u64 = 0,
        stride_k: i64 = 0,
        stride_nb: i64 = 0,
        svh: u64 = 0,
        bias: u64 = 0,
        out: u64 = 0,
        o_stride: i64 = 0,
        M: c_int = 0,
        K: c_int = 0,
        N: c_int = 0,
        group: c_int = 0,
        out_type: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "M") == 64 and @sizeOf(Args) == 88);
    }

    pub fn threads(cfg: usize) u32 {
        return cfgs[cfg][1] * cfgs[cfg][2] * 32;
    }

    /// Lay<K2, LANES, Cfg>::SMEM (the layout does not depend on LANES).
    pub fn smem(cfg: usize, k2: u32) u32 {
        const bm, const wm, _, const ks, const nst = cfgs[cfg];
        const a_bytes = bm * (16 * ks) * 2;
        const w_bytes = ks * 8 * (4 * k2) * 4;
        const ring = nst * (a_bytes + w_bytes);
        const main = ring + 2 * (ks * 8 * 32 * 16);
        const epi = (bm / wm) * 128 * 2 * 2;
        return @max(main, epi);
    }

    pub fn symbol(comptime k2: u32, comptime lanes: bool, comptime cfg: usize) [:0]const u8 {
        const c = cfgs[cfg];
        return std.fmt.comptimePrint("_ZN9dsv41_pfd10pfd_kernelILi{d}ELb{d}ENS_3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS_4ArgsE", .{ k2, @intFromBool(lanes), c[0], c[1], c[2], c[3], c[4] });
    }

    pub fn dequantSymbol(comptime k2: u32, comptime lanes: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN9dsv41_pfd18pfd_dequant_kernelILi{d}ELb{d}EEEvPKjxxP6__halfi", .{ k2, @intFromBool(lanes) });
    }

    /// pfdense.cpp gemm's checks on the shapes and strides (dtype / alignment are the caller's).
    pub fn check(a: Args, k2: u32, lanes: bool, cfg: usize, words: usize) Error!void {
        if (std.mem.indexOfScalar(u32, &k2s, k2) == null or (lanes and k2 == 16)) return error.Unsupported;
        if (cfg >= cfgs.len) return error.Unsupported;
        const ks: i64 = cfgs[cfg][3];
        const K: i64 = a.K;
        const N: i64 = a.N;
        if (a.M < 1 or @rem(K, 128) != 0 or K < 128 or @rem(K, 16 * ks) != 0) return error.Shape;
        if (@rem(N, 128) != 0 or N < 128 or @rem(a.o_stride, 2) != 0 or a.o_stride < N) return error.Shape;
        const tw: i64 = 4 * @as(i64, k2);
        if (@rem(a.stride_k, 4) != 0 or @rem(a.stride_nb, 4) != 0 or a.stride_k < 8 * tw or a.stride_nb < 8 * tw) return error.Shape;
        const last = (@divTrunc(K, 16) - 1) * a.stride_k + (@divTrunc(N, 128) - 1) * a.stride_nb + 8 * tw;
        if (last > @as(i64, @intCast(words))) return error.Shape;
        if (a.group < 1 or a.group > 1024) return error.Shape;
    }

    /// pfdense.heuristic (no sweep table): the largest row tile whose last wave is not mostly empty.
    pub fn heuristic(k: usize, n: usize, k2: u32, m: usize, sms: usize) [2]usize {
        const nb = n / 128;
        var best: ?usize = null;
        var best_eff: f64 = -1.0;
        for ([_]usize{ 0, 3, 6 }) |cfg| {
            if (k % (16 * cfgs[cfg][3]) != 0) continue;
            const bm = cfgs[cfg][0];
            const blocks: usize = if (2 * (smem(cfg, k2) + 1024) <= 102400) 2 else 1;
            const ctas = ((m + bm - 1) / bm) * nb;
            const slots = sms * blocks;
            const eff = @as(f64, @floatFromInt(ctas)) / @as(f64, @floatFromInt(((ctas + slots - 1) / slots) * slots));
            if (eff > best_eff + 0.10) {
                best = cfg;
                best_eff = eff;
            }
        }
        return .{ best orelse 0, 8 };
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Loaded functions and launches

pub const Functions = struct {
    attn: [8]cuda.Function,
    attn_split: [2][8]cuda.Function, // [P 1, P 4][instance]
    topk: cuda.Function,
    /// topk_b.cu's row-bounded twin (tf_dsv41_topk_b_v1.topk)
    topk_b: cuda.Function,
    mhc: [2][mhc.instances.len]cuda.Function, // [tail][instance]
    mhc_coef: cuda.Function,
    mhc_pf: [mhc_pf.instances.len]cuda.Function,
    gemv: [rg.rts.len][rg.wide.len]cuda.Function,
    narrow: [rg.rts.len][rg.narrow.len]cuda.Function,
    pfd: [pfd.k2s.len][2][pfd.cfgs.len]?cuda.Function, // [k2][lanes][cfg]; K2 16 has no lanes kernels
    pfd_dequant: [pfd.k2s.len][2]?cuda.Function,
    gate_copy: cuda.Function,
    paced: cuda.Function,
    segments: [2]cuda.Function, // bulk false / true
    topk_keys: cuda.Function,
    swiglu: cuda.Function,
    add: cuda.Function,
    add_bf16: cuda.Function,
    kv_dense: cuda.Function,
    /// kvsplit's compact pack (TF_DSV41_KV_SPLIT_COMPACT); null in a fatbin built before it
    kv_pack: ?cuda.Function = null,
    /// kvsplit's dedup steps (TF_DSV41_KV_SPLIT_DEDUP: mark, flags, pack, tok); null in a fatbin built before them
    kv_dedup: ?[4]cuda.Function = null,
    kv_gather: cuda.Function,
    glue: [10]cuda.Function, // glue.syms order
    /// glue.cu's DSpark candidates (TF_DSV41_DRAFT_GRAPHS); null in a fatbin built before it (the host's candidates)
    ds_cands: ?cuda.Function = null,
    /// glue.cu's device pick (TF_DSV41_GREEDY_GPU); null in a fatbin built before it (the pairs merged on the host)
    pick_pack: ?cuda.Function = null,
    pick_merge: ?cuda.Function = null,
    /// glue.cu's DSpark statics from the device pick (the speculative pass's device start); optional, as ds_cands
    ds_stage: ?cuda.Function = null,
    /// glue.cu's row-window accept and several-slot statics (the slot passes' device start); optional
    ds_accept_rows: ?cuda.Function = null,
    ds_stage_rows: ?cuda.Function = null,
    /// glue.cu's device keyed choice (vsample.choose; the sampled rounds' device start); optional
    vs_choose: ?cuda.Function = null,
    pfglue: [pfglue.syms.len]cuda.Function, // pfglue.syms order

    /// `pf`: the four pfdense modules (K2 8, 10, 12, 16).
    pub fn resolve(m_attn: cuda.Module, m_topk: cuda.Module, m_mhc: cuda.Module, m_mhc_pf: cuda.Module, m_rg: cuda.Module, pf: [4]cuda.Module, m_gate: cuda.Module, m_pace: cuda.Module, m_l2pf: cuda.Module, m_keys: cuda.Module, m_pw: cuda.Module, m_kv: cuda.Module, m_glue: cuda.Module, m_pfglue: cuda.Module, m_topk_b: cuda.Module, optin: u32) !Functions {
        var f: Functions = undefined;
        inline for (0..8) |i| f.attn[i] = try m_attn.function(attn.symbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0));
        inline for (f.attn, 0..) |fun, i| {
            if (!try fitShared(fun, attn.symbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0), attn.SMEM, optin)) return error.SharedMemoryTooSmall;
        }
        inline for (.{ 1, 4 }, 0..) |p, pi| inline for (0..8) |i| {
            const name = attn.splitSymbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0, p);
            f.attn_split[pi][i] = try m_attn.function(name);
            if (!try fitShared(f.attn_split[pi][i], name, attn.splitSmem(p), optin)) return error.SharedMemoryTooSmall;
        };
        f.topk = try m_topk.function(topk.symbol);
        if (!try fitShared(f.topk, topk.symbol, topk.smem(topk.MAXEPT), optin)) return error.SharedMemoryTooSmall;
        f.topk_b = try m_topk_b.function(topk.symbol_b);
        if (!try fitShared(f.topk_b, topk.symbol_b, topk.smem(topk.MAXEPT), optin)) return error.SharedMemoryTooSmall;
        inline for (.{ false, true }, 0..) |tail, ti| inline for (0..mhc.instances.len) |i| {
            f.mhc[ti][i] = try m_mhc.function(mhc.symbol(i, tail));
        };
        f.mhc_coef = try m_mhc.function(mhc.coef_symbol);
        inline for (0..mhc_pf.instances.len) |i| {
            f.mhc_pf[i] = try m_mhc_pf.function(mhc_pf.symbol(i));
            if (!try fitShared(f.mhc_pf[i], mhc_pf.symbol(i), mhc_pf.smem(mhc_pf.instances[i].fn32), optin)) return error.SharedMemoryTooSmall;
        }
        inline for (rg.rts, 0..) |rt, i| {
            inline for (rg.wide, 0..) |c, j| f.gemv[i][j] = try m_rg.function(rg.gemvSymbol(rt, c[0], c[1]));
            inline for (rg.narrow, 0..) |nw, j| f.narrow[i][j] = try m_rg.function(rg.narrowSymbol(rt, nw));
        }
        inline for (pfd.k2s, 0..) |k2, ki| inline for (.{ false, true }, 0..) |lanes, li| {
            inline for (0..pfd.cfgs.len) |c| {
                if (lanes and k2 == 16) {
                    f.pfd[ki][li][c] = null;
                } else {
                    const fun = try pf[ki].function(pfd.symbol(k2, lanes, c));
                    f.pfd[ki][li][c] = if (try fitShared(fun, pfd.symbol(k2, lanes, c), pfd.smem(c, k2), optin)) fun else null;
                }
            }
            f.pfd_dequant[ki][li] = if (lanes and k2 == 16) null else try pf[ki].function(pfd.dequantSymbol(k2, lanes));
        };
        f.gate_copy = try m_gate.function("_ZN17dsv41_engram_gate16gate_copy_kernelEPxiiPKhxPhxxi");
        f.paced = try m_pace.function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        f.segments = .{ try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb0EEEvPKlij"), try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij") };
        f.topk_keys = try m_keys.function("_ZN15dsv41_topk_keys11topk_kernelEPKfxiiPfPx");
        f.swiglu = try m_pw.function("_ZN15dsv41_pointwise13swiglu_kernelEPKfxS1_xPfxif");
        f.add = try m_pw.function("_ZN15dsv41_pointwise10add_kernelEPfPKfx");
        f.add_bf16 = try m_pw.function("_ZN15dsv41_pointwise15add_bf16_kernelEPKfS1_P13__nv_bfloat16x");
        f.kv_dense = try m_kv.function("_ZN13dsv41_kvsplit12dense_kernelENS_9DenseArgsE");
        f.kv_gather = try m_kv.function("_ZN13dsv41_kvsplit13gather_kernelEPKhjPKjjPh");
        f.kv_pack = m_kv.function("_ZN13dsv41_kvsplit11pack_kernelENS_8PackArgsE") catch null;
        f.kv_dedup = kvsplit.dedupFns(m_kv) catch null;
        inline for (glue.syms, 0..) |name, i| f.glue[i] = try m_glue.function(name);
        f.ds_cands = m_glue.function(glue.ds_cands_sym) catch null;
        f.pick_pack = m_glue.function(glue.pick_pack_sym) catch null;
        f.pick_merge = m_glue.function(glue.pick_merge_sym) catch null;
        f.ds_stage = m_glue.function(glue.ds_stage_sym) catch null;
        f.ds_accept_rows = m_glue.function(glue.ds_accept_rows_sym) catch null;
        f.ds_stage_rows = m_glue.function(glue.ds_stage_rows_sym) catch null;
        f.vs_choose = m_glue.function(glue.vs_choose_sym) catch null;
        inline for (pfglue.syms, 0..) |name, i| f.pfglue[i] = try m_pfglue.function(name);
        return f;
    }
};

/// kvsplit.cu: split KV's exchange copies (kv/split.zig's DenseArgs, the same layout; HostKernels the reference).
pub const kvsplit = struct {
    pub const DenseArgs = extern struct {
        sel: u64,
        rows: u32,
        k: u32,
        table: u64,
        pts: u32,
        rslot: u64,
        psh: u32,
        base: u64,
        row_bytes: u32,
        world: u32,
        send: u64,
        tok: u64,
    };

    /// kv/split.zig's PackArgs (DenseArgs' fields at the same offsets, rank in psh's padding, lens after tok).
    pub const PackArgs = extern struct {
        sel: u64,
        rows: u32,
        k: u32,
        table: u64,
        pts: u32,
        rslot: u64,
        psh: u32,
        rank: u32,
        base: u64,
        row_bytes: u32,
        world: u32,
        send: u64,
        tok: u64,
        lens: u64,
    };

    /// kv/split.zig's DedupArgs: PackArgs, then the map and the scratch.
    pub const DedupArgs = extern struct {
        p: PackArgs,
        map: u64,
        first: u64,
        place: u64,
        bcnt: u64,
    };

    const dedup_syms = [4][:0]const u8{
        "_ZN13dsv41_kvsplit17dedup_mark_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit18dedup_flags_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit17dedup_pack_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit16dedup_tok_kernelENS_9DedupArgsE",
    };

    fn dedupFns(m: cuda.Module) ![4]cuda.Function {
        var out: [4]cuda.Function = undefined;
        for (dedup_syms, &out) |name, *f| f.* = try m.function(name);
        return out;
    }

    comptime {
        std.debug.assert(@sizeOf(DedupArgs) == 120 and @offsetOf(DedupArgs, "map") == 88 and @offsetOf(DedupArgs, "bcnt") == 112);
        std.debug.assert(@sizeOf(DenseArgs) == 80 and @offsetOf(DenseArgs, "rslot") == 32 and @offsetOf(DenseArgs, "send") == 64);
        std.debug.assert(@sizeOf(PackArgs) == 88 and @offsetOf(PackArgs, "rank") == 44 and @offsetOf(PackArgs, "lens") == 80);
    }

    fn grid(words: u64) usize {
        return @intCast(@max(1, @min((words + 255) / 256, 4096)));
    }
};

/// glue.cu: the window's torch glue (forward.py / pick.py), torch's arithmetic bit for bit.
pub const glue = struct {
    pub const syms = [_][:0]const u8{
        "_ZN10dsv41_glue17embed_send_kernelEPKxiPKtxiiPt",
        "_ZN10dsv41_glue16embed_sum_kernelEPKtiiiPt",
        "_ZN10dsv41_glue16cast_bf16_kernelEPKfPtx",
        "_ZN10dsv41_glue18kit_weights_kernelEPKfPfx",
        "_ZN10dsv41_glue17kit_logits_kernelEPfx",
        "_ZN10dsv41_glue16positions_kernelEiiPiPxS0_",
        "_ZN10dsv41_glue18gather_cols_kernelEPKtiiiPt",
        "_ZN10dsv41_glue12carry_kernelEPKtxPKiiiPf",
        "_ZN10dsv41_glue20positions_dev_kernelEPKiiPiPxS2_",
        "_ZN10dsv41_glue16widen_f32_kernelEPKtPfx",
    };
    pub const Kind = enum(u4) { embed_send, embed_sum, cast_bf16, kit_weights, kit_logits, positions, gather_cols, carry, positions_dev, widen_f32 };
    /// DSpark's candidates (optional: resolved apart, so an older fatbin still loads)
    pub const ds_cands_sym: [:0]const u8 = "_ZN10dsv41_glue15ds_cands_kernelEPKfxiiiPf";
    /// the most candidates a rank (the kernel's shared sort)
    pub const ds_cands_max = 128;
    /// DSpark's statics from the device pick (optional, as ds_cands)
    pub const ds_stage_sym: [:0]const u8 = "_ZN10dsv41_glue15ds_stage_kernelEPKxiPxPiiiixiiiiiii";
    pub const ds_accept_rows_sym: [:0]const u8 = "_ZN10dsv41_glue21ds_accept_rows_kernelEPKxS1_S1_iPx";
    pub const ds_stage_rows_sym: [:0]const u8 = "_ZN10dsv41_glue20ds_stage_rows_kernelEPKxS1_iPxPiiiiiiiiiii";
    pub const vs_choose_sym: [:0]const u8 = "_ZN10dsv41_glue16vs_choose_kernelEPKfiiiiyxidddPx";
    /// the candidates vs_choose merges at most (W k)
    pub const vs_max = 2048;
    /// the device pick (optional, as ds_cands)
    pub const pick_pack_sym: [:0]const u8 = "_ZN10dsv41_glue16pick_pack_kernelEPKfPKxixPd";
    pub const pick_merge_sym: [:0]const u8 = "_ZN10dsv41_glue17pick_merge_kernelEPKdiiPKxxPx";
    /// the most rows pick_merge takes (one block)
    pub const pick_max_rows = 1024;

    fn grid(n: u64) usize {
        return @intCast(@max(1, @min((n + 255) / 256, 4096)));
    }
};

/// prefill_glue.cu: the prefill segment's torch glue (block_prefill.zig's glue steps), torch's integers and casts.
pub const pfglue = struct {
    pub const syms = [_][:0]const u8{
        "_ZN12dsv41_pfglue20top_positions_kernelEPKxxiiPi",
        "_ZN12dsv41_pfglue16cand_keys_kernelEPKixiiiPi",
        "_ZN12dsv41_pfglue13counts_kernelEPKiiPKxiiPi",
        "_ZN12dsv41_pfglue15gm_picks_kernelEPKiPKfiiiPiPf",
        "_ZN12dsv41_pfglue17width_mask_kernelEPKixS1_iiPi",
        "_ZN12dsv41_pfglue19f64_bf16_f32_kernelEPKdPfx",
        "_ZN12dsv41_pfglue16widen_cat_kernelEPKtS1_iiPf",
        "_ZN12dsv41_pfglue16ring_copy_kernelEPKhiPhixxx",
    };
    pub const Kind = enum(u3) { top_positions, cand_keys, counts, gm_picks, width_mask, f64_bf16_f32, widen_cat, ring_copy };
    /// index.top_positions' largest `count` (TOPK 512, TOPK_BLOCKS 2,048)
    pub const max_count = 2048;
};

/// topk_keys.cu: pick.top's bounds (C columns a row within pick.COLS, k within the shared sort).
pub const topk_keys = struct {
    pub const max_cols = 1 << 16;
    pub const max_k = 1024;
};

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: usize, shared: u32, cluster: ?cuda.Dim3, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = dim(grid[0]), .y = dim(grid[1]), .z = dim(grid[2]) }, .block = .{ .x = dim(block) }, .shared = shared, .cluster = cluster }, o.s, a);
    }

    /// dsv41_attn_run_cuda: `rows` = the slots table is given (sl), `has_hi` = window ends are given; kw: the Triton
    /// layout's k width (attn_cuda.py reads it from Triton: 4 on Triton 3.8).
    /// `split`: 0 attn_kernel (today's launch), 1 / 4 R1's split_kernel with that many CTAs a chunk (a cluster of 4),
    /// the same bits; with split > 0 the ticket holds R H / 16 * attn.MAXP ints.
    pub fn attention(o: Ops, a: attn.Args, kw: u32, rows: bool, has_hi: bool, split: u32) !void {
        const i = try attn.instance(a, kw, rows, has_hi);
        if (a.qrope != 0 and split == 0) return error.Unsupported; // "q's RoPE folds into the split kernel only"
        var args: cuda.Args = .{};
        args.add(a);
        const hb: usize = @intCast(@divTrunc(a.H, attn.BMQ));
        const R: usize = @intCast(a.R);
        switch (split) {
            0 => try o.go(o.f.attn[i], .{ attn.NCH, hb, R }, attn.NT, attn.SMEM, null, &args),
            1 => try o.go(o.f.attn_split[0][i], .{ attn.NCH, hb, R }, attn.NT, attn.splitSmem(1), null, &args),
            4 => try o.go(o.f.attn_split[1][i], .{ attn.NCH * 4, hb, R }, attn.NT, attn.splitSmem(4), .{ .x = 4 }, &args),
            else => return error.Unsupported,
        }
    }

    /// dsv41_topk_run_cuda: `jobs` 1-2 selections, R rows, CL CTAs a cluster, ept entries a thread.
    pub fn topK(o: Ops, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        return o.topKOf(o.f.topk, a, jobs, R, CL, ept);
    }

    /// topk_b.cu (TF_DSV41_INDEX_BOUND): topK's launch of the row-bounded twin (the same args and bytes).
    pub fn topKB(o: Ops, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        return o.topKOf(o.f.topk_b, a, jobs, R, CL, ept);
    }

    fn topKOf(o: Ops, f: cuda.Function, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        if (R < 1 or jobs < 1 or jobs > 2 or CL < 1 or CL > topk.MAXCL or ept < 1 or ept > topk.MAXEPT) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ CL, R, jobs }, topk.NT, topk.smem(ept), .{ .x = dim(CL) }, &args);
    }

    /// dsv41_mhc_run_cuda: `fn32` = fn is fp32.
    /// `a.spin` >= 0: R1's parallel tail (cnt holds 3 self-resetting ints); `a.defer` = 1 then leaves the
    /// coefficients to mhcCoef on a side stream, joined before any reader of opre / opost / ocomb.
    pub fn mhcBoundary(o: Ops, a: mhc.Args, mode: u32, fn32: bool) !void {
        const i = try mhc.instance(a, mode, fn32);
        var b = a;
        if (b.spin < 0) b.@"defer" = 0; // as the binding: defer only with the tail
        var args: cuda.Args = .{};
        args.add(b);
        try o.go(o.f.mhc[@intFromBool(b.spin >= 0)][i], .{ mhc.NB, if (mhc.instances[i].all4) 1 else 4, 1 }, mhc.NT, @intCast(@as(u32, @intCast(a.R)) * mhc.ROWB), null, &args);
    }

    /// dsv41_mhc_coef_cuda (R1): the next site's coefficients of a deferred boundary, a CTA a row; reads part, base,
    /// scale, R, eps, hc_eps, post_alpha, iters and writes opre / opost / ocomb.
    pub fn mhcCoef(o: Ops, a: mhc.Args) !void {
        if (a.R < 1 or a.R > mhc.COEF_ROWS) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.mhc_coef, .{ @intCast(a.R), 1, 1 }, mhc.NT, 0, null, &args);
    }

    /// dsv41_mhc_pf_run_cuda: `tap` = a tap buffer is given.
    pub fn mhcSite(o: Ops, a: mhc_pf.Args, mode: u32, fn32: bool, tap: bool) !void {
        const i = try mhc_pf.instance(a, mode, fn32, tap);
        var args: cuda.Args = .{};
        args.add(a);
        const R: usize = @intCast(a.R);
        try o.go(o.f.mhc_pf[i], .{ mhc_pf.NB, (R + mhc_pf.BM - 1) / mhc_pf.BM, 1 }, mhc_pf.NT, mhc_pf.smem(mhc_pf.instances[i].fn32), null, &args);
    }

    /// dsv41_rg_route_cuda: (rt, nw, ew) a wide instance, or ew 0 the narrow kernel. The narrow kernel's shared
    /// memory carveout preference (speed only) is left at the driver default.
    pub fn route(o: Ops, a: rg.Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) !void {
        try rg.check(a, rt, nw, ew, lg_len, cnt_len);
        if (a.R == 0) return;
        const ri = std.mem.indexOfScalar(u32, &rg.rts, rt).?;
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        var args: cuda.Args = .{};
        args.add(a);
        if (ew == 0) {
            const ni = std.mem.indexOfScalar(u32, &rg.narrow, nw).?;
            return o.go(o.f.narrow[ri][ni], .{ (E + nw - 1) / nw, (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
        }
        var wi: usize = 0;
        for (rg.wide, 0..) |c, j| if (c[0] == nw and c[1] == ew) {
            wi = j;
        };
        try o.go(o.f.gemv[ri][wi], .{ (E + nw * ew - 1) / (nw * ew), (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
    }

    /// pfdense.cpp gemm: out [M, N] = xh [M, K] (rotated, fp16) through the trellis decoded in the kernel.
    pub fn pfdGemm(o: Ops, a: pfd.Args, k2: u32, lanes: bool, cfg: usize, words: usize) !void {
        try pfd.check(a, k2, lanes, cfg, words);
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2).?;
        const f = o.f.pfd[ki][@intFromBool(lanes)][cfg] orelse return error.Unsupported;
        const bm = pfd.cfgs[cfg][0];
        const M: usize = @intCast(a.M);
        const N: usize = @intCast(a.N);
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ ((M + bm - 1) / bm) * (N >> 7), 1, 1 }, pfd.threads(cfg), pfd.smem(cfg, k2), null, &args);
    }

    /// pfdense.cpp dequant: W_q [K, N] fp16 through the kernel's decode (tests).
    pub fn pfdDequant(o: Ops, T: u64, stride_k: i64, stride_nb: i64, out: u64, K: usize, N: usize, k2: u32, lanes: bool) !void {
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2) orelse return error.Unsupported;
        const f = o.f.pfd_dequant[ki][@intFromBool(lanes)] orelse return error.Unsupported;
        var args: cuda.Args = .{};
        args.add(T);
        args.add(stride_k);
        args.add(stride_nb);
        args.add(out);
        args.add(@as(c_int, @intCast(N)));
        try o.go(f, .{ K / 32, N / 128, 1 }, 128, 0, null, &args);
    }

    /// dsv41_engram_gate_copy: wait (bounded) for the host's ready word, then copy the selected buffer.
    pub fn engramGate(o: Ops, ctl: u64, ready: c_int, err_slot: c_int, base: u64, stride: i64, dst: u64, nbytes: i64, timeout_ns: i64, vec: bool) !void {
        const units: i64 = if (vec) (nbytes + 15) >> 4 else nbytes;
        const blocks: usize = @intCast(std.math.clamp(@divTrunc(units + 255, 256), 1, 64));
        var args: cuda.Args = .{};
        args.add(ctl);
        args.add(ready);
        args.add(err_slot);
        args.add(base);
        args.add(stride);
        args.add(dst);
        args.add(nbytes);
        args.add(timeout_ns);
        args.add(@as(c_int, @intFromBool(vec)));
        try o.go(o.f.gate_copy, .{ blocks, 1, 1 }, 256, 0, null, &args);
    }

    /// dsv41_l2pace launch_paced: (address, bytes) table prefetched to L2 at a paced rate on `ctas` warps.
    pub fn l2Paced(o: Ops, table: u64, n: usize, ctas: usize, chunk: u32, ns_per_piece: u64, delay_ns: u64) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        args.add(ns_per_piece);
        args.add(delay_ns);
        try o.go(o.f.paced, .{ ctas, 1, 1 }, 32, 0, null, &args);
    }

    /// pick.top (topk_keys.cu): each of `rows` rows' best k of bf16-valued fp32 logits [rows, C] (row stride `ld`),
    /// value descending then lower column; vals fp32 [rows, k], cols int64 [rows, k]. Logits that are not
    /// bf16-valued take Python's stable-sort path instead: not this kernel.
    pub fn topKeys(o: Ops, lg: u64, ld: usize, rows: usize, C: usize, k: usize, vals: u64, cols: u64) !void {
        if (rows < 1 or C < 1 or C > topk_keys.max_cols or k < 1 or k > @min(C, topk_keys.max_k)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(lg);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(c_int, @intCast(C)));
        args.add(@as(c_int, @intCast(k)));
        args.add(vals);
        args.add(cols);
        try o.go(o.f.topk_keys, .{ rows, 1, 1 }, 1024, 0, null, &args);
    }

    /// prefill_moe.shared_expert's pointwise part: act = silu(min(g, limit)) * clamp(u, +-limit) (fp32, row strides
    /// in elements).
    pub fn swiglu(o: Ops, g: u64, ldg: usize, u: u64, ldu: usize, act: u64, lda: usize, rows: usize, n: usize, limit: f32) !void {
        var args: cuda.Args = .{};
        args.add(g);
        args.add(@as(i64, @intCast(ldg)));
        args.add(u);
        args.add(@as(i64, @intCast(ldu)));
        args.add(act);
        args.add(@as(i64, @intCast(lda)));
        args.add(@as(c_int, @intCast(n)));
        args.add(limit);
        try o.go(o.f.swiglu, .{ @min((n + 255) / 256, 64), rows, 1 }, 256, 0, null, &args);
    }

    /// out += b over n fp32 elements (prefill_moe.forward: the routed partial plus the shared expert's).
    pub fn addInto(o: Ops, out: u64, b: u64, n: usize) !void {
        var args: cuda.Args = .{};
        args.add(out);
        args.add(b);
        args.add(@as(i64, @intCast(n)));
        try o.go(o.f.add, .{ @min((n + 255) / 256, 4096), 1, 1 }, 256, 0, null, &args);
    }

    /// y = bf16_rn(a + b) over n elements (torch.add(out, shared, out=bf16): TF_DSV41_PF_COPIES).
    pub fn addBf16(o: Ops, a: u64, b: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        args.add(a);
        args.add(b);
        args.add(y);
        args.add(@as(i64, @intCast(n)));
        try o.go(o.f.add_bf16, .{ @min((n + 255) / 256, 4096), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit dense: the selected rows of a split family packed for the all-gather, and their remapped tokens.
    /// `a` is kv/split.zig's DenseArgs (an extern struct of the same layout: pass it through @bitCast).
    pub fn kvDense(o: Ops, a: kvsplit.DenseArgs) !void {
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.psh > 30) return error.Shape;
        const words = @as(u64, a.rows) * a.k * (a.row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.kv_dense, .{ kvsplit.grid(words), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit pack (compact): this rank's owned rows of the selection first in send, tokens owner x R x K + place,
    /// lens[W] each owner's bytes; one launch, block 256, grid ceil(R x K / 256).
    pub fn kvPack(o: Ops, a: kvsplit.PackArgs) !void {
        const f = o.f.kv_pack orelse return error.Unsupported;
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.world > 8 or a.psh > 30) return error.Shape;
        const n = @as(u64, a.rows) * a.k;
        if (n == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ @intCast((n + 255) / 256), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit dedup: the pack over each slot's unique rows (mark, flags, pack, tok: four launches of 256 threads).
    pub fn kvDedup(o: Ops, d: kvsplit.DedupArgs) !void {
        const fs = o.f.kv_dedup orelse return error.Unsupported;
        const a = d.p;
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.world > 8 or a.psh > 30) return error.Shape;
        const n = @as(u64, a.rows) * a.k;
        if (n == 0) return;
        const blocks: usize = @intCast((n + 255) / 256);
        for (fs, [4]usize{ kvsplit.grid(n), blocks, blocks, kvsplit.grid(n) }) |f, g| {
            var args: cuda.Args = .{};
            args.add(d);
            try o.go(f, .{ g, 1, 1 }, 256, 0, null, &args);
        }
    }

    /// kvsplit gather: send[i] = base row phys[i] (uint32 [n]).
    pub fn kvGather(o: Ops, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64) !void {
        if (row_bytes == 0 or row_bytes % 8 != 0) return error.Shape;
        const words = @as(u64, n) * (row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(base);
        args.add(row_bytes);
        args.add(phys);
        args.add(n);
        args.add(send);
        try o.go(o.f.kv_gather, .{ kvsplit.grid(words), 1, 1 }, 256, 0, null, &args);
    }

    fn glueGo(o: Ops, kind: glue.Kind, n: u64, args: *cuda.Args) !void {
        try o.go(o.f.glue[@intFromEnum(kind)], .{ glue.grid(n), 1, 1 }, 256, 0, null, args);
    }

    /// forward.embed_rows' send: rows [n, D] bf16 = embed[ids - vocab_lo] where 0 <= ids - vocab_lo < V, else +0
    /// (`ids` int64 [n], `embed` this rank's bf16 [V, D] slice).
    pub fn embedSend(o: Ops, ids: u64, n: usize, embed: u64, vocab_lo: i64, V: usize, D: usize, rows: u64) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(@as(c_int, @intCast(n)));
        a.add(embed);
        a.add(vocab_lo);
        a.add(@as(c_int, @intCast(V)));
        a.add(@as(c_int, @intCast(D)));
        a.add(rows);
        try o.glueGo(.embed_send, @as(u64, n) * D, &a);
    }

    /// forward.allsum then .to(bf16).repeat(1, 4): recv [W, n, D] bf16 -> out [n, 4 D] bf16 (fp32 adds in rank order
    /// from rank 0's value, one rounding).
    pub fn embedSum(o: Ops, recv: u64, W: usize, n: usize, D: usize, out: u64) !void {
        if (n == 0) return;
        if (W < 1) return error.Shape;
        var a: cuda.Args = .{};
        a.add(recv);
        for ([_]usize{ W, n, D }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(out);
        try o.glueGo(.embed_sum, @as(u64, n) * D, &a);
    }

    /// y bf16 = x.to(bf16) over n elements (the MoE partial into the exchange).
    pub fn castBf16(o: Ops, x: u64, y: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try o.glueGo(.cast_bf16, n, &a);
    }

    /// pick.kit_weights: y = x.to(fp16).to(fp32) over n elements (y may be x).
    pub fn kitWeights(o: Ops, x: u64, y: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try o.glueGo(.kit_weights, n, &a);
    }

    /// pick.kit_logits in place: x = x.to(bf16).to(fp32) over n elements.
    pub fn kitLogits(o: Ops, x: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try o.glueGo(.kit_logits, n, &a);
    }

    /// Win.make's tensors: pos int32 [1] = start, pos64 int64 [n] = start + i, lo int32 [n] = 0.
    pub fn positions(o: Ops, start: i32, n: usize, pos: u64, pos64: u64, lo: u64) !void {
        var a: cuda.Args = .{};
        a.add(start);
        a.add(@as(c_int, @intCast(n)));
        a.add(pos);
        a.add(pos64);
        a.add(lo);
        try o.go(o.f.glue[@intFromEnum(glue.Kind.positions)], .{ (@max(n, 1) + 255) / 256, 1, 1 }, 256, 0, null, &a);
    }

    /// DSpark's candidates (glue.cu ds_cands): each of `rows` fp32 logits rows [rows, V] (row stride ld) -> its best
    /// k by (value desc, column asc) as packed [rows][2 k] fp32 (values, then id0 + column as int bits; -inf / -1
    /// past V), dspark.candidates' result. One CTA a row.
    pub fn dsCandidates(o: Ops, lg: u64, ld: usize, rows: usize, V: usize, k: usize, id0: u32, out: u64) !void {
        const f = o.f.ds_cands orelse return error.Unsupported;
        if (k < 1 or k > glue.ds_cands_max or V < 1) return error.Shape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(lg);
        a.add(@as(i64, @intCast(ld)));
        a.add(@as(i32, @intCast(V)));
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(id0)));
        a.add(out);
        try o.go(f, .{ rows, 1, 1 }, 1024, 0, null, &a);
    }

    /// DSpark's one-slot statics from the device pick `pick` ([nw + 3]: picks, accepted, bonus, next position) into
    /// "w.ds.i64" / "w.ds.i32" (`s64` / `s32`): dspark_gpu.fillStatics' values for (bonus, next position); `o`: the
    /// offsets of positions, tokens, counts, lo, hi, anchor (ids and start at 0).
    pub fn dsStage(o: Ops, pick: u64, nw: usize, s64: u64, s32: u64, n: usize, W: usize, ring: usize, valid: u64, noise: u32, off: [6]i64) !void {
        const f = o.f.ds_stage orelse return error.Unsupported;
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(@as(i32, @intCast(nw)));
        a.add(s64);
        a.add(s32);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(W)));
        a.add(@as(i32, @intCast(ring)));
        a.add(@as(i64, @intCast(valid)));
        a.add(@as(i32, @intCast(noise)));
        for (off) |x| a.add(@as(i32, @intCast(x)));
        try o.go(f, .{ 1, 1, 1 }, 256, 0, null, &a);
    }

    /// A row window's slots' (accepted, bonus, next position) into `acc` [k][3] from the device picks `picks` and the
    /// window's ids `ids` (int64 a row), `segs` [k][3] = (first row, rows, start) int64 (pick_merge's walk a slot).
    pub fn dsAcceptRows(o: Ops, picks: u64, ids: u64, segs: u64, k: usize, acc: u64) !void {
        const f = o.f.ds_accept_rows orelse return error.Unsupported;
        if (k < 1 or k > 32) return error.Shape;
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(ids);
        a.add(segs);
        a.add(@as(i32, @intCast(k)));
        a.add(acc);
        try o.go(f, .{ 1, 1, 1 }, 32, 0, null, &a);
    }

    /// A several-slot pass's statics (dspark_rows.stage's position-dependent values) for `g` members, `mem` [g][3] =
    /// (index in `acc`, slot, valid) int64, from acc's (bonus, next position); `off`: Layout's ids, positions,
    /// tokens, counts, lo, hi.
    pub fn dsStageRows(o: Ops, acc: u64, mem: u64, g: usize, s64: u64, s32: u64, n: usize, W: usize, ring: usize, noise: u32, off: [6]i64) !void {
        const f = o.f.ds_stage_rows orelse return error.Unsupported;
        var a: cuda.Args = .{};
        a.add(acc);
        a.add(mem);
        a.add(@as(i32, @intCast(g)));
        a.add(s64);
        a.add(s32);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(W)));
        a.add(@as(i32, @intCast(ring)));
        a.add(@as(i32, @intCast(noise)));
        for (off) |x| a.add(@as(i32, @intCast(x)));
        try o.go(f, .{ 1, 1, 1 }, 256, 0, null, &a);
    }

    /// vsample.choose over the gathered candidates `g` [W][n][2 k] (count 0: all) for `n` rows drawn at pos0 + r under
    /// one sampling: each row's token into `out` int64 [n] (-1: a nucleus row; speculation only, the host decides).
    pub fn vsChoose(o: Ops, g: u64, W: usize, n: usize, k: usize, count: usize, seed: u64, pos0: u64, top_k: u32, temp: f64, top_p: f64, min_p: f64, out: u64) !void {
        const f = o.f.vs_choose orelse return error.Unsupported;
        if (W * k > glue.vs_max or n == 0) return error.Shape;
        var a: cuda.Args = .{};
        a.add(g);
        a.add(@as(i32, @intCast(W)));
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i32, @intCast(count)));
        a.add(seed);
        a.add(@as(i64, @bitCast(pos0))); // the kernel draws at pos0 + r in int64: a pos0 below 0 (a row-shifted start) wraps back
        a.add(@as(i32, @intCast(top_k)));
        a.add(temp);
        a.add(top_p);
        a.add(min_p);
        a.add(out);
        try o.go(f, .{ n, 1, 1 }, 256, 0, null, &a);
    }

    /// bf16 [n] -> fp32 [n], exact (the DSpark pass's q.float() before _rope).
    pub fn widenF32(o: Ops, src: u64, dst: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i64, @intCast(n)));
        try o.glueGo(.widen_f32, n, &a);
    }

    fn pfGo(o: Ops, kind: pfglue.Kind, n: u64, args: *cuda.Args) !void {
        try o.go(o.f.pfglue[@intFromEnum(kind)], .{ glue.grid(n), 1, 1 }, 256, 0, null, args);
    }

    /// index.top_positions: each of R rows' `count` largest int64 keys (row stride ld, n a row) as their positions,
    /// ascending, -1 padded: out int32 [R, count]. One CTA a row.
    pub fn topPositions(o: Ops, keys: u64, ld: usize, R: usize, n: usize, count: usize, out: u64) !void {
        if (count < 1 or count > pfglue.max_count) return error.Shape;
        if (R == 0) return;
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(i64, @intCast(ld)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(count)));
        a.add(out);
        try o.go(o.f.pfglue[@intFromEnum(pfglue.Kind.top_positions)], .{ R, 1, 1 }, 1024, 0, null, &a);
    }

    /// index.candidate_keys: blocks int32 [R, nb] (row stride ld) -> out int32 [R, nb * bs].
    pub fn candKeys(o: Ops, blocks: u64, ld: usize, R: usize, nb: usize, bs: usize, out: u64) !void {
        if (R * nb == 0) return;
        var a: cuda.Args = .{};
        a.add(blocks);
        a.add(@as(i64, @intCast(ld)));
        for ([_]usize{ R, nb, bs }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(out);
        try o.pfGo(.cand_keys, R * nb * bs, &a);
    }

    /// backend.visible_counts: out int32 [R] = #(sel >= 0 and sel < (pos + 1) / ratio), sel int32 [R, K], pos int64 [R].
    pub fn visibleCounts(o: Ops, sel: u64, K: usize, pos: u64, ratio: u32, R: usize, out: u64) !void {
        if (R == 0) return;
        if (ratio < 1) return error.Shape;
        var a: cuda.Args = .{};
        a.add(sel);
        a.add(@as(c_int, @intCast(K)));
        a.add(pos);
        a.add(@as(c_int, @intCast(ratio)));
        a.add(@as(c_int, @intCast(R)));
        a.add(out);
        try o.go(o.f.pfglue[@intFromEnum(pfglue.Kind.counts)], .{ (R + 7) / 8, 1, 1 }, 256, 0, null, &a);
    }

    /// prefill_moe's routed picks: pk int32 [n, routed] = pick[:, :routed], w6 fp32 [n, routed] = f32(f16(wts[:, :routed]))
    /// (pick / wts [n, ld], the shared slot past topk).
    pub fn gmPicks(o: Ops, pick: u64, wts: u64, ld: usize, routed: usize, n: usize, pk: u64, w6: u64) !void {
        if (n * routed == 0) return;
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(wts);
        for ([_]usize{ ld, routed, n }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(pk);
        a.add(w6);
        try o.pfGo(.gm_picks, n * routed, &a);
    }

    /// x3gm._run_ragged's picks of one width: out[p] = k2tab[pick[p]] == k2 ? pick[p] : E (k2tab int32 [E]; picks
    /// outside [0, E) never match), the input of gmPlan.
    pub fn widthMask(o: Ops, pick: u64, P: usize, k2tab: u64, E: usize, k2: u32, out: u64) !void {
        if (P == 0) return;
        var a: cuda.Args = .{};
        a.add(pick);
        a.add(@as(i64, @intCast(P)));
        a.add(k2tab);
        a.add(@as(c_int, @intCast(E)));
        a.add(@as(c_int, @intCast(k2)));
        a.add(out);
        try o.pfGo(.width_mask, P, &a);
    }

    /// The indexer head weights' casts: y fp32 = f32(bf16(f32(x fp64))), n elements (glue step ix_w).
    pub fn f64Bf16F32(o: Ops, x: u64, y: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try o.pfGo(.f64_bf16_f32, n, &a);
    }

    /// blocks._projection's fp32: out [n, hd] = f32(kv), or (gate != 0, ratio 2) [n, 2 hd] = [f32(kv) | f32(gate)].
    pub fn widenCat(o: Ops, kv: u64, gate: u64, n: usize, hd: usize, out: u64) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(kv);
        a.add(gate);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(hd)));
        a.add(out);
        try o.pfGo(.widen_cat, n * hd * @as(usize, if (gate != 0) 2 else 1), &a);
    }

    /// blocks.stage_rows of one tensor: dst row p % dst_ring = src row p % src_ring for p in [lo, hi), row_bytes a row.
    pub fn ringCopy(o: Ops, src: u64, src_ring: usize, dst: u64, dst_ring: usize, row_bytes: usize, lo: u64, hi: u64) !void {
        if (hi <= lo) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_int, @intCast(src_ring)));
        a.add(dst);
        a.add(@as(c_int, @intCast(dst_ring)));
        a.add(@as(i64, @intCast(row_bytes)));
        a.add(@as(i64, @intCast(lo)));
        a.add(@as(i64, @intCast(hi)));
        try o.pfGo(.ring_copy, (hi - lo) * (row_bytes / 4 + 1), &a);
    }

    /// positions with `start` read from the device (int32 at `start_dev`): a graphed window's form, the same values.
    pub fn positionsDev(o: Ops, start_dev: u64, n: usize, pos: u64, pos64: u64, lo: u64) !void {
        var a: cuda.Args = .{};
        a.add(start_dev);
        a.add(@as(c_int, @intCast(n)));
        a.add(pos);
        a.add(pos64);
        a.add(lo);
        try o.go(o.f.glue[@intFromEnum(glue.Kind.positions_dev)], .{ (@max(n, 1) + 255) / 256, 1, 1 }, 256, 0, null, &a);
    }

    /// forward.gather_cols after the all-gather: recv [W, n, c] -> out [n, W c] (2-byte elements, rank order).
    pub fn gatherCols(o: Ops, recv: u64, W: usize, n: usize, c: usize, out: u64) !void {
        if (n == 0 or c == 0) return;
        var a: cuda.Args = .{};
        a.add(recv);
        for ([_]usize{ W, n, c }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(out);
        try o.glueGo(.gather_cols, @as(u64, W) * n * c, &a);
    }

    /// glue.cu pick_pack: f64 (value, column + id0) pairs [n][2] from topk_keys' k-1 vals fp32 [n] / cols int64 [n].
    pub fn pickPack(o: Ops, vals: u64, cols: u64, n: usize, id0: u64, pairs: u64) !void {
        const f = o.f.pick_pack orelse return error.NoPickKernel;
        var a: cuda.Args = .{};
        a.add(vals);
        a.add(cols);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(i64, @intCast(id0)));
        a.add(pairs);
        try o.go(f, .{ glue.grid(n), 1, 1 }, 256, 0, null, &a);
    }

    /// glue.cu pick_merge: out int64 [n + 3] = each row's pick over the ranks' pairs [W][n][2], then the chain's
    /// accepted rows, the bonus and the next start (ids: the window's tokens int64 [n], 0: no chain).
    pub fn pickMerge(o: Ops, all: u64, W: usize, n: usize, ids: u64, start: i64, out: u64) !void {
        const f = o.f.pick_merge orelse return error.NoPickKernel;
        if (n == 0 or n > glue.pick_max_rows) return error.Shape;
        var a: cuda.Args = .{};
        a.add(all);
        a.add(@as(c_int, @intCast(W)));
        a.add(@as(c_int, @intCast(n)));
        a.add(ids);
        a.add(start);
        a.add(out);
        try o.go(f, .{ 1, 1, 1 }, @intCast(@max(32, std.mem.alignForward(usize, n, 32))), 0, null, &a);
    }

    /// The compressor's carry: carry fp32 [cols] = bf16 row of src (row stride ld elements); the row is *accepted
    /// (int32 on the device, graph-safe) when `accepted` != 0, else `row`.
    pub fn carry(o: Ops, src: u64, ld: usize, accepted: u64, row: usize, cols: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(ld)));
        a.add(accepted);
        a.add(@as(c_int, @intCast(row)));
        a.add(@as(c_int, @intCast(cols)));
        a.add(out);
        try o.glueGo(.carry, cols, &a);
    }

    /// tfl2pf launch_segments: (address, bytes) table prefetched to L2 (bulk: cp.async.bulk.prefetch).
    pub fn l2Segments(o: Ops, table: u64, n: usize, grid: usize, threads: usize, chunk: u32, bulk: bool) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        try o.go(o.f.segments[@intFromBool(bulk)], .{ grid, 1, 1 }, threads, 0, null, &args);
    }
};

test "pfdense shared memory and heuristic as pfdense.py" {
    // cfg 0 (128, 2, 4, 2, 3) at K2 10: 3 x (128 x 32 x 2 + 2 x 8 x 40 x 4) + 2 x 8192 = 3 x 10752 + 16384
    try std.testing.expectEqual(@as(u32, 3 * 10752 + 16384), pfd.smem(0, 10));
    try std.testing.expectEqual(@as(u32, 256), pfd.threads(0));
    // a 2,048-row [5120 -> 1024] call on 48 SMs: 16 x 8 = 128 CTAs of BM 128
    const h = pfd.heuristic(5120, 1024, 10, 2048, 48);
    try std.testing.expectEqual(@as(usize, 8), h[1]);
}

test "mhc instances follow dispatch" {
    var a: mhc.Args = .{ .R = 3 };
    try std.testing.expectEqual(@as(usize, 6), try mhc.instance(a, 3, true));
    try std.testing.expectEqual(@as(usize, 3), try mhc.instance(a, 1, true));
    a.R = 17;
    try std.testing.expectError(error.Shape, mhc.instance(a, 0, false));
    const p: mhc_pf.Args = .{ .R = 40 };
    try std.testing.expectEqual(@as(usize, 5), try mhc_pf.instance(p, 1, true, false));
    try std.testing.expectError(error.Shape, mhc_pf.instance(p, 1, false, true));
    try std.testing.expectEqual(@as(u32, 47872), mhc_pf.smem(false));
}
