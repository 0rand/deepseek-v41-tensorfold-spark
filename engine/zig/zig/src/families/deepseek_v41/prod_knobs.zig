//! Prod's engine knobs (dsv41-tensorfold-spark config/prod.env) that the Zig engine reads under the Python engine's
//! names, where the emitters' prod path (M1_KNOBS=prod) does not already fix them (docs DEEPSEEK-V41-CUDA.md, "Prod knob
//! parity"). Every rank reads the same environment, so the ranks agree; each knob defaults to today's behaviour.
//!
//! - `TF_DSV41_PF_DENSE` = off | fused, `_CFG` = id[,group], `_TABLE` = a JSON tuning table (pfdense.py): the dense EXL3
//!   prefill GEMM with the trellis decoded inside it (`tf_dsv41_pfdense_v1.gemm`) instead of the W_q unpack + Triton
//!   `_gemm` in column blocks. The same bits by construction (pfdense.py's exactness notes: the same mma chain over the
//!   same k16 slices from +0, the same epilogue); the configuration only moves where an output is computed.
//! - `TF_DSV41_PREFILL_CHUNK` / `TF_DSV41_PREFILL_ROWS`: a prompt segment's rows (the smaller of the two; the boot
//!   buffer plan's prefill program is sized for it). Prefill is segmentation-invariant (memory.py, G19).
//! - `TF_DSV41_PREFETCH_AHEAD`: a prompt segment's Engram reads issued before the previous segment runs (prefetch.py).
//! - `TF_DSV41_PREFILL`: full (every layer over every prompt row) | replay (Python's CED bounded replay, ced.zig: the
//!   encoder pass over every row, the decoder over the prompt's last 127 rows). Replay needs the served shape
//!   (TF_DSV41_OWN_PREFILL=1, TF_DSV41_PROMPT_TAIL=verify: model.zig refuses the others, as Python's server is the only
//!   place it runs).
const std = @import("std");
const dk = @import("dsv41_kernels");
const Config = @import("config.zig").Config;

const pfd = dk.ops.pfd;

pub const Error = error{ BadKnob, BadTable };

/// Reads one setting; null when unset or empty (the environment in `env`, a fixture in tests).
pub const Get = *const fn ([]const u8) ?[]const u8;

/// pfdense.SMS: the heuristic prices waves on GB10's 48 SMs on every GPU (the pick never changes a bit, but the
/// launches then match Python's on the pods' captures too).
pub const pfd_sms: usize = 48;

/// The fused dense prefill GEMM's choices (pfdense.pick): an override, the JSON table, else the heuristic.
pub const PfDense = struct {
    entries: []const Entry = &.{},
    /// TF_DSV41_PF_DENSE_CFG: (id, group) for every shape
    override: ?[2]u32 = null,

    pub const Entry = struct { k: u32, n: u32, cfg: u32, group: u32 = 8, small: ?u32 = null, small_rows: u32 = 0 };

    /// pfdense.why_not's shape and width rules (the codebook is mul1 for every dense group of our packs).
    pub fn eligible(k: i64, n: i64, k2: u32, lanes: bool) bool {
        const widths: []const u32 = if (lanes) &.{ 8, 10, 12 } else &.{ 8, 10, 12, 16 };
        if (std.mem.indexOfScalar(u32, widths, k2) == null) return false;
        return k > 0 and n > 0 and @rem(k, 128) == 0 and @rem(n, 128) == 0;
    }

    /// (cfg, group) for an [m, k] @ [k, n] call: the override, the table (its `small` at m <= small_rows), else
    /// the heuristic; a choice whose stage does not divide K falls back to the heuristic.
    pub fn pick(p: *const PfDense, k: usize, n: usize, k2: u32, m: usize) [2]u32 {
        var got: ?[2]u32 = p.override;
        if (got == null) for (p.entries) |e| {
            if (e.k != k or e.n != n) continue;
            const c = if (e.small != null and m <= e.small_rows) e.small.? else e.cfg;
            got = .{ c, e.group };
            break;
        };
        if (got) |g| if (g[0] < pfd.cfgs.len and k % (16 * pfd.cfgs[g[0]][3]) == 0) return g;
        const h = pfd.heuristic(k, n, k2, m, pfd_sms);
        return .{ @intCast(h[0]), @intCast(h[1]) };
    }

    /// pfdense._table_file: `{"K,N": {"cfg": id, "group": g, "small": id, "small_rows": r}}`.
    pub fn parseTable(a: std.mem.Allocator, json: []const u8) ![]const Entry {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadTable;
        if (parsed != .object) return error.BadTable;
        const out = try a.alloc(Entry, parsed.object.count());
        var it = parsed.object.iterator();
        var i: usize = 0;
        while (it.next()) |kv| : (i += 1) {
            var ks = std.mem.tokenizeScalar(u8, kv.key_ptr.*, ',');
            const k = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            const n = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            if (kv.value_ptr.* != .object) return error.BadTable;
            const o = kv.value_ptr.object;
            out[i] = .{ .k = k, .n = n, .cfg = try field(o, "cfg") orelse return error.BadTable };
            if (try field(o, "group")) |g| out[i].group = g;
            out[i].small = try field(o, "small");
            out[i].small_rows = (try field(o, "small_rows")) orelse 0;
        }
        return out;
    }

    fn field(o: std.json.ObjectMap, name: []const u8) !?u32 {
        const v = o.get(name) orelse return null;
        return switch (v) {
            .integer => |x| std.math.cast(u32, x) orelse error.BadTable,
            else => error.BadTable,
        };
    }

    fn parseOverride(raw: []const u8) ![2]u32 {
        var it = std.mem.tokenizeScalar(u8, raw, ',');
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, it.next() orelse return error.BadKnob, " "), 10) catch return error.BadKnob;
        const g: u32 = if (it.next()) |x| std.fmt.parseInt(u32, std.mem.trim(u8, x, " "), 10) catch return error.BadKnob else 8;
        if (it.next() != null or id >= pfd.cfgs.len or g < 1 or g > 1024) return error.BadKnob;
        return .{ id, g };
    }
};

/// TF_DSV41_PF_DENSE (off | fused; 0 / no / false = off): the fused path's settings, null when off. The table file
/// is read through `io` (its bytes and the table owned by `a`).
pub fn pfDense(a: std.mem.Allocator, io: std.Io, get: Get) !?*const PfDense {
    const raw = get("TF_DSV41_PF_DENSE") orelse return null;
    const v = std.mem.trim(u8, raw, " ");
    const off = [_][]const u8{ "off", "0", "no", "false" };
    for (off) |o| if (std.ascii.eqlIgnoreCase(v, o)) return null;
    if (!std.ascii.eqlIgnoreCase(v, "fused")) {
        std.log.warn("TF_DSV41_PF_DENSE={s}: expected off or fused", .{raw});
        return error.BadKnob;
    }
    const p = try a.create(PfDense);
    p.* = .{};
    if (get("TF_DSV41_PF_DENSE_CFG")) |c| p.override = PfDense.parseOverride(c) catch {
        std.log.warn("TF_DSV41_PF_DENSE_CFG={s}: expected id[,group], id 0-{d}, group 1-1024", .{ c, pfd.cfgs.len - 1 });
        return error.BadKnob;
    };
    if (get("TF_DSV41_PF_DENSE_TABLE")) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch |e| {
            std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: {t}: copy the recipe's config/pfdense-table.json to that path on every rank (scripts/run-zig.sh puts it in CACHE_DIR when missing), or unset the knob for the heuristic", .{ path, e });
            return e;
        };
        p.entries = PfDense.parseTable(a, bytes) catch |e| {
            std.log.warn("TF_DSV41_PF_DENSE_TABLE={s}: not a pfdense tuning table ({t})", .{ path, e });
            return e;
        };
    }
    return p;
}

/// pf4k.TARGET: the only prefill segment above 2,048 rows (G16 4K)
pub const pf4k_rows: u32 = 4096;

/// TF_DSV41_PF_4K (pf4k.on / check: "0" or "1"): 4,096-row prefill segments, refused without
/// TF_DSV41_PF_4K_MEMORY_OK=1 (a 4K segment holds about 0.66 GiB more on the worker than a 2K one, pf4k.terms). Unset
/// or "0": off (Python prod's 2,048).
pub fn pf4k(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_4K") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (!std.mem.eql(u8, raw, "1")) {
        std.log.warn("TF_DSV41_PF_4K={s}: 0 or 1", .{raw});
        return error.BadKnob;
    }
    const ok = std.mem.trim(u8, get("TF_DSV41_PF_4K_MEMORY_OK") orelse "", " ");
    if (!std.mem.eql(u8, ok, "1")) {
        std.log.warn("TF_DSV41_PF_4K=1: 4,096-row prefill segments hold about 0.66 GiB more worker memory than 2,048-row ones; set TF_DSV41_PF_4K_MEMORY_OK=1 to acknowledge", .{});
        return error.BadKnob;
    }
    return true;
}

/// A prompt segment's rows: min(TF_DSV41_PREFILL_CHUNK, TF_DSV41_PREFILL_ROWS), on the 16-row grid, 32-2,048; null
/// when neither is set (the forward's default, 2,048). Under TF_DSV41_PF_4K=1: 4,096 (pf4k.chunk; the forward's
/// segment is also the round planner's rows, so an explicit CHUNK or ROWS other than 4,096 is refused rather than
/// silently keeping 2,048-row segments).
pub fn prefillRows(get: Get) !?u32 {
    const big = try pf4k(get);
    var rows: ?u32 = null;
    for ([_][]const u8{ "TF_DSV41_PREFILL_CHUNK", "TF_DSV41_PREFILL_ROWS" }) |name| if (get(name)) |raw| {
        const v = std.fmt.parseInt(u32, std.mem.trim(u8, raw, " "), 10) catch return badRows(name, raw, big);
        if (big and v != pf4k_rows) return badRows(name, raw, big);
        if (v < 32 or (v > 2048 and !big) or v % 16 != 0) return badRows(name, raw, big);
        rows = if (rows) |r| @min(r, v) else v;
    };
    return if (big) pf4k_rows else rows;
}

fn badRows(name: []const u8, raw: []const u8, big: bool) error{BadKnob} {
    if (big) std.log.warn("{s}={s}: TF_DSV41_PF_4K=1 means 4,096-row segments (unset it or set 4096)", .{ name, raw }) else std.log.warn("{s}={s}: expected a multiple of 16 in 32..2048", .{ name, raw });
    return error.BadKnob;
}

/// TF_DSV41_PF_OVERLAP (ours, no Python twin; unset / 0 / 1: off): a prefill segment's two exchanges a layer in this
/// many row pieces (2-8), each on the branches' side stream behind its producer's piece (block_prefill.zig `pieces`).
pub fn pfOverlap(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_OVERLAP") orelse "0", " ");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v < 0 or v > 8) {
        std.log.warn("TF_DSV41_PF_OVERLAP={s}: 0 (off) or row pieces 2-8", .{raw});
        return error.BadKnob;
    }
    return if (v < 2) 0 else v;
}

/// TF_DSV41_MHC_SITE_ROWS (ours; unset / 0 off, else 256-4,096 and a multiple of 16): a prefill boundary site's rows a
/// piece (block_prefill.zig `sitePieces`).
pub fn mhcSiteRows(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_MHC_SITE_ROWS") orelse "0", " ");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v == 0) return 0;
    if (v < 256 or v > 4096 or @rem(v, 16) != 0) {
        std.log.warn("TF_DSV41_MHC_SITE_ROWS={s}: 0 (off) or 256-4096, a multiple of 16", .{raw});
        return error.BadKnob;
    }
    return v;
}

/// TF_DSV41_GM_GU2 (ours; 0 / 1, default 0; with TF_DSV41_GM_V2=1): a 4,096-row segment's routed gate / up at x3gm.cu's
/// 128-member tile GU2 (gm2pf.zig).
pub fn gmGu2(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_GM_GU2") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    std.log.warn("TF_DSV41_GM_GU2={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_OVERLAP_SITE (ours; 0 / 1, default 0; with TF_DSV41_PF_OVERLAP >= 2): every exchange piece on the side
/// stream, and the next boundary mixing site in the same row pieces, each after its own exchange piece
/// (block_prefill.zig `sitePieces`).
pub fn pfOverlapSite(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_OVERLAP_SITE") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    std.log.warn("TF_DSV41_PF_OVERLAP_SITE={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_TBO (ours; 0 / 1, default 0): CED encoder segments two at a time on two streams (block_prefill.emitTbo).
pub fn pfTbo(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_TBO") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    std.log.warn("TF_DSV41_PF_TBO={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PF_TBO_GM_SMS (under TF_DSV41_PF_TBO): the SMs x3gm's persistent grid covers at most, 8-48; 0 (the
/// default: proto/tbo_bench in the device benchmark, best stream split at 4,096 rows uncapped): every SM.
pub const tbo_gm_sms: u32 = 0;
pub fn pfTboGmSms(get: Get) !u32 {
    const raw = std.mem.trim(u8, get("TF_DSV41_PF_TBO_GM_SMS") orelse return tbo_gm_sms, " ");
    const v = std.fmt.parseInt(u32, raw, 10) catch return badSms(raw);
    if (v != 0 and (v < 8 or v > 48)) return badSms(raw);
    return v;
}

fn badSms(raw: []const u8) error{BadKnob} {
    std.log.warn("TF_DSV41_PF_TBO_GM_SMS={s}: 0 (every SM) or 8-48", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_STREAM_RB (ours; unset / 0 off, 1, 2 or 4): the stream top-k's twin (block_prefill streamSelect): 1 the
/// prefetch twin `_stream_pf` (one row a program), 2 / 4 the row-blocked `_stream_rb<N>`.
pub fn streamRb(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_STREAM_RB") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return 0;
    if (std.mem.eql(u8, raw, "1")) return 1;
    if (std.mem.eql(u8, raw, "2")) return 2;
    if (std.mem.eql(u8, raw, "4")) return 4;
    std.log.warn("TF_DSV41_STREAM_RB={s}: 0, 1, 2 or 4", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_INDEX_BOUND (ours, 0 by default): decode windows' dense index scores as `_scores_b` (block.Options.index_bound).
pub fn indexBound(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_INDEX_BOUND") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    std.log.warn("TF_DSV41_INDEX_BOUND={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// TF_DSV41_PREFETCH_AHEAD (0 by default): the next prompt segment's Engram reads before this one runs.
pub fn prefetchAhead(get: Get) bool {
    const v = get("TF_DSV41_PREFETCH_AHEAD") orelse return false;
    return !std.mem.eql(u8, std.mem.trim(u8, v, " "), "0");
}

/// TF_DSV41_KV_NORM_STORE=1: KV-only RMS + SWA RoPE/FP8 store, default off until GPU/reply gates pass.
pub fn kvNormStore(get: Get) bool {
    const v = get("TF_DSV41_KV_NORM_STORE") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
}

/// TF_DSV41_ROUTER_GROUP_ROT=1: histogram grouping + input rotation, default off until GPU/reply gates pass.
pub fn routerGroupRot(get: Get) bool {
    const v = get("TF_DSV41_ROUTER_GROUP_ROT") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
}

/// TF_DSV41_X3LD_EPI (ours, no Python twin; unset / 0 off, 1 on, anything else refused): the decode MoE's expert
/// epilogues in x3ld's tail (block.Options.x3ld_epi, x3ld_epi.cu).
pub fn x3ldEpi(get: Get) !bool {
    const raw = std.mem.trim(u8, get("TF_DSV41_X3LD_EPI") orelse "0", " ");
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    std.log.warn("TF_DSV41_X3LD_EPI={s}: 0 or 1", .{raw});
    return error.BadKnob;
}

/// prod-perf1's decode R1 under Python's names (ATTN_SPLIT=4 and the seven switches at 1), or TF_DSV41_R1 (ours,
/// which wins): Zig runs R1 whole or not at all, so a partial set is logged and runs without it (R1 is the same bits).
pub fn r1(get: Get) bool {
    if (get("TF_DSV41_R1")) |v| return std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
    const names = [_][]const u8{ "TF_DSV41_ATTN_SPLIT", "TF_DSV41_ATTN_ROPE", "TF_DSV41_MHC_TAIL", "TF_DSV41_MHC_DEFER", "TF_DSV41_MOE_BF16", "TF_DSV41_PRUNE_KIT", "TF_DSV41_RG_PRUNE", "TF_DSV41_RMS_FOLD" };
    var set: usize = 0;
    var prod: usize = 0;
    for (names, 0..) |n, i| if (get(n)) |raw| {
        set += 1;
        if (std.mem.eql(u8, std.mem.trim(u8, raw, " "), if (i == 0) "4" else "1")) prod += 1;
    };
    if (prod == names.len) return true;
    if (set > 0) std.log.warn("decode R1: {d} of its {d} Python knobs at prod's values; Zig runs R1 whole (TF_DSV41_R1=1, or all of ATTN_SPLIT=4 ATTN_ROPE MHC_TAIL MHC_DEFER MOE_BF16 PRUNE_KIT RG_PRUNE RMS_FOLD = 1) or not at all: off", .{ prod, names.len });
    return false;
}

/// TF_DSV41_INDEX_BUDGET_MIB (Python's default 256; prod 64): the indexer's scores and sort keys a launch, in bytes;
/// prefill's row blocks follow it (csa2 backend; the same bits at any budget), so its scratch shrinks with it.
/// TF_DSV41_MHC_PFDEC (ours, block_wide.zig): unset / 0 off, else the fewest rows a window's mixing mHC sites take
/// mhc_pf + coef_kernel at (1-64; 17: only the windows past mhc_cuda's 16 rows, which run Triton today).
pub fn mhcPfRows(get: Get) !i64 {
    const raw = std.mem.trim(u8, get("TF_DSV41_MHC_PFDEC") orelse return 0, " \t");
    const v = std.fmt.parseInt(i64, raw, 10) catch -1;
    if (v < 0 or v > 64) {
        std.log.warn("TF_DSV41_MHC_PFDEC={s}: expected 0 (off) or the fewest rows, 1..64", .{raw});
        return error.BadKnob;
    }
    return v;
}

pub fn indexBudget(get: Get) !?i64 {
    const raw = get("TF_DSV41_INDEX_BUDGET_MIB") orelse return null;
    const v = std.fmt.parseInt(i64, std.mem.trim(u8, raw, " "), 10) catch return badBudget(raw);
    if (v < 1 or v > 4096) return badBudget(raw);
    return v << 20;
}

fn badBudget(raw: []const u8) error{BadKnob} {
    std.log.warn("TF_DSV41_INDEX_BUDGET_MIB={s}: expected 1..4096", .{raw});
    return error.BadKnob;
}

/// block_prefill's stream threshold (TF_DSV41_INDEX_STREAM_MIN, prod 4,096 visible keys): the emitter's constant.
pub const stream_min: i64 = 4096;

/// The long-prompt stream top-k's scratch ("L.ix.stream", longpf.zig) the buffer plan holds for a slot of `limit`
/// positions and `rows`-row segments: the largest one any streamed layer takes at the limit (stream_topk.plan's rows x
/// splits x 2 K int64, at most stream_topk.BUDGET), 0 when no layer reaches the threshold. Python allocates it per
/// launch, so its measured terms (kv/budget.zig Serve) hold none; Zig plans it at boot, resident.
pub fn streamScratch(cfg: *const Config, limit: i64, rows: i64) u64 {
    const longpf = @import("longpf.zig");
    const k: i64 = cfg.index_topk;
    if (rows <= 16 or @popCount(k) != 1) return 0;
    var most: i64 = 0;
    for (0..cfg.layers) |i| {
        const L: u32 = @intCast(i);
        if (cfg.mode(L) != .full or !cfg.isIndexSource(L) or cfg.isCandidateSource(L)) continue;
        const keys = @divFloor(limit, cfg.compressRatio(L));
        if (keys < stream_min) continue;
        most = @max(most, longpf.plan(rows, keys, k) * longpf.splits(keys) * 2 * k * 8);
    }
    return @intCast(most);
}

pub const PrefillMode = enum { full, replay };

/// TF_DSV41_PREFILL: the prompt's prefill mode (forward_prefill.State.mode, ced.zig). Unset is `full` here (new
/// behaviour defaults off until its gate passes); Python's server defaults to `replay` (stack.py DEFAULT_PREFILL) and
/// prod sets it, so Zig-as-prod sets it too (prod-zig.env).
pub fn prefillMode(get: Get) !PrefillMode {
    const raw = get("TF_DSV41_PREFILL") orelse return .full;
    return std.meta.stringToEnum(PrefillMode, std.mem.trim(u8, raw, " ")) orelse {
        std.log.warn("TF_DSV41_PREFILL={s}: expected full or replay", .{raw});
        return error.BadKnob;
    };
}

/// Where a served prompt's last token runs (`TF_DSV41_PROMPT_TAIL`, rank 0's choice: the followers replay its forward
/// operations). `verify` (default) is Python's server (batch.py `_pieces` / `finals`, rounds.py): the rows before
/// the last token are prefilled and the last token is the reply's first decode row, the pending row of its first
/// verify window (decode kernels, the decode expert top-p, R1), so its KV and the first choice are a decode row's.
/// `prefill`: every prompt row through the prefill, as Python's Forward.prompt (the gates' references that call it).
pub const PromptTail = enum { verify, prefill };

pub fn promptTail(get: Get) !PromptTail {
    const raw = get("TF_DSV41_PROMPT_TAIL") orelse return .verify;
    return std.meta.stringToEnum(PromptTail, std.mem.trim(u8, raw, " ")) orelse {
        std.log.warn("TF_DSV41_PROMPT_TAIL={s}: expected verify or prefill", .{raw});
        return error.BadKnob;
    };
}

/// A prompt of `n` rows in `tail`'s shape: the rows through the prefill, then the rows of one decode window.
pub fn promptSplit(n: usize, tail: PromptTail) struct { prefill: usize, window: usize } {
    return switch (tail) {
        .prefill => .{ .prefill = n, .window = 0 },
        .verify => .{ .prefill = n -| 1, .window = @min(n, 1) },
    };
}

/// The knobs onto a forward before its buffer plan, on every rank (model.zig serving, m2b.zig's gates): the fused
/// dense prefill GEMM, the segment rows, the indexer's budget, the Engram read-ahead of the next segment, x3gm v2, the
/// prefill mode (CED replay: the model's decoder layer checked). R1 is read where the forward is made (`r1`).
pub fn apply(f: *@import("forward.zig").Forward, a: std.mem.Allocator, io: std.Io, rank: u32) !void {
    f.opts.kv_norm_store = kvNormStore(&env);
    f.opts.router_group_rot = routerGroupRot(&env);
    f.opts.x3ld_epi = try x3ldEpi(&env);
    f.opts.pfd = try pfDense(a, io, &env);
    // TF_DSV41_PF_4K: the forward's own options (decode windows, the drafter, the buffer plan) keep 2,048-row
    // segments byte for byte; 4,096-row segments run in the prefill's own workspace (forward_prefill.K4)
    const big = try pf4k(&env);
    if (try prefillRows(&env)) |r| if (!big) {
        f.opts.prefill_rows = r;
    };
    if (try indexBudget(&env)) |b| f.opts.index_budget = b;
    f.prefill_state.ahead = prefetchAhead(&env);
    f.opts.gm_v2 = try @import("gm2pf.zig").mode(&env);
    f.opts.mhc_pf_rows = try mhcPfRows(&env);
    f.prefill_state.mode = try prefillMode(&env);
    f.prefill_state.k4 = big;
    // x3gm v1 at 4K runs pf4k's tuning table (128-member passes past 48 members an expert), which the emitter does not
    // take; v2 keeps G6's tiles at any rows (x3gm.tuned2), so 4K is v2's alone; the workspace is released at a
    // prompt's CED finish, so 4K is replay's alone
    if (big and (f.opts.gm_v2 == .off or f.prefill_state.mode != .replay)) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_4K=1 needs TF_DSV41_GM_V2=1 (or split) and TF_DSV41_PREFILL=replay", .{});
        return error.BadKnob;
    }
    f.opts.pf_tbo = try pfTbo(&env);
    f.opts.stream_rb = try streamRb(&env);
    f.opts.index_bound = try indexBound(&env);
    // TF_DSV41_BRANCHES: the side stream exists before any capture
    if (f.branches == null) f.branches = try @import("branches.zig").install(f.gpa, f, rank);
    // TF_DSV41_PF_OVERLAP: the pieces' exchanges go on the branches' side stream (one peer: TP=2)
    f.opts.pf_overlap = try pfOverlap(&env);
    f.opts.pf_overlap_site = try pfOverlapSite(&env);
    f.opts.gm_gu2 = try gmGu2(&env);
    f.opts.mhc_site_rows = try mhcSiteRows(&env);
    if (f.opts.mhc_site_rows > 0) std.log.scoped(.dsv41).info("mhc: prefill boundary sites in pieces of about {d} rows", .{f.opts.mhc_site_rows});
    if (f.opts.gm_gu2 and f.opts.gm_v2 != .one) {
        std.log.scoped(.dsv41).err("TF_DSV41_GM_GU2=1 needs TF_DSV41_GM_V2=1 (x3gm v2's one plan a block)", .{});
        return error.BadKnob;
    }
    if (f.opts.gm_gu2) std.log.scoped(.dsv41).info("x3gm: gate / up at 128-member passes (GU2) in 4,096-row segments", .{});
    if (f.opts.pf_overlap_site and f.opts.pf_overlap < 2) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_OVERLAP_SITE=1 needs TF_DSV41_PF_OVERLAP=2..8 (the pieces it overlaps)", .{});
        return error.BadKnob;
    }
    // TF_DSV41_PF_TBO: the second micro-batch on the side stream, CED's encoder segments only; x3gm's grid capped
    if (f.opts.pf_tbo) {
        if (!f.opts.branches or f.prefill_state.mode != .replay) {
            std.log.scoped(.dsv41).err("TF_DSV41_PF_TBO=1 needs TF_DSV41_BRANCHES=1 (its side stream) and TF_DSV41_PREFILL=replay", .{});
            return error.BadKnob;
        }
        dk.exl3.gm_sms_cap = try pfTboGmSms(&env);
    }
    if (f.opts.pf_overlap > 0 and (!f.opts.branches or f.opts.world != 2)) {
        std.log.scoped(.dsv41).err("TF_DSV41_PF_OVERLAP={d} needs TF_DSV41_BRANCHES=1 (its side stream) and TP=2 (world {d})", .{ f.opts.pf_overlap, f.opts.world });
        return error.BadKnob;
    }
    if (f.prefill_state.mode == .replay) {
        const ced = @import("ced.zig");
        const d = ced.decoderStart(f.cfg) catch |e| {
            std.log.scoped(.dsv41).err("TF_DSV41_PREFILL=replay: no decoder layer in this model ({t}: the first ratio-1 layer must be a kv source and no later layer Engram's)", .{e});
            return e;
        };
        if (std.mem.indexOfScalar(u32, try f.backbone(a), d) == null) {
            std.log.scoped(.dsv41).err("TF_DSV41_PREFILL=replay: the forward's layers do not reach the decoder's first ({d})", .{d});
            return error.BadKnob;
        }
    }
    if (rank != 0) return;
    const log = std.log.scoped(.dsv41);
    log.info("KV norm/store fusion: {s}", .{if (f.opts.kv_norm_store) "on" else "off"});
    log.info("router grouping/rotation fusion: {s}", .{if (f.opts.router_group_rot) "on" else "off"});
    if (f.opts.index_bound) log.info("index bound: decode index scores stop at each row's own keys (_scores_b)", .{});
    if (f.opts.x3ld_epi) log.info("x3ld epilogues: gateup_epilogue / down_combine fused into the decode expert streamer (TF_DSV41_X3LD_EPI)", .{});
    if (f.opts.stream_rb == 1) log.info("stream top-k: _stream_pf (the next tile's key loads issued ahead)", .{});
    if (f.opts.stream_rb > 1) log.info("stream top-k: {d} rows a program (_stream_rb{d})", .{ f.opts.stream_rb, f.opts.stream_rb });
    if (f.opts.pf_tbo) log.info("prefill tbo: encoder segments two at a time on two streams, x3gm on at most {d} SMs (0: all){s}", .{ dk.exl3.gm_sms_cap, if (f.opts.pf_overlap > 0) "; TF_DSV41_PF_OVERLAP is off under it" else "" });
    if (f.opts.pf_overlap > 0) log.info("prefill overlap: a segment's exchanges in {d} row pieces on the side stream{s}", .{ f.opts.pf_overlap, if (f.opts.pf_overlap_site) ", every piece; the next mixing site in the same pieces, each after its own" else "" });
    log.info("prod knobs: pf_dense {s}, prefill rows {d}{s}, index budget {d} MiB, R1 {s}, prefetch ahead {s}, x3gm v2 {t}, prefill {t}, mhc pfdec {d}", .{ if (f.opts.pfd != null) "fused" else "off", if (big) pf4k_rows else @as(u32, @intCast(f.opts.prefill_rows)), if (big) " (4K: one x3gm block a segment, in the prefill's own workspace)" else "", f.opts.index_budget >> 20, if (f.opts.r1) "on" else "off", if (f.prefill_state.ahead) "on" else "off", f.opts.gm_v2, f.prefill_state.mode, f.opts.mhc_pf_rows });
}

pub fn env(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.mem.span(std.c.getenv(buf[0..name.len :0]) orelse return null);
    return if (v.len == 0) null else v;
}
