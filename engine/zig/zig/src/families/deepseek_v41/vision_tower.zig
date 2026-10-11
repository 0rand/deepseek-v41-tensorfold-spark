//! DeepSeek-V4.1's vision tower on rank 0 (vision.py ``Tower``, ``TF_DSV41_IMAGES=native``): the ViT (patch embedding
//! 588 -> 1,024, 32 pre-norm blocks with a 2D RoPE and full attention within the image, SwiGLU 1,024 -> 2,816, a final
//! RMSNorm) and the aligner (3 x 3 unfold, 9,216 -> 5,120, GELU, 5,120 -> 5,120), BF16 (0.97 GB), with the span's
//! delimiter rows. The torch ops' bits come from vision.cu (each op torch's order) and torch's two GEMM paths: a
//! linear with a bias is addmm's cuBLASLt call (``gemm_and_bias``: fp32 compute, the bias epilogue, the heuristic's
//! first algorithm under a 1 MiB workspace and the pointers' alignments), one without is ``mm``'s cublasGemmEx
//! (vision_blas.zig). ``Encoder`` caches recent images' span rows by digest (TF_DSV41_VISION_CACHE_MB, default 64).
const std = @import("std");
const cuda = @import("cuda");
const pack_mod = @import("pack.zig");
const vk = @import("dsv41_kernels").vision;
const held = @import("dsv41_serve").vision.held;
const lt_mod = cuda.cublaslt;
const blas_mod = @import("vision_blas.zig");

const log = std.log.scoped(.dsv41);
const Allocator = std.mem.Allocator;

pub const prefixes = [_][]const u8{ "vision.", "aligner.", "image_start", "image_end", "image_newline" };

/// ``vision_config`` (config.json).
pub const Shape = struct {
    layers: u32 = 32,
    heads: u32 = 16,
    dim: u32 = 1024,
    inter: u32 = 2816,
    patch: u32 = 14,
    ratio: u32 = 3,
    theta: f32 = 10000,
    out: u32 = 5120,
};

const epilogue_attr: c_int = 7; // CUBLASLT_MATMUL_DESC_EPILOGUE
const bias_pointer_attr: c_int = 8; // CUBLASLT_MATMUL_DESC_BIAS_POINTER
const epilogue_bias: u32 = 4; // CUBLASLT_EPILOGUE_BIAS
const pref_min_alignment_a: c_int = 5; // CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES (B 6, C 7, D 8)

pub const Tower = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    shape: Shape,
    k: vk.Functions,
    lt: lt_mod.Library,
    handle: lt_mod.Handle,
    blas: blas_mod.Blas,
    /// torch's cuBLAS workspace (32 MiB); cuBLASLt shares it (``TORCH_CUBLASLT_UNIFIED_WORKSPACE``) at `lt_ws` bytes
    workspace: cuda.DeviceBuffer,
    lt_ws: usize,
    weights: std.StringHashMapUnmanaged(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,
    scratch: ?cuda.DeviceBuffer = null,

    /// Every vision tensor of `pack` uploaded (BF16, as stored), the kernels and cuBLASLt opened.
    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, pack: *const pack_mod.Pack, shape: Shape) !*Tower {
        const t = try gpa.create(Tower);
        t.* = init(gpa, d, shape) catch |e| {
            gpa.destroy(t);
            return e;
        };
        errdefer t.deinit();
        var it = pack.names.iterator();
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            const keep = for (prefixes) |p| {
                if (std.mem.startsWith(u8, name, p)) break true;
            } else false;
            if (!keep) continue;
            const info = e.value_ptr.*;
            if (info.dtype != .bf16) return error.VisionDtype;
            try buf.resize(gpa, @intCast(info.nbytes));
            try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, buf.items);
            const dev = try cuda.DeviceBuffer.fromHost(d, buf.items);
            try t.weights.put(gpa, try gpa.dupe(u8, name), dev);
            t.bytes += info.nbytes;
        }
        if (t.weights.count() == 0) return error.NoVisionTower;
        log.info("vision tower: {d} tensors, {d:.2} GB on this rank", .{ t.weights.count(), @as(f64, @floatFromInt(t.bytes)) / 1e9 });
        return t;
    }

    /// The kernels, cuBLASLt, cuBLAS and the shared workspace; no weights yet.
    fn init(gpa: Allocator, d: *const cuda.Driver, shape: Shape) !Tower {
        var lt = try lt_mod.Library.open();
        errdefer lt.close();
        var handle: lt_mod.Handle = null;
        try lt.check(lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(handle);
        var blas = try blas_mod.Blas.open();
        errdefer blas.close();
        // ``CUBLASLT_WORKSPACE_SIZE`` (KiB, default 1,024), capped by the cuBLAS workspace as torch caps it
        const lt_ws = @min(envUsize("CUBLASLT_WORKSPACE_SIZE", 1024) << 10, blas_mod.workspace_bytes);
        var ks = try vk.Functions.load(d);
        errdefer ks.unload();
        const ws = try cuda.DeviceBuffer.alloc(d, blas_mod.workspace_bytes);
        return .{ .gpa = gpa, .d = d, .shape = shape, .k = ks, .lt = lt, .handle = handle, .blas = blas, .workspace = ws, .lt_ws = lt_ws };
    }

    pub fn deinit(t: *Tower) void {
        var it = t.weights.iterator();
        while (it.next()) |e| {
            t.gpa.free(e.key_ptr.*);
            e.value_ptr.free();
        }
        t.weights.deinit(t.gpa);
        if (t.scratch) |*s| s.free();
        t.workspace.free();
        t.blas.close();
        _ = t.lt.api.cublasLtDestroy(t.handle);
        t.lt.close();
        t.k.unload();
        t.gpa.destroy(t);
    }

    fn w(t: *const Tower, comptime fmt: []const u8, args: anytype) !u64 {
        var nb: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, fmt, args);
        return (t.weights.get(name) orelse return error.MissingVisionWeight).ptr;
    }

    fn wOpt(t: *const Tower, comptime fmt: []const u8, args: anytype) ?u64 {
        var nb: [96]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, fmt, args) catch return null;
        return if (t.weights.get(name)) |b| b.ptr else null;
    }

    /// ``F.linear(x [m, k], W [n, k], b)`` -> y [m, n] bf16: addmm's cuBLASLt call with the bias epilogue, or mm's
    /// cublasGemmEx without a bias.
    fn linear(t: *Tower, s: cuda.Stream, x: u64, xt: bool, m: usize, k: usize, wt: u64, bias: ?u64, n: usize, y: u64) !void {
        const b = bias orelse return t.blas.linear(@ptrCast(s.handle), t.workspace.ptr, x, xt, m, k, wt, n, y);
        const api = t.lt.api;
        var desc: lt_mod.MatmulDesc = null;
        try t.lt.check(api.cublasLtMatmulDescCreate(&desc, lt_mod.compute_32f, .f32), "desc");
        defer _ = api.cublasLtMatmulDescDestroy(desc);
        const ta: c_int = @backingInt(lt_mod.Op.t);
        // `xt`: x a [k, m] buffer read as [m, k] (strides 1, m), passed as cublasCommonArgs passes it: transb T, ldb m
        const tb: c_int = @backingInt(if (xt) lt_mod.Op.t else lt_mod.Op.n);
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, lt_mod.desc_transa, &ta, @sizeOf(c_int)), "transa");
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, lt_mod.desc_transb, &tb, @sizeOf(c_int)), "transb");
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, epilogue_attr, &epilogue_bias, @sizeOf(u32)), "epilogue");
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, bias_pointer_attr, &b, @sizeOf(u64)), "bias");
        var la: lt_mod.Layout = null;
        var lb: lt_mod.Layout = null;
        var lc: lt_mod.Layout = null;
        try t.lt.check(api.cublasLtMatrixLayoutCreate(&la, .bf16, k, n, @intCast(k)), "layout W");
        defer _ = api.cublasLtMatrixLayoutDestroy(la);
        // ``CuBlasLtMatrixLayout(k, m, ldb, transb)``: a transposed operand's rows and columns swap
        if (xt) {
            try t.lt.check(api.cublasLtMatrixLayoutCreate(&lb, .bf16, m, k, @intCast(m)), "layout X");
        } else try t.lt.check(api.cublasLtMatrixLayoutCreate(&lb, .bf16, k, m, @intCast(k)), "layout X");
        defer _ = api.cublasLtMatrixLayoutDestroy(lb);
        try t.lt.check(api.cublasLtMatrixLayoutCreate(&lc, .bf16, n, m, @intCast(n)), "layout Y");
        defer _ = api.cublasLtMatrixLayoutDestroy(lc);
        var pref: lt_mod.Preference = null;
        try t.lt.check(api.cublasLtMatmulPreferenceCreate(&pref), "pref");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = t.lt_ws;
        try t.lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, lt_mod.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "workspace");
        // ``_getAlignment``: A = W, B = x, C = y, D = the bias
        for ([_]u64{ wt, x, y, b }, 0..) |p, i| {
            const al: u32 = blas_mod.alignment(p);
            try t.lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, pref_min_alignment_a + @as(c_int, @intCast(i)), &al, @sizeOf(u32)), "alignment");
        }
        var found: [1]lt_mod.Heuristic = undefined;
        var count: c_int = 0;
        try t.lt.check(api.cublasLtMatmulAlgoGetHeuristic(t.handle, desc, la, lb, lc, lc, pref, 1, &found, &count), "heuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        const one: f32 = 1;
        const zero: f32 = 0;
        try t.lt.check(api.cublasLtMatmul(t.handle, desc, &one, wt, la, x, lb, &zero, y, lc, y, lc, &found[0].algo, t.workspace.ptr, t.lt_ws, s.handle), "matmul");
    }

    fn grow(t: *Tower, bytes: usize) !u64 {
        if (t.scratch == null or t.scratch.?.len < bytes) {
            if (t.scratch) |*b| b.free();
            t.scratch = try cuda.DeviceBuffer.alloc(t.d, bytes);
        }
        return t.scratch.?.ptr;
    }

    /// ``Tower.span``: the image's span rows bf16 [llm_h (llm_w + 1) + 2, out] into `out` (a device buffer the caller
    /// owns), on `s`. `patches` is the image's bf16 patches already on the device.
    /// A stage's output for the gates (``tf-dsv41-m1 vision``): `name`, its device rows and bytes, after the stream.
    pub const Tap = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, s: cuda.Stream, name: []const u8, ptr: u64, bytes: usize) anyerror!void };

    pub fn span(t: *Tower, s: cuda.Stream, img: *const held.HeldImage, patches: u64, out: u64) !void {
        return t.spanTapped(s, img, patches, out, null);
    }

    pub fn spanTapped(t: *Tower, s: cuda.Stream, img: *const held.HeldImage, patches: u64, out: u64, tap: ?Tap) !void {
        const sh = t.shape;
        const n: usize = @as(usize, img.vit_h) * img.vit_w;
        const D: usize = sh.dim;
        const I: usize = sh.inter;
        const r: usize = sh.ratio;
        const lh = (img.vit_h + sh.ratio - 1) / sh.ratio;
        const lw = (img.vit_w + sh.ratio - 1) / sh.ratio;
        if (lh != img.llm_h or lw != img.llm_w) return error.VisionGrid;
        const L: usize = @as(usize, lh) * lw;
        const pdim: usize = 3 * sh.patch * sh.patch;
        // scratch: x, h, o [n, D]; qkv [n, 3D]; q k v [3, n, D]; m [n, 2I] (then mm [n, I]); [9D, L] + [L, out] x 2
        const sizes = [_]usize{ n * D, n * D, n * D, n * 3 * D, 3 * n * D, n * 2 * I, n * I, L * r * r * D, L * sh.out, L * sh.out };
        var offs: [sizes.len]usize = undefined;
        var total: usize = 0;
        for (sizes, &offs) |z, *o| {
            o.* = total;
            total += std.mem.alignForward(usize, z * 2, 256);
        }
        const base = try t.grow(total);
        const x = base + offs[0];
        const h = base + offs[1];
        const o = base + offs[2];
        const qkv = base + offs[3];
        const q = base + offs[4];
        const kk = q + n * D * 2;
        const v = kk + n * D * 2;
        const m = base + offs[5];
        const mm = base + offs[6];
        const un = base + offs[7];
        const a1 = base + offs[8];
        const feats = base + offs[9];
        const ks = t.k.ops(s);
        try t.linear(s, patches, false, n, pdim, try t.w("vision.patch_embed.proj.weight", .{}), t.wOpt("vision.patch_embed.proj.bias", .{}), D, x);
        if (tap) |tp| try tp.at(tp.ctx, s, "embed", x, n * D * 2);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D / sh.heads)));
        const scale_log2: f32 = @floatCast(@as(f64, scale) * std.math.log2e);
        for (0..sh.layers) |i| {
            // block 0's every op as its own stage ("b0.<op>", Python's layout), so a gate names the op
            const t0: ?Tap = if (i == 0) tap else null;
            try ks.rms(x, try t.w("vision.blocks.{d}.norm1.weight", .{i}), h, n, D, 1e-6);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.norm1", h, n * D * 2);
            try t.linear(s, h, false, n, D, try t.w("vision.blocks.{d}.attn.wqkv.weight", .{i}), t.wOpt("vision.blocks.{d}.attn.wqkv.bias", .{i}), 3 * D, qkv);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.qkv", qkv, n * 3 * D * 2);
            try ks.rope(qkv, q, kk, v, n, img.vit_w, sh.heads, sh.theta);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.rope", q, 3 * n * D * 2); // q k v as [3, heads, n, 64]
            try ks.attn(q, kk, v, o, n, sh.heads, scale_log2);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.attn", o, n * D * 2);
            try t.linear(s, o, false, n, D, try t.w("vision.blocks.{d}.attn.wo.weight", .{i}), t.wOpt("vision.blocks.{d}.attn.wo.bias", .{i}), D, h);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.wo", h, n * D * 2);
            try ks.add(x, h, n * D);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.res1", x, n * D * 2);
            try ks.rms(x, try t.w("vision.blocks.{d}.norm2.weight", .{i}), h, n, D, 1e-6);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.norm2", h, n * D * 2);
            try t.linear(s, h, false, n, D, try t.w("vision.blocks.{d}.mlp.w1.weight", .{i}), t.wOpt("vision.blocks.{d}.mlp.w1.bias", .{i}), 2 * I, m);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.w1", m, n * 2 * I * 2);
            try ks.siluMul(m, mm, n, I);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.silu", mm, n * I * 2);
            try t.linear(s, mm, false, n, I, try t.w("vision.blocks.{d}.mlp.w2.weight", .{i}), t.wOpt("vision.blocks.{d}.mlp.w2.bias", .{i}), D, h);
            if (t0) |tp| try tp.at(tp.ctx, s, "b0.w2", h, n * D * 2);
            try ks.add(x, h, n * D);
            if (tap) |tp| {
                var nb: [32]u8 = undefined;
                try tp.at(tp.ctx, s, try std.fmt.bufPrint(&nb, "block{d}", .{i}), x, n * D * 2);
            }
        }
        try ks.rms(x, try t.w("vision.norm.weight", .{}), h, n, D, 1e-6);
        if (tap) |tp| try tp.at(tp.ctx, s, "vit", h, n * D * 2);
        try ks.unfold(h, un, img.vit_h, img.vit_w, D, r, L);
        if (tap) |tp| try tp.at(tp.ctx, s, "al.unfold", un, L * r * r * D * 2); // F.unfold's [9D, L]
        try t.linear(s, un, true, L, r * r * D, try t.w("aligner.w1.weight", .{}), t.wOpt("aligner.w1.bias", .{}), sh.out, a1);
        if (tap) |tp| try tp.at(tp.ctx, s, "al.w1", a1, L * sh.out * 2);
        try ks.gelu(a1, L * sh.out);
        if (tap) |tp| try tp.at(tp.ctx, s, "gelu", a1, L * sh.out * 2);
        try t.linear(s, a1, false, L, sh.out, try t.w("aligner.w2.weight", .{}), t.wOpt("aligner.w2.bias", .{}), sh.out, feats);
        try ks.span(feats, try t.w("image_start", .{}), try t.w("image_newline", .{}), try t.w("image_end", .{}), out, lh, lw, sh.out);
    }
};

fn envUsize(name: [*:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.trim(u8, std.mem.span(raw), " "), 10) catch default;
}
