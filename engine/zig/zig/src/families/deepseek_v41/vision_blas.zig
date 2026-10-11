//! Plain cuBLAS (libcublas, opened at run time) for the vision tower's unbiased linears: torch's ``F.linear`` without a
//! bias is ``mm``, and with the preferred BLAS backend cuBLAS that is ``gemm_internal_cublas_bfloat16_helper``:
//! ``cublasSetMathMode(DEFAULT_MATH)`` (bf16 reduced-precision reduction allowed), ``cublasGemmEx`` with fp32 compute
//! and ``CUBLAS_GEMM_DEFAULT_TENSOR_OP``, on a handle whose stream and 32 MiB workspace are set before each call
//! (``getCurrentCUDABlasHandle``: set stream, then workspace, as setting the stream resets it).
const std = @import("std");

pub const Status = c_int;
pub const Handle = ?*opaque {};

const op_n: c_int = 0;
const op_t: c_int = 1;
const r_16bf: c_int = 14; // CUDA_R_16BF
const compute_32f: c_int = 68; // CUBLAS_COMPUTE_32F (what CUDA_R_32F migrates to under DEFAULT_MATH)
const default_math: c_int = 0; // CUBLAS_DEFAULT_MATH
const gemm_default_tensor_op: c_int = 99; // CUBLAS_GEMM_DEFAULT_TENSOR_OP

/// torch's cuBLAS workspace on major 9 / 10 / 12 (``parseChosenWorkspaceSize``: 4096 * 8 KiB).
pub const workspace_bytes: usize = 32 << 20;

const S = Status;
const CPtr = ?*const anyopaque;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, ?*anyopaque) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, u64, usize) callconv(.c) S,
    cublasSetMathMode: *const fn (Handle, c_int) callconv(.c) S,
    cublasGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasGemmEx: *const fn (Handle, c_int, c_int, c_int, c_int, c_int, CPtr, u64, c_int, c_int, u64, c_int, c_int, CPtr, u64, c_int, c_int, c_int, c_int) callconv(.c) S,
};

pub const Blas = struct {
    lib: std.DynLib,
    api: Api,
    handle: Handle = null,

    pub fn open() !Blas {
        for ([_][]const u8{ "libcublas.so.13", "libcublas.so" }) |path| {
            var lib = std.DynLib.open(path) catch continue;
            errdefer lib.close();
            var b: Blas = .{ .lib = lib, .api = undefined };
            const info = @typeInfo(Api).@"struct";
            inline for (info.field_names, info.field_types) |name, T| {
                @field(b.api, name) = lib.lookup(T, name) orelse {
                    std.log.err("{s} has no {s}", .{ path, name });
                    return error.MissingSymbol;
                };
            }
            try b.check(b.api.cublasCreate_v2(&b.handle), "cublasCreate");
            return b;
        }
        return error.LibraryUnavailable;
    }

    pub fn close(b: *Blas) void {
        _ = b.api.cublasDestroy_v2(b.handle);
        b.lib.close();
    }

    fn check(b: *const Blas, s: Status, what: []const u8) !void {
        if (s == 0) return;
        const n = b.api.cublasGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (n) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }

    /// ``F.linear(x [m, k], W [n, k])`` -> y [m, n] bf16: column-major y^T [n, m] = W^T' x^T (opT on W, lda = ldb = k).
    pub fn linear(b: *const Blas, stream: ?*anyopaque, workspace: u64, x: u64, xt: bool, m: usize, k: usize, w: u64, n: usize, y: u64) !void {
        const api = b.api;
        try b.check(api.cublasSetStream_v2(b.handle, stream), "cublasSetStream");
        try b.check(api.cublasSetWorkspace_v2(b.handle, workspace, workspace_bytes), "cublasSetWorkspace");
        try b.check(api.cublasSetMathMode(b.handle, default_math), "cublasSetMathMode");
        const one: f32 = 1;
        const zero: f32 = 0;
        const ki: c_int = @intCast(k);
        const ni: c_int = @intCast(n);
        // `xt`: x a [k, m] buffer read as [m, k] (strides 1, m), as torch passes such a view: opT on x, ldb = m
        const opx = if (xt) op_t else op_n;
        const ldx: c_int = @intCast(if (xt) m else k);
        try b.check(api.cublasGemmEx(b.handle, op_t, opx, ni, @intCast(m), ki, &one, w, r_16bf, ki, x, r_16bf, ldx, &zero, y, r_16bf, ni, compute_32f, gemm_default_tensor_op), "cublasGemmEx");
    }
};

/// torch's ``_getAlignment`` (the cuBLASLt preference's MIN_ALIGNMENT_*): the largest power of two up to 256 dividing
/// the address.
pub fn alignment(p: u64) u32 {
    return @intCast(@min(@as(u64, 256), p & (~p +% 1)));
}

test alignment {
    try std.testing.expectEqual(@as(u32, 256), alignment(0x7f0000001000));
    try std.testing.expectEqual(@as(u32, 128), alignment(0x7f0000001080));
    try std.testing.expectEqual(@as(u32, 2), alignment(0x7f0000001082));
    try std.testing.expectEqual(@as(u32, 1), alignment(0x7f0000001081));
}
