// DeepSeek-V4.1's vision tower glue (vision.py ``Tower``), each kernel the torch op sequence the Python engine runs,
// in torch's order and roundings (separate fp32 ops as _rn intrinsics, which never contract; libdevice's expf / erff /
// powf / cosf / sinf under nvcc's defaults as torch builds them: no fast math, fmad on), so the tower's bits are torch's:
//   rms        _rms: x.float(); sum(x^2) as at::native's inner-dim reduce (32 lanes, 4 accumulators of float4
//              loads, combined 0+1+2+3, shuffle-down 16..1), * 1/D, + eps, rsqrtf, x * r, w.float() * x, -> bf16
//   rope       _rope + _rotate on q and k of the qkv rows, v copied: [n, 3 D] -> q, k, v as [heads, n, 64]
//   attn       F.scaled_dot_product_attention's FLASH backend (FA2 fwd, hdim 64: 128 x 128 tiles, 4 warps, key
//              blocks last first, exp2f(s * scale - max * scale) unfused (flash built with UNFUSE_FMA), P in bf16,
//              the row sums quad-reduced at the end)
//   silu_mul   F.silu(g) * u on the w1 rows' halves; gelu  F.gelu (erf); add  x + y (bf16, fp32 math)
//   span       the image's span rows: image_start / image_newline / image_end and the aligner rows in prompt order
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv41_vision {

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 rn(const float v) { return __float2bfloat16_rn(v); }

// rms: one warp a row (blockDim 32 x 16, the reduce config of a [n, 1024] float mean with n >= 16)
extern "C" __global__ void __launch_bounds__(512) dsv41_vision_rms(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                  __nv_bfloat16* __restrict__ out, int n, int D, float eps) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;
    const int lane = threadIdx.x;
    const __nv_bfloat16* xr = x + (size_t)row * D;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int idx = lane; idx * 4 + 3 < D; idx += 32) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float v = bf(xr[idx * 4 + i]);
            acc[i] = __fadd_rn(acc[i], __fmul_rn(v, v));
        }
    }
    float s = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
    s = __shfl_sync(0xffffffffu, s, 0);
    const float r = rsqrtf(__fadd_rn(__fmul_rn(s, 1.0f / (float)D), eps));
    for (int j = lane; j < D; j += 32) out[(size_t)row * D + j] = rn(__fmul_rn(bf(w[j]), __fmul_rn(bf(xr[j]), r)));
}

// rope: inv[i] = 1 / 10000^(2i/32) (torch: theta ** (arange / dim), reciprocal); f = (h | w) * inv; cos / sin fp32
extern "C" __global__ void dsv41_vision_rope(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                            __nv_bfloat16* __restrict__ v, int n, int nw, int heads, float theta) {
    const int p = blockIdx.x;          // patch
    const int t = threadIdx.x;         // heads * 32 threads: (head, j) for j < 32
    const int h = t / 32, j = t % 32;
    if (p >= n || h >= heads) return;
    const int D = heads * 64;
    const int half = j < 16 ? j : j - 16;
    const float e = __fdiv_rn((float)(2 * half), 32.0f);
    const float inv = __fdiv_rn(1.0f, powf(theta, e));
    const float pos = (float)(j < 16 ? p / nw : p % nw);
    const float f = __fmul_rn(pos, inv);
    const float c = cosf(f), s = sinf(f);
    const __nv_bfloat16* row = qkv + (size_t)p * 3 * D;
    const size_t o = ((size_t)h * n + p) * 64;
#pragma unroll
    for (int which = 0; which < 2; which++) {
        const __nv_bfloat16* src = row + which * D + h * 64;
        const float a = bf(src[j]), b = bf(src[j + 32]);
        __nv_bfloat16* dst = which == 0 ? q : k;
        dst[o + j] = rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
        dst[o + j + 32] = rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
    v[o + j] = row[2 * D + h * 64 + j];
    v[o + j + 32] = row[2 * D + h * 64 + j + 32];
}

// ---- attention: FA2's forward (hdim 64, kBlockM = kBlockN = 128, 4 warps, SM80 m16n8k16 bf16 MMA) ----------------
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

constexpr int BM = 128, BN = 128, HD = 64;

// one CTA: 128 query rows of one head; warp w owns rows w*16 + {0, 64} (two m16 tiles), every key column of a block
extern "C" __global__ void __launch_bounds__(128) dsv41_vision_attn(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                                   const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                                                   int n, int heads, float scale_log2) {
    __shared__ __nv_bfloat16 sk[BN][HD + 8];
    __shared__ __nv_bfloat16 sv[BN][HD + 8];
    const int head = blockIdx.y, m0 = blockIdx.x * BM;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, q4 = lane % 4; // the MMA fragment's row group and column pair
    const __nv_bfloat16* Qh = Q + (size_t)head * n * HD;
    const __nv_bfloat16* Kh = K + (size_t)head * n * HD;
    const __nv_bfloat16* Vh = V + (size_t)head * n * HD;
    // Q's A fragments stay in registers for the whole key loop (rows past n read as 0)
    uint32_t qa[2][HD / 16][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = m0 + warp * 16 + mi * 64 + g + (e & 1) * 8;
                const int c = kk * 16 + q4 * 2 + (e >> 1) * 8;
                const __nv_bfloat16 z = __float2bfloat16_rn(0.f);
                qa[mi][kk][e] = r < n ? pack2(Qh[(size_t)r * HD + c], Qh[(size_t)r * HD + c + 1]) : pack2(z, z);
            }
    float acc_o[2][8][4];   // [m tile][hd / 8][4]
    float row_max[2][2], row_sum[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
#pragma unroll
        for (int d = 0; d < 8; d++) acc_o[mi][d][0] = acc_o[mi][d][1] = acc_o[mi][d][2] = acc_o[mi][d][3] = 0.f;
        row_max[mi][0] = row_max[mi][1] = -INFINITY;
        row_sum[mi][0] = row_sum[mi][1] = 0.f;
    }
    const int nblocks = (n + BN - 1) / BN;
    for (int nb = nblocks - 1; nb >= 0; nb--) {
        const bool first = nb == nblocks - 1;
        __syncthreads();
        for (int i = tid; i < BN * HD; i += 128) {
            const int r = i / HD, c = i % HD, key = nb * BN + r;
            sk[r][c] = key < n ? Kh[(size_t)key * HD + c] : __float2bfloat16_rn(0.f);
            sv[r][c] = key < n ? Vh[(size_t)key * HD + c] : __float2bfloat16_rn(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            float s[16][4];
#pragma unroll
            for (int ni = 0; ni < 16; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk++) {
                const uint32_t* a = qa[mi][kk];
#pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    uint32_t b[2];
                    b[0] = pack2(sk[ni * 8 + g][kk * 16 + q4 * 2], sk[ni * 8 + g][kk * 16 + q4 * 2 + 1]);
                    b[1] = pack2(sk[ni * 8 + g][kk * 16 + q4 * 2 + 8], sk[ni * 8 + g][kk * 16 + q4 * 2 + 9]);
                    mma16816(s[ni], a, b);
                }
            }
            // the mask past the last key (FA2 apply_mask: -inf)
#pragma unroll
            for (int ni = 0; ni < 16; ni++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    if (nb * BN + ni * 8 + q4 * 2 + (e & 1) >= n) s[ni][e] = -INFINITY;
            // softmax_rescale_o: rows (g) -> e 0,1 ; (g + 8) -> e 2,3 ; columns in (ni, pair) order
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                float mx = first ? s[0][hr * 2] : row_max[mi][hr];
#pragma unroll
                for (int ni = 0; ni < 16; ni++)
#pragma unroll
                    for (int e = 0; e < 2; e++)
                        if (!(first && ni == 0 && e == 0)) mx = fmaxf(mx, s[ni][hr * 2 + e]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                float sc = 1.f;
                if (!first) {
                    const float prev = row_max[mi][hr];
                    const float cur = mx == -INFINITY ? 0.f : mx;
                    sc = exp2f(__fmul_rn(__fsub_rn(prev, cur), scale_log2));
#pragma unroll
                    for (int d = 0; d < 8; d++) {
                        acc_o[mi][d][hr * 2] = __fmul_rn(acc_o[mi][d][hr * 2], sc);
                        acc_o[mi][d][hr * 2 + 1] = __fmul_rn(acc_o[mi][d][hr * 2 + 1], sc);
                    }
                }
                row_max[mi][hr] = mx;
                const float ms = mx == -INFINITY ? 0.f : __fmul_rn(mx, scale_log2);
                // FA2's row_sum *= scale and the first add are one FFMA (UNFUSE_FMA guards only exp)
                float sum = 0.f;
#pragma unroll
                for (int ni = 0; ni < 16; ni++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const float p = exp2f(__fsub_rn(__fmul_rn(s[ni][hr * 2 + e], scale_log2), ms));
                        s[ni][hr * 2 + e] = p;
                        if (ni == 0 && e == 0) sum = first ? p : __fmaf_rn(row_sum[mi][hr], sc, p);
                        else sum = __fadd_rn(sum, p);
                    }
                row_sum[mi][hr] = sum;
            }
            // O += P V: P (bf16) as the A operand, 8 k16 steps over the block's keys
#pragma unroll
            for (int kk = 0; kk < BN / 16; kk++) {
                uint32_t a[4];
                a[0] = pack2(rn(s[2 * kk][0]), rn(s[2 * kk][1]));
                a[1] = pack2(rn(s[2 * kk][2]), rn(s[2 * kk][3]));
                a[2] = pack2(rn(s[2 * kk + 1][0]), rn(s[2 * kk + 1][1]));
                a[3] = pack2(rn(s[2 * kk + 1][2]), rn(s[2 * kk + 1][3]));
#pragma unroll
                for (int d = 0; d < 8; d++) {
                    uint32_t b[2];
                    b[0] = pack2(sv[kk * 16 + q4 * 2][d * 8 + g], sv[kk * 16 + q4 * 2 + 1][d * 8 + g]);
                    b[1] = pack2(sv[kk * 16 + q4 * 2 + 8][d * 8 + g], sv[kk * 16 + q4 * 2 + 9][d * 8 + g]);
                    mma16816(acc_o[mi][d], a, b);
                }
            }
        }
    }
    // normalize_softmax_lse: the row sums quad-allreduced, 1 / sum (1 for 0 / NaN), O * inv, bf16
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            float sum = row_sum[mi][hr];
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
            const float inv = (sum == 0.f || sum != sum) ? 1.f : __fdiv_rn(1.f, sum);
            const int row = m0 + warp * 16 + mi * 64 + g + hr * 8;
            if (row >= n) continue;
#pragma unroll
            for (int d = 0; d < 8; d++) {
                const int col = d * 8 + q4 * 2;
                __nv_bfloat16* o = O + (size_t)row * heads * HD + head * HD + col;
                o[0] = rn(__fmul_rn(acc_o[mi][d][hr * 2], inv));
                o[1] = rn(__fmul_rn(acc_o[mi][d][hr * 2 + 1], inv));
            }
        }
}

// silu_mul: h [n, 2 I] -> out [n, I] = bf16(bf16(silu(g)) * u)
extern "C" __global__ void dsv41_vision_silu_mul(const __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ out, long long n, int I) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * I) return;
    const long long r = i / I, c = i % I;
    const float g = bf(h[r * 2 * I + c]);
    const float sg = bf(rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g)))));
    out[i] = rn(__fmul_rn(sg, bf(h[r * 2 * I + I + c])));
}

// gelu (approximate='none'): x * 0.5 * (1 + erf(x * M_SQRT1_2)), in place
extern "C" __global__ void dsv41_vision_gelu(__nv_bfloat16* __restrict__ x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = bf(x[i]);
    x[i] = rn(__fmul_rn(__fmul_rn(v, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(v, 0.70710678118654752440f)))));
}

// add: x += y (bf16, fp32 sum)
extern "C" __global__ void dsv41_vision_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = rn(__fadd_rn(bf(x[i]), bf(y[i])));
}

// unfold (the aligner's pad + unfold): x [h, w, C] -> F.unfold's [C r r, L], row c r r + kh r + kw, zero past h / w
extern "C" __global__ void dsv41_vision_unfold(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int h, int w, int C, int r) {
    const int L = blockIdx.x;
    const size_t rows = gridDim.x;
    const int lw = (w + r - 1) / r;
    const int by = L / lw, bx = L % lw;
    for (int col = threadIdx.x; col < C * r * r; col += blockDim.x) {
        const int c = col / (r * r), kh = (col / r) % r, kw = col % r;
        const int y = by * r + kh, xx = bx * r + kw;
        out[(size_t)col * rows + L] = (y < h && xx < w) ? x[((size_t)y * w + xx) * C + c] : __float2bfloat16_rn(0.f);
    }
}

// span: out [lh (lw + 1) + 2, D]: start, (lw aligner rows, newline) x lh, end
extern "C" __global__ void dsv41_vision_span(const __nv_bfloat16* __restrict__ feats, const __nv_bfloat16* __restrict__ start,
                            const __nv_bfloat16* __restrict__ newline, const __nv_bfloat16* __restrict__ end,
                            __nv_bfloat16* __restrict__ out, int lh, int lw, int D) {
    const int i = blockIdx.x;
    const int total = lh * (lw + 1) + 2;
    const __nv_bfloat16* src;
    if (i == 0) src = start;
    else if (i == total - 1) src = end;
    else {
        const int r = (i - 1) / (lw + 1), c = (i - 1) % (lw + 1);
        src = c == lw ? newline : feats + ((size_t)r * lw + c) * D;
    }
    for (int j = threadIdx.x; j < D; j += blockDim.x) out[(size_t)i * D + j] = src[j];
}

} // namespace dsv41_vision
