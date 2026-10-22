//! VP8 reconstruction as libwebp 1.6.0 runs it (``dsp/dec.c``, the IDCT as ``dsp/dec_sse2.c``).
const std = @import("std");

/// The work buffer's stride (``BPS``).
pub const bps = 32;

fn clip8(v: i32) u8 {
    return if (v & ~@as(i32, 0xff) == 0) @intCast(v) else if (v < 0) 0 else 255;
}

fn clip1(v: i32) u8 { // ``VP8kclip1``
    return @intCast(std.math.clamp(v, 0, 255));
}

fn sclip1(v: i32) i32 { // ``VP8ksclip1``
    return std.math.clamp(v, -128, 127);
}

fn sclip2(v: i32) i32 { // ``VP8ksclip2``
    return std.math.clamp(v, -16, 15);
}

fn abs0(v: i32) i32 { // ``VP8kabs0``
    return @intCast(@abs(v));
}

fn mul1(a: i32) i32 { // ``WEBP_TRANSFORM_AC3_MUL1``
    return ((a *% 20091) >> 16) +% a;
}

fn mul2(a: i32) i32 { // ``WEBP_TRANSFORM_AC3_MUL2``
    return (a *% 35468) >> 16;
}

fn store(b: []u8, at: usize, v: i32) void {
    b[at] = clip8(@as(i32, b[at]) +% (v >> 3));
}

fn mulHi(a: i16, k: i32) i16 { // ``_mm_mulhi_epi16``
    return @truncate((@as(i32, a) * k) >> 16);
}

/// One pass of ``Transform_SSE2`` on a column: ``x * K >> 16`` as ``mulhi(x, K - 65536) + x``, 16-bit wrapping sums.
fn pass(x0: i16, x1: i16, x2: i16, x3: i16) [4]i16 {
    const a = x0 +% x2;
    const b = x0 -% x2;
    const c = (x1 -% x3) +% (mulHi(x1, -30068) -% mulHi(x3, 20091));
    const d = (x1 +% x3) +% (mulHi(x1, 20091) +% mulHi(x3, -30068));
    return .{ a +% d, b +% c, b -% c, a -% d };
}

/// ``VP8Transform``: the 4x4 inverse DCT of ``in`` added to the block at ``d``, in ``Transform_SSE2``'s 16-bit lanes.
pub fn transform(in: []const i16, b: []u8, d: usize) void {
    var t: [4][4]i16 = undefined; // t[i] = column i after the vertical pass, then read across for the horizontal one
    for (0..4) |i| t[i] = pass(in[i], in[4 + i], in[8 + i], in[12 + i]);
    for (0..4) |i| {
        const r = pass(t[0][i] +% 4, t[1][i], t[2][i], t[3][i]);
        for (0..4) |x| {
            const v = @as(i16, b[d + i * bps + x]) +% (r[x] >> 3);
            b[d + i * bps + x] = @intCast(std.math.clamp(v, 0, 255));
        }
    }
}

/// ``TransformAC3_C``: the inverse DCT when only ``in[0]``, ``in[1]`` and ``in[4]`` are non-zero.
pub fn transformAc3(in: []const i16, b: []u8, d: usize) void {
    const a = @as(i32, in[0]) + 4;
    const c4 = mul2(in[4]);
    const d4 = mul1(in[4]);
    const c1 = mul2(in[1]);
    const d1 = mul1(in[1]);
    const dcs = [4]i32{ a + d4, a + c4, a - c4, a - d4 };
    for (dcs, 0..) |dc, y| {
        const row = d + y * bps;
        store(b, row + 0, dc + d1);
        store(b, row + 1, dc + c1);
        store(b, row + 2, dc - c1);
        store(b, row + 3, dc - d1);
    }
}

/// ``TransformDC_C``: the inverse DCT of a DC-only block.
pub fn transformDc(in: []const i16, b: []u8, d: usize) void {
    const dc = @as(i32, in[0]) + 4;
    for (0..4) |y| {
        for (0..4) |x| store(b, d + x + y * bps, dc);
    }
}

/// ``TransformUV_C``: the four 4x4 blocks of an 8x8 chroma block.
pub fn transformUv(in: []const i16, b: []u8, d: usize) void {
    transform(in[0..], b, d);
    transform(in[16..], b, d + 4);
    transform(in[32..], b, d + 4 * bps);
    transform(in[48..], b, d + 4 * bps + 4);
}

/// ``TransformDCUV_C``: the DC-only transform of each chroma block with a non-zero DC.
pub fn transformDcUv(in: []const i16, b: []u8, d: usize) void {
    const offs = [4]usize{ 0, 4, 4 * bps, 4 * bps + 4 };
    for (offs, 0..) |o, k| {
        if (in[k * 16] != 0) transformDc(in[k * 16 ..], b, d + o);
    }
}

/// ``TransformWHT_C``: the inverse Walsh-Hadamard transform of the 16 luma DCs into ``out[16 k]``.
pub fn transformWht(in: *const [16]i16, out: []i16) void {
    var tmp: [16]i32 = undefined;
    for (0..4) |i| {
        const a0 = @as(i32, in[0 + i]) + in[12 + i];
        const a1 = @as(i32, in[4 + i]) + in[8 + i];
        const a2 = @as(i32, in[4 + i]) - in[8 + i];
        const a3 = @as(i32, in[0 + i]) - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for (0..4) |i| {
        const dc = tmp[0 + i * 4] + 3;
        const a0 = dc + tmp[3 + i * 4];
        const a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        const a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        const a3 = dc - tmp[3 + i * 4];
        out[64 * i + 0] = @truncate((a0 + a1) >> 3);
        out[64 * i + 16] = @truncate((a3 + a2) >> 3);
        out[64 * i + 32] = @truncate((a0 - a1) >> 3);
        out[64 * i + 48] = @truncate((a3 - a2) >> 3);
    }
}

fn trueMotion(b: []u8, d0: usize, size: usize) void {
    const top = d0 - bps;
    const tl: i32 = b[top - 1];
    var d = d0;
    for (0..size) |_| {
        const left: i32 = b[d - 1];
        for (0..size) |x| b[d + x] = clip1(left + b[top + x] - tl);
        d += bps;
    }
}

fn fill(b: []u8, d: usize, size: usize, v: u8) void {
    for (0..size) |j| @memset(b[d + j * bps ..][0..size], v);
}

/// ``VP8PredLuma16``: modes DC, TM, VE, HE, DC without top, DC without left, DC without either.
pub fn predLuma16(mode: u8, b: []u8, d: usize) void {
    switch (mode) {
        0 => {
            var dc: u32 = 16;
            for (0..16) |j| dc += @as(u32, b[d - 1 + j * bps]) + b[d + j - bps];
            fill(b, d, 16, @intCast(dc >> 5));
        },
        1 => trueMotion(b, d, 16),
        2 => for (0..16) |j| @memcpy(b[d + j * bps ..][0..16], b[d - bps ..][0..16]),
        3 => for (0..16) |j| @memset(b[d + j * bps ..][0..16], b[d + j * bps - 1]),
        4 => {
            var dc: u32 = 8;
            for (0..16) |j| dc += b[d - 1 + j * bps];
            fill(b, d, 16, @intCast(dc >> 4));
        },
        5 => {
            var dc: u32 = 8;
            for (0..16) |i| dc += b[d + i - bps];
            fill(b, d, 16, @intCast(dc >> 4));
        },
        else => fill(b, d, 16, 0x80),
    }
}

/// ``VP8PredChroma8``: the same seven modes on an 8x8 chroma block.
pub fn predChroma8(mode: u8, b: []u8, d: usize) void {
    switch (mode) {
        0 => {
            var dc: u32 = 8;
            for (0..8) |i| dc += @as(u32, b[d + i - bps]) + b[d - 1 + i * bps];
            fill(b, d, 8, @intCast(dc >> 4));
        },
        1 => trueMotion(b, d, 8),
        2 => for (0..8) |j| @memcpy(b[d + j * bps ..][0..8], b[d - bps ..][0..8]),
        3 => for (0..8) |j| @memset(b[d + j * bps ..][0..8], b[d + j * bps - 1]),
        4 => {
            var dc: u32 = 4;
            for (0..8) |i| dc += b[d - 1 + i * bps];
            fill(b, d, 8, @intCast(dc >> 3));
        },
        5 => {
            var dc: u32 = 4;
            for (0..8) |i| dc += b[d + i - bps];
            fill(b, d, 8, @intCast(dc >> 3));
        },
        else => fill(b, d, 8, 0x80),
    }
}

fn avg3(a: u32, b: u32, c: u32) u8 {
    return @intCast((a + 2 * b + c + 2) >> 2);
}

fn avg2(a: u32, b: u32) u8 {
    return @intCast((a + b + 1) >> 1);
}

/// ``VP8PredLuma4``: the ten 4x4 modes (DC, TM, VE, HE, RD, VR, LD, VL, HD, HU) at ``d``.
pub fn predLuma4(mode: u8, b: []u8, d: usize) void {
    const t = d - bps; // top row; t - 1 is the top-left sample
    const i_: u32 = b[d - 1];
    const j_: u32 = b[d - 1 + bps];
    const k_: u32 = b[d - 1 + 2 * bps];
    const l_: u32 = b[d - 1 + 3 * bps];
    const x_: u32 = b[t - 1];
    var top: [8]u32 = undefined;
    for (&top, 0..) |*v, n| v.* = b[t + n];
    const a, const b_, const c, const d_, const e, const f, const g, const h = top;
    var o: [4][4]u8 = undefined; // o[y][x]
    switch (mode) {
        0 => {
            var dc: u32 = 4;
            for (0..4) |n| dc += @as(u32, b[t + n]) + b[d - 1 + n * bps];
            o = @splat(@splat(@intCast(dc >> 3)));
        },
        1 => return trueMotion(b, d, 4),
        2 => {
            const r = [4]u8{ avg3(x_, a, b_), avg3(a, b_, c), avg3(b_, c, d_), avg3(c, d_, e) };
            o = .{ r, r, r, r };
        },
        3 => {
            const r = [4]u8{ avg3(x_, i_, j_), avg3(i_, j_, k_), avg3(j_, k_, l_), avg3(k_, l_, l_) };
            for (0..4) |y| o[y] = @splat(r[y]);
        },
        4 => { // down-right: constant along each diagonal x - y
            const v = [7]u8{ avg3(j_, k_, l_), avg3(i_, j_, k_), avg3(x_, i_, j_), avg3(a, x_, i_), avg3(b_, a, x_), avg3(c, b_, a), avg3(d_, c, b_) };
            for (0..4) |y| for (0..4) |x| {
                o[y][x] = v[3 + x - y];
            };
        },
        5 => o = .{
            .{ avg2(x_, a), avg2(a, b_), avg2(b_, c), avg2(c, d_) },
            .{ avg3(i_, x_, a), avg3(x_, a, b_), avg3(a, b_, c), avg3(b_, c, d_) },
            .{ avg3(j_, i_, x_), avg2(x_, a), avg2(a, b_), avg2(b_, c) },
            .{ avg3(k_, j_, i_), avg3(i_, x_, a), avg3(x_, a, b_), avg3(a, b_, c) },
        },
        6 => { // down-left: constant along each anti-diagonal x + y
            const v = [7]u8{ avg3(a, b_, c), avg3(b_, c, d_), avg3(c, d_, e), avg3(d_, e, f), avg3(e, f, g), avg3(f, g, h), avg3(g, h, h) };
            for (0..4) |y| for (0..4) |x| {
                o[y][x] = v[x + y];
            };
        },
        7 => o = .{
            .{ avg2(a, b_), avg2(b_, c), avg2(c, d_), avg2(d_, e) },
            .{ avg3(a, b_, c), avg3(b_, c, d_), avg3(c, d_, e), avg3(d_, e, f) },
            .{ avg2(b_, c), avg2(c, d_), avg2(d_, e), avg3(e, f, g) },
            .{ avg3(b_, c, d_), avg3(c, d_, e), avg3(d_, e, f), avg3(f, g, h) },
        },
        8 => o = .{
            .{ avg2(i_, x_), avg3(i_, x_, a), avg3(x_, a, b_), avg3(a, b_, c) },
            .{ avg2(j_, i_), avg3(j_, i_, x_), avg2(i_, x_), avg3(i_, x_, a) },
            .{ avg2(k_, j_), avg3(k_, j_, i_), avg2(j_, i_), avg3(j_, i_, x_) },
            .{ avg2(l_, k_), avg3(l_, k_, j_), avg2(k_, j_), avg3(k_, j_, i_) },
        },
        else => {
            const l8: u8 = @intCast(l_);
            o = .{
                .{ avg2(i_, j_), avg3(i_, j_, k_), avg2(j_, k_), avg3(j_, k_, l_) },
                .{ avg2(j_, k_), avg3(j_, k_, l_), avg2(k_, l_), avg3(k_, l_, l_) },
                .{ avg2(k_, l_), avg3(k_, l_, l_), l8, l8 },
                .{ l8, l8, l8, l8 },
            };
        },
    }
    for (0..4) |y| @memcpy(b[d + y * bps ..][0..4], &o[y]);
}

fn doFilter2(b: []u8, p: usize, s: usize) void {
    const p1: i32 = b[p - 2 * s];
    const p0: i32 = b[p - s];
    const q0: i32 = b[p];
    const q1: i32 = b[p + s];
    const a = 3 * (q0 - p0) + sclip1(p1 - q1);
    const a1 = sclip2((a + 4) >> 3);
    const a2 = sclip2((a + 3) >> 3);
    b[p - s] = clip1(p0 + a2);
    b[p] = clip1(q0 - a1);
}

fn doFilter4(b: []u8, p: usize, s: usize) void {
    const p1: i32 = b[p - 2 * s];
    const p0: i32 = b[p - s];
    const q0: i32 = b[p];
    const q1: i32 = b[p + s];
    const a = 3 * (q0 - p0);
    const a1 = sclip2((a + 4) >> 3);
    const a2 = sclip2((a + 3) >> 3);
    const a3 = (a1 + 1) >> 1;
    b[p - 2 * s] = clip1(p1 + a3);
    b[p - s] = clip1(p0 + a2);
    b[p] = clip1(q0 - a1);
    b[p + s] = clip1(q1 - a3);
}

fn doFilter6(b: []u8, p: usize, s: usize) void {
    const p2: i32 = b[p - 3 * s];
    const p1: i32 = b[p - 2 * s];
    const p0: i32 = b[p - s];
    const q0: i32 = b[p];
    const q1: i32 = b[p + s];
    const q2: i32 = b[p + 2 * s];
    const a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    const a1 = (27 * a + 63) >> 7;
    const a2 = (18 * a + 63) >> 7;
    const a3 = (9 * a + 63) >> 7;
    b[p - 3 * s] = clip1(p2 + a3);
    b[p - 2 * s] = clip1(p1 + a2);
    b[p - s] = clip1(p0 + a1);
    b[p] = clip1(q0 - a1);
    b[p + s] = clip1(q1 - a2);
    b[p + 2 * s] = clip1(q2 - a3);
}

fn px(b: []const u8, p: usize, s: usize, k: i32) i32 {
    return b[if (k < 0) p - @as(usize, @intCast(-k)) * s else p + @as(usize, @intCast(k)) * s];
}

fn hev(b: []const u8, p: usize, s: usize, thresh: i32) bool {
    return abs0(px(b, p, s, -2) - px(b, p, s, -1)) > thresh or abs0(px(b, p, s, 1) - px(b, p, s, 0)) > thresh;
}

fn needsFilter(b: []const u8, p: usize, s: usize, t: i32) bool {
    return 4 * abs0(px(b, p, s, -1) - px(b, p, s, 0)) + abs0(px(b, p, s, -2) - px(b, p, s, 1)) <= t;
}

fn needsFilter2(b: []const u8, p: usize, s: usize, t: i32, it: i32) bool {
    var v: [8]i32 = undefined; // p3 p2 p1 p0 q0 q1 q2 q3
    for (&v, 0..) |*x, k| x.* = px(b, p, s, @as(i32, @intCast(k)) - 4);
    if (4 * abs0(v[3] - v[4]) + abs0(v[2] - v[5]) > t) return false;
    return abs0(v[0] - v[1]) <= it and abs0(v[1] - v[2]) <= it and abs0(v[2] - v[3]) <= it and
        abs0(v[7] - v[6]) <= it and abs0(v[6] - v[5]) <= it and abs0(v[5] - v[4]) <= it;
}

/// ``SimpleVFilter16_C`` (``vertical``) / ``SimpleHFilter16_C``: the simple filter across one 16-sample edge.
pub fn simpleFilter16(b: []u8, p: usize, stride: usize, thresh: i32, vertical: bool) void {
    const thresh2 = 2 * thresh + 1;
    for (0..16) |i| {
        const q = if (vertical) p + i else p + i * stride;
        const s = if (vertical) stride else 1;
        if (needsFilter(b, q, s, thresh2)) doFilter2(b, q, s);
    }
}

/// ``SimpleVFilter16i_C`` / ``SimpleHFilter16i_C``: the simple filter across the three inner edges.
pub fn simpleFilter16i(b: []u8, p: usize, stride: usize, thresh: i32, vertical: bool) void {
    for (1..4) |k| simpleFilter16(b, if (vertical) p + 4 * k * stride else p + 4 * k, stride, thresh, vertical);
}

/// ``FilterLoop26_C`` (``six``) / ``FilterLoop24_C``: the complex filter along ``size`` samples.
pub fn filterLoop(b: []u8, p0: usize, hstride: usize, vstride: usize, size: usize, thresh: i32, ithresh: i32, hev_t: i32, six: bool) void {
    const thresh2 = 2 * thresh + 1;
    var p = p0;
    for (0..size) |_| {
        if (needsFilter2(b, p, hstride, thresh2, ithresh)) {
            if (hev(b, p, hstride, hev_t)) {
                doFilter2(b, p, hstride);
            } else if (six) doFilter6(b, p, hstride) else doFilter4(b, p, hstride);
        }
        p += vstride;
    }
}

test "the inverse DCT of a DC-only block adds (dc + 4) >> 3 everywhere" {
    var b: [4 * bps]u8 = @splat(100);
    var in: [16]i16 = @splat(0);
    in[0] = 80;
    var b2 = b;
    transform(&in, &b, 0);
    transformDc(&in, &b2, 0);
    try std.testing.expectEqual(@as(u8, 110), b[0]);
    try std.testing.expectEqualSlices(u8, &b2, &b);
    transformAc3(&in, &b2, 0);
    try std.testing.expectEqual(@as(u8, 120), b2[3 * bps + 3]);
}

test "the WHT of a lone DC spreads (dc + 3) >> 3 to all sixteen blocks" {
    var in: [16]i16 = @splat(0);
    in[0] = 85;
    var out: [256]i16 = @splat(0);
    transformWht(&in, &out);
    for (0..16) |k| try std.testing.expectEqual(@as(i16, 11), out[k * 16]);
}

test "4x4 DC and TM predictions, and the simple filter smoothing a step" {
    var b: [6 * bps]u8 = @splat(0);
    const d = bps + 1;
    for (0..5) |x| b[d - bps - 1 + x] = 40;
    for (0..4) |y| b[d - 1 + y * bps] = 80;
    predLuma4(0, &b, d);
    try std.testing.expectEqual(@as(u8, 60), b[d + 3 * bps + 3]);
    predLuma4(1, &b, d);
    try std.testing.expectEqual(@as(u8, 80), b[d]);
    var e: [16 * 8]u8 = undefined;
    for (0..8) |y| for (0..16) |x| {
        e[y * 16 + x] = if (y < 4) 100 else 110;
    };
    simpleFilter16(&e, 4 * 16, 16, 30, true);
    try std.testing.expectEqualSlices(u8, &.{ 100, 102, 107, 110 }, &.{ e[2 * 16], e[3 * 16], e[4 * 16], e[5 * 16] });
}
