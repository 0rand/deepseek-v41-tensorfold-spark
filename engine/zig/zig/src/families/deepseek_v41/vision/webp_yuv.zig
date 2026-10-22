//! Lossy WebP's YUV 4:2:0 to RGB as libwebp 1.6.0 outputs it (``dsp/upsampling.c``, ``dsp/yuv.h``).
const std = @import("std");
const Yuv = @import("webp_vp8.zig").Yuv;

const yuv_fix2 = 6;
const yuv_mask2 = (256 << yuv_fix2) - 1;

fn multHi(v: i32, coeff: i32) i32 { // ``_mm_mulhi_epu16`` emulation
    return (v * coeff) >> 8;
}

fn clip8(v: i32) u8 { // ``VP8Clip8``
    return if (v & ~@as(i32, yuv_mask2) == 0) @intCast(v >> yuv_fix2) else if (v < 0) 0 else 255;
}

/// ``VP8YuvToRgb``: one pixel's R, G, B (the alpha byte is left as it is).
fn yuvToRgb(y: u8, u: u8, v: u8, dst: []u8) void {
    const yy = multHi(y, 19077);
    dst[0] = clip8(yy + multHi(v, 26149) - 14234);
    dst[1] = clip8(yy - multHi(u, 6419) - multHi(v, 13320) + 8708);
    dst[2] = clip8(yy + multHi(u, 33050) - 17685);
}

fn loadUv(u: u8, v: u8) u32 {
    return @as(u32, u) | @as(u32, v) << 16;
}

fn put(y: u8, uv: u32, dst: []u8) void {
    yuvToRgb(y, @truncate(uv), @truncate(uv >> 16), dst);
}

/// ``UpsampleRgbaLinePair_C``: one or two luma rows between chroma rows ``top`` and ``cur`` (U, V packed in 16 bits).
fn upsamplePair(top_y: []const u8, bottom_y: ?[]const u8, top_u: []const u8, top_v: []const u8, cur_u: []const u8, cur_v: []const u8, top_dst: []u8, bottom_dst: []u8, len: usize) void {
    const last_pixel_pair = (len - 1) >> 1;
    var tl_uv = loadUv(top_u[0], top_v[0]);
    var l_uv = loadUv(cur_u[0], cur_v[0]);
    put(top_y[0], (3 * tl_uv + l_uv + 0x00020002) >> 2, top_dst[0..]);
    if (bottom_y) |by| put(by[0], (3 * l_uv + tl_uv + 0x00020002) >> 2, bottom_dst[0..]);
    for (1..last_pixel_pair + 1) |x| {
        const t_uv = loadUv(top_u[x], top_v[x]);
        const uv = loadUv(cur_u[x], cur_v[x]);
        const avg = tl_uv +% t_uv +% l_uv +% uv +% 0x00080008;
        const diag_12 = (avg +% 2 *% (t_uv +% l_uv)) >> 3;
        const diag_03 = (avg +% 2 *% (tl_uv +% uv)) >> 3;
        put(top_y[2 * x - 1], (diag_12 +% tl_uv) >> 1, top_dst[(2 * x - 1) * 4 ..]);
        put(top_y[2 * x], (diag_03 +% t_uv) >> 1, top_dst[(2 * x) * 4 ..]);
        if (bottom_y) |by| {
            put(by[2 * x - 1], (diag_03 +% l_uv) >> 1, bottom_dst[(2 * x - 1) * 4 ..]);
            put(by[2 * x], (diag_12 +% uv) >> 1, bottom_dst[(2 * x) * 4 ..]);
        }
        tl_uv = t_uv;
        l_uv = uv;
    }
    if (len & 1 == 0) {
        put(top_y[len - 1], (3 * tl_uv + l_uv + 0x00020002) >> 2, top_dst[(len - 1) * 4 ..]);
        if (bottom_y) |by| put(by[len - 1], (3 * l_uv + tl_uv + 0x00020002) >> 2, bottom_dst[(len - 1) * 4 ..]);
    }
}

/// RGBA rows (stride ``w * 4``) from ``yuv`` as ``EmitFancyRGB`` drives the upsampler; the A bytes are untouched.
pub fn toRgba(yuv: *const Yuv, out: []u8) void {
    const w: usize = yuv.w;
    const h: usize = yuv.h;
    const uw: usize = yuv.uv_w;
    std.debug.assert(out.len >= w * h * 4);
    const yrow = struct {
        fn f(p: []const u8, stride: usize, r: usize, n: usize) []const u8 {
            return p[r * stride ..][0..n];
        }
    }.f;
    const orow = out[0 .. w * 4];
    upsamplePair(yrow(yuv.y, w, 0, w), null, yuv.u[0..uw], yuv.v[0..uw], yuv.u[0..uw], yuv.v[0..uw], orow, orow, w);
    var y: usize = 0;
    while (y + 2 < h) : (y += 2) { // rows y + 1 and y + 2, between chroma rows y / 2 and y / 2 + 1
        const k = y / 2;
        upsamplePair(yrow(yuv.y, w, y + 1, w), yrow(yuv.y, w, y + 2, w), yrow(yuv.u, uw, k, uw), yrow(yuv.v, uw, k, uw), yrow(yuv.u, uw, k + 1, uw), yrow(yuv.v, uw, k + 1, uw), out[(y + 1) * w * 4 ..][0 .. w * 4], out[(y + 2) * w * 4 ..][0 .. w * 4], w);
    }
    if (h & 1 == 0) { // the last row of an even height mirrors its chroma row
        const k = h / 2 - 1;
        const last = out[(h - 1) * w * 4 ..][0 .. w * 4];
        upsamplePair(yrow(yuv.y, w, h - 1, w), null, yrow(yuv.u, uw, k, uw), yrow(yuv.v, uw, k, uw), yrow(yuv.u, uw, k, uw), yrow(yuv.v, uw, k, uw), last, last, w);
    }
}

test "grey stays grey, and the fixed-point constants match libwebp's corners" {
    var y = [_]u8{ 128, 128, 128, 128, 128, 128 };
    var u = [_]u8{ 128, 128 };
    var v = [_]u8{ 128, 128 };
    const yuv: Yuv = .{ .w = 3, .h = 2, .y = &y, .u = &u, .v = &v, .uv_w = 2 };
    var out: [24]u8 = @splat(7);
    toRgba(&yuv, &out);
    for (0..6) |i| {
        try std.testing.expectEqualSlices(u8, &.{ 130, 130, 130 }, out[i * 4 ..][0..3]); // VP8YuvToRgb(128, 128, 128)
        try std.testing.expectEqual(@as(u8, 7), out[i * 4 + 3]);
    }
    var px: [3]u8 = undefined;
    yuvToRgb(16, 128, 128, &px);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, &px);
    yuvToRgb(235, 128, 128, &px);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255 }, &px);
}

test "a single column and a single row upsample without reading past the planes" {
    var y = [_]u8{ 10, 200, 90 };
    var u = [_]u8{ 50, 200 };
    var v = [_]u8{ 220, 30 };
    var out: [12]u8 = undefined;
    const col: Yuv = .{ .w = 1, .h = 3, .y = &y, .u = &u, .v = &v, .uv_w = 1 };
    toRgba(&col, &out);
    const row: Yuv = .{ .w = 3, .h = 1, .y = &y, .u = &u, .v = &v, .uv_w = 2 };
    toRgba(&row, &out);
    var px: [3]u8 = undefined;
    yuvToRgb(10, 50, 220, &px);
    try std.testing.expectEqualSlices(u8, &px, out[0..3]);
}
