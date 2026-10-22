//! Lossless WebP's inverse transforms, ported from libwebp 1.6.0 ``src/dsp/lossless.c`` (C versions, same arithmetic).
const std = @import("std");

pub const Kind = enum(u2) { predictor = 0, cross_color = 1, subtract_green = 2, color_indexing = 3 };

/// ``VP8LTransform``: ``xsize`` is the width the transform's output has; ``data`` its sub-image or colour map.
pub const Transform = struct {
    kind: Kind = .subtract_green,
    bits: u5 = 0,
    xsize: u32 = 0,
    ysize: u32 = 0,
    data: []u32 = &.{},
};

pub const argb_black: u32 = 0xff000000;

pub fn subSample(size: u32, bits: u5) u32 {
    return (size + (@as(u32, 1) << bits) - 1) >> bits;
}

/// ``VP8LAddPixels``: per-channel sum modulo 256.
pub fn addPixels(a: u32, b: u32) u32 {
    const ag = (a & 0xff00ff00) +% (b & 0xff00ff00);
    const rb = (a & 0x00ff00ff) +% (b & 0x00ff00ff);
    return (ag & 0xff00ff00) | (rb & 0x00ff00ff);
}

fn average2(a0: u32, a1: u32) u32 {
    return (((a0 ^ a1) & 0xfefefefe) >> 1) + (a0 & a1);
}

fn average3(a0: u32, a1: u32, a2: u32) u32 {
    return average2(average2(a0, a2), a1);
}

fn average4(a0: u32, a1: u32, a2: u32, a3: u32) u32 {
    return average2(average2(a0, a1), average2(a2, a3));
}

fn clip255(a: u32) u32 {
    if (a < 256) return a;
    return ~a >> 24;
}

fn ch(c: u32, shift: u5) u32 {
    return (c >> shift) & 0xff;
}

fn clampedAddSubtractFull(c0: u32, c1: u32, c2: u32) u32 {
    var out: u32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |s| out |= clip255(ch(c0, s) +% ch(c1, s) -% ch(c2, s)) << s;
    return out;
}

fn clampedAddSubtractHalf(c0: u32, c1: u32, c2: u32) u32 {
    const ave = average2(c0, c1);
    var out: u32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |s| {
        const a: i32 = @intCast(ch(ave, s));
        const b: i32 = @intCast(ch(c2, s));
        out |= clip255(@bitCast(a + @divTrunc(a - b, 2))) << s;
    }
    return out;
}

fn sub3(a: u32, b: u32, c: u32) i32 {
    const pb = @as(i32, @intCast(b)) - @as(i32, @intCast(c));
    const pa = @as(i32, @intCast(a)) - @as(i32, @intCast(c));
    return @as(i32, @intCast(@abs(pb))) - @as(i32, @intCast(@abs(pa)));
}

fn select(a: u32, b: u32, c: u32) u32 {
    var pa_minus_pb: i32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |s| pa_minus_pb += sub3(ch(a, s), ch(b, s), ch(c, s));
    return if (pa_minus_pb <= 0) a else b;
}

/// ``VP8LPredictor{mode}_C`` for modes 2..13 (left = out[x - 1], top = the row above at x); 14 and 15 are black.
fn predict(mode: u32, left: u32, tl: u32, t: u32, tr: u32) u32 {
    return switch (mode) {
        2 => t,
        3 => tr,
        4 => tl,
        5 => average3(left, t, tr),
        6 => average2(left, tl),
        7 => average2(left, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average4(left, tl, t, tr),
        11 => select(t, left, tl),
        12 => clampedAddSubtractFull(left, t, tl),
        13 => clampedAddSubtractHalf(left, t, tl),
        else => argb_black,
    };
}

/// ``PredictorInverseTransform_C`` over the whole image, in place (``px`` is ``xsize * ysize``).
pub fn predictorInverse(t: *const Transform, px: []u32) void {
    const w: usize = t.xsize;
    if (w == 0 or t.ysize == 0) return;
    px[0] = addPixels(px[0], argb_black);
    for (1..w) |x| px[x] = addPixels(px[x], px[x - 1]);
    const tile_w = @as(usize, 1) << t.bits;
    const tiles_per_row: usize = subSample(t.xsize, t.bits);
    for (1..t.ysize) |y| {
        const row = y * w;
        const up = row - w;
        const modes = t.data[(y >> t.bits) * tiles_per_row ..][0..tiles_per_row];
        px[row] = addPixels(px[row], px[up]);
        var x: usize = 1;
        while (x < w) {
            const mode = (modes[x >> t.bits] >> 8) & 0xf;
            const x_end = @min((x & ~(tile_w - 1)) + tile_w, w);
            while (x < x_end) : (x += 1) {
                const i = row + x;
                if (mode == 1) {
                    px[i] = addPixels(px[i], px[i - 1]);
                    continue;
                }
                // the row above, read as one buffer: top-right of the last pixel is this row's first
                const pred = predict(mode, px[i - 1], px[up + x - 1], px[up + x], px[up + x + 1]);
                px[i] = addPixels(px[i], pred);
            }
        }
    }
}

/// ``VP8LAddGreenToBlueAndRed_C``, in place.
pub fn addGreen(px: []u32) void {
    for (px) |*p| {
        const argb = p.*;
        const green = (argb >> 8) & 0xff;
        var red_blue = argb & 0x00ff00ff;
        red_blue +%= (green << 16) | green;
        red_blue &= 0x00ff00ff;
        p.* = (argb & 0xff00ff00) | red_blue;
    }
}

fn delta(pred: u8, color: u8) i32 {
    return (@as(i32, @as(i8, @bitCast(pred))) * @as(i8, @bitCast(color))) >> 5;
}

/// ``ColorSpaceInverseTransform_C`` (``VP8LTransformColorInverse_C`` per tile), in place.
pub fn crossColorInverse(t: *const Transform, px: []u32) void {
    const w: usize = t.xsize;
    const tiles_per_row: usize = subSample(t.xsize, t.bits);
    for (0..t.ysize) |y| {
        const codes = t.data[(y >> t.bits) * tiles_per_row ..][0..tiles_per_row];
        for (px[y * w ..][0..w], 0..) |*p, x| {
            const code = codes[x >> t.bits];
            const g2r: u8 = @truncate(code);
            const g2b: u8 = @truncate(code >> 8);
            const r2b: u8 = @truncate(code >> 16);
            const argb = p.*;
            const green: u8 = @truncate(argb >> 8);
            var new_red: i32 = @intCast((argb >> 16) & 0xff);
            var new_blue: i32 = @intCast(argb & 0xff);
            new_red += delta(g2r, green);
            new_red &= 0xff;
            new_blue += delta(g2b, green);
            new_blue += delta(r2b, @intCast(new_red));
            new_blue &= 0xff;
            p.* = (argb & 0xff00ff00) | (@as(u32, @intCast(new_red)) << 16) | @as(u32, @intCast(new_blue));
        }
    }
}

/// ``ColorIndexInverseTransform_C``: ``src`` rows of ``subSample(xsize, bits)`` bundled indices (in green) to ``dst``.
pub fn colorIndexInverse(t: *const Transform, src: []const u32, dst: []u32) void {
    const w: usize = t.xsize;
    const map = t.data;
    if (t.bits == 0) {
        for (src[0 .. w * t.ysize], dst[0 .. w * t.ysize]) |s, *d| d.* = map[(s >> 8) & 0xff];
        return;
    }
    const bpp: u5 = @as(u5, 8) >> @intCast(t.bits);
    const count_mask = (@as(usize, 1) << t.bits) - 1;
    const bit_mask = (@as(u32, 1) << bpp) - 1;
    var si: usize = 0;
    var di: usize = 0;
    for (0..t.ysize) |_| {
        var packed_px: u32 = 0;
        for (0..w) |x| {
            if (x & count_mask == 0) {
                packed_px = (src[si] >> 8) & 0xff;
                si += 1;
            }
            dst[di] = map[packed_px & bit_mask];
            di += 1;
            packed_px >>= bpp;
        }
    }
}

/// ``VP8LColorIndexInverseTransformAlpha``: byte indices to the colour map's green, bundled as above.
pub fn colorIndexAlpha(t: *const Transform, src: []const u8, dst: []u8) void {
    const w: usize = t.xsize;
    const map = t.data;
    if (t.bits == 0) {
        for (src[0 .. w * t.ysize], dst[0 .. w * t.ysize]) |s, *d| d.* = @truncate(map[s] >> 8);
        return;
    }
    const bpp: u5 = @as(u5, 8) >> @intCast(t.bits);
    const count_mask = (@as(usize, 1) << t.bits) - 1;
    const bit_mask = (@as(u32, 1) << bpp) - 1;
    var si: usize = 0;
    var di: usize = 0;
    for (0..t.ysize) |_| {
        var packed_px: u32 = 0;
        for (0..w) |x| {
            if (x & count_mask == 0) {
                packed_px = src[si];
                si += 1;
            }
            dst[di] = @truncate(map[packed_px & bit_mask] >> 8);
            di += 1;
            packed_px >>= bpp;
        }
    }
}

test "predictors 11-13 and the averages" {
    try std.testing.expectEqual(@as(u32, 0x7f7f7f7f), average2(0xffffffff, 0x00000000));
    try std.testing.expectEqual(@as(u32, 0xff00ff00), clampedAddSubtractFull(0xff00ff00, 0xffff0000, 0x00ff0000));
    // (a + (a - b) / 2) truncates toward zero, clamps at 0 and 255
    try std.testing.expectEqual(@as(u32, 0x00ff0002), clampedAddSubtractHalf(0x00c00002, 0x00c00002, 0x00010003));
    try std.testing.expectEqual(@as(u32, 0x05060708), select(0x01020304, 0x05060708, 0x01020304));
    try std.testing.expectEqual(@as(u32, 0x01020304), select(0x01020304, 0x05060708, 0x05060708));
}

test "cross-colour multipliers are signed" {
    var px = [_]u32{0xff10f020};
    const codes = [_]u32{0x00000080}; // green_to_red -128: red += (-128 * -16) >> 5 = 64
    const t: Transform = .{ .kind = .cross_color, .bits = 2, .xsize = 1, .ysize = 1, .data = @constCast(&codes) };
    crossColorInverse(&t, &px);
    try std.testing.expectEqual(@as(u32, 0xff50f020), px[0]);
}

test "bundled indices: 2 colours, 8 a byte" {
    var map = [_]u32{ 0xff000000, 0xffffffff };
    const t: Transform = .{ .kind = .color_indexing, .bits = 3, .xsize = 9, .ysize = 1, .data = &map };
    const src = [_]u32{ 0x0000a500, 0x00000100 };
    var dst: [9]u32 = undefined;
    colorIndexInverse(&t, &src, &dst);
    const want = [_]u32{ 1, 0, 1, 0, 0, 1, 0, 1, 1 };
    for (dst, want) |d, b| try std.testing.expectEqual(map[b], d);
}
