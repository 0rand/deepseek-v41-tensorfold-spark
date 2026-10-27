//! WebP's first frame as Pillow 12.3 loads it: libwebp 1.6.0's WebPAnimDecoder (MODE_RGBA) over a demuxed file.
const std = @import("std");
const pixels = @import("pixels.zig");
const vp8l = @import("webp_vp8l.zig");
const vp8 = @import("webp_vp8.zig");
const yuv = @import("webp_yuv.zig");
const alpha = @import("webp_alpha.zig");
const demux = @import("webp_demux.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadWebp, TooManyPixels, TooLarge } || Allocator.Error;

pub const max_table_bytes = vp8l.max_table_bytes;

/// Pillow's ``_accept``: RIFF, WEBP, then a VP8, VP8L or VP8X chunk.
pub fn isWebp(raw: []const u8) bool {
    if (raw.len < 16 or !std.mem.eql(u8, raw[0..4], "RIFF") or !std.mem.eql(u8, raw[8..12], "WEBP")) return false;
    const t = raw[12..16];
    return std.mem.eql(u8, t, "VP8 ") or std.mem.eql(u8, t, "VP8L") or std.mem.eql(u8, t, "VP8X");
}

/// The first frame's bitstreams: ``image`` the VP8 or VP8L payload with its pad, ``alph`` the ALPH payload.
const Frame = struct {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    lossless: bool,
    image: []const u8,
    alph: ?[]const u8,
};

/// The demuxed first frame as ``SynthesizeFrame`` hands it to ``WebPDecode``.
fn firstFrame(raw: []const u8, f: demux.Frame) Frame {
    const img = raw[f.image.at..][0..f.image.len];
    var alph: ?[]const u8 = null;
    if (f.alpha.len > 0) {
        const a = raw[f.alpha.at..][0..f.alpha.len];
        const size = std.mem.readInt(u32, a[4..8], .little);
        alph = a[8..][0..@min(size, a.len - 8)];
    }
    const lossless = std.mem.eql(u8, img[0..4], "VP8L");
    return .{ .x = f.x, .y = f.y, .w = f.w, .h = f.h, .lossless = lossless, .image = img[8..], .alph = alph };
}

/// The first frame's RGBA in ``canvas`` (stride ``cw``) at its offset; a full-canvas lossy frame decodes straight in.
fn decodeFrame(gpa: Allocator, f: Frame, canvas: []u8, cw: u32, max_pixels: u64) Error!void {
    const stride = @as(usize, cw) * 4;
    const at = @as(usize, f.y) * stride + @as(usize, f.x) * 4;
    if (f.lossless) {
        var im = try vp8l.decode(gpa, f.image, max_pixels);
        defer im.deinit(gpa);
        if (im.w != f.w or im.h != f.h) return error.BadWebp;
        for (0..f.h) |y| {
            const row = canvas[at + y * stride ..][0 .. @as(usize, f.w) * 4];
            for (0..f.w) |x| {
                const p = im.px[y * f.w + x];
                row[x * 4 ..][0..4].* = .{ @truncate(p >> 16), @truncate(p >> 8), @truncate(p), @truncate(p >> 24) };
            }
        }
        return;
    }
    const n = @as(usize, f.w) * f.h;
    const full = f.x == 0 and f.y == 0 and f.w == cw and n * 4 == canvas.len;
    const rgba = if (full) canvas else try gpa.alloc(u8, n * 4);
    defer if (!full) gpa.free(rgba);
    {
        var planes = try vp8.decode(gpa, f.image, max_pixels);
        defer planes.deinit(gpa);
        if (planes.w != f.w or planes.h != f.h) return error.BadWebp;
        @memset(rgba, 255);
        yuv.toRgba(&planes, rgba);
    }
    if (f.alph) |a| {
        const plane = try gpa.alloc(u8, n);
        defer gpa.free(plane);
        try alpha.decode(gpa, a, f.w, f.h, plane);
        for (plane, 0..) |v, i| rgba[i * 4 + 3] = v;
    }
    if (full) return;
    const rs = @as(usize, f.w) * 4;
    for (0..f.h) |y| @memcpy(canvas[at + y * stride ..][0..rs], rgba[y * rs ..][0..rs]);
}

/// Pillow's mode is RGBA when ``WebPGetFeatures`` reports alpha, which ``WebPAnimDecoderNew`` also requires to pass.
pub fn decode(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    if (!isWebp(raw)) return error.BadWebp;
    const feat = demux.features(raw) orelse return error.BadWebp;
    const d = try demux.demux(raw);
    const f = firstFrame(raw, d.first);
    const n = @as(u64, d.canvas_w) * d.canvas_h;
    if (n > max_pixels or @as(u64, f.w) * f.h > max_pixels) return error.TooManyPixels;
    const np: usize = @intCast(n);
    const canvas = try gpa.alloc(u8, np * 4);
    errdefer gpa.free(canvas);
    @memset(canvas, 0);
    try decodeFrame(gpa, f, canvas, d.canvas_w, max_pixels);
    if (feat.has_alpha) return .{ .w = d.canvas_w, .h = d.canvas_h, .mode = .rgba, .data = canvas };
    // RGB in place: pixel i's three bytes move down to i * 3, which never passes pixel i's own bytes at i * 4
    for (0..np) |i| std.mem.copyForwards(u8, canvas[i * 3 ..][0..3], canvas[i * 4 ..][0..3]);
    return .{ .w = d.canvas_w, .h = d.canvas_h, .mode = .rgb, .data = try gpa.realloc(canvas, np * 3) };
}

test {
    _ = alpha;
    _ = vp8l;
    _ = vp8;
    _ = yuv;
    _ = demux;
}

test "the container is refused when truncated or inconsistent" {
    const a = std.testing.allocator;
    try std.testing.expect(!isWebp("RIFF\x10\x00\x00\x00WEBPVP8Z"));
    // RIFF size beyond the bytes present
    try std.testing.expectError(error.BadWebp, decode(a, "RIFF\xff\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\x00", 1 << 20));
    // a VP8L chunk whose version bits are not zero
    try std.testing.expectError(error.BadWebp, decode(a, "RIFF\x12\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\x20\x00", 1 << 20));
}
