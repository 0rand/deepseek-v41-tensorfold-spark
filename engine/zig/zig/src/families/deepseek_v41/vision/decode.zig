//! Bytes -> Pillow's RGB by magic: PNG, JPEG, BMP and first-frame GIF / WebP bit for bit; other formats are refused.
const std = @import("std");
const pixels = @import("pixels.zig");
const png = @import("png.zig");
const jpeg = @import("jpeg.zig");
const gif = @import("gif.zig");
const bmp = @import("bmp.zig");
const webp = @import("webp.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ Undecodable, TooManyPixels, TooLarge, Unsupported } || Allocator.Error;

/// The most a WebP's Huffman groups and tables may allocate (``error.TooLarge`` past it).
pub const max_table_bytes = webp.max_table_bytes;

/// One image format: its magic test, its size from the header, its decoder.
pub const Decoder = struct {
    name: []const u8,
    matches: *const fn (raw: []const u8) bool,
    decode: *const fn (gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image,
};

fn isPng(raw: []const u8) bool {
    return std.mem.startsWith(u8, raw, png.signature);
}
fn isJpeg(raw: []const u8) bool {
    return raw.len >= 3 and raw[0] == 0xFF and raw[1] == 0xD8 and raw[2] == 0xFF;
}
fn decodePng(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    return png.decode(gpa, raw, max_pixels) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyPixels => error.TooManyPixels,
        error.UnsupportedPng => error.Unsupported,
        error.BadPng => error.Undecodable,
    };
}
fn decodeJpeg(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    return jpeg.decode(gpa, raw, max_pixels) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyPixels => error.TooManyPixels,
        error.UnsupportedJpeg => error.Unsupported,
        error.BadJpeg => error.Undecodable,
    };
}

fn decodeGif(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    return gif.decode(gpa, raw, max_pixels) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyPixels => error.TooManyPixels,
        error.BadGif => error.Undecodable,
    };
}
fn decodeBmp(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    return bmp.decode(gpa, raw, max_pixels) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyPixels => error.TooManyPixels,
        error.UnsupportedBmp => error.Unsupported,
        error.BadBmp => error.Undecodable,
    };
}

fn decodeWebp(gpa: Allocator, raw: []const u8, max_pixels: u64) Error!pixels.Image {
    return webp.decode(gpa, raw, max_pixels) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyPixels => error.TooManyPixels,
        error.TooLarge => error.TooLarge,
        error.BadWebp => error.Undecodable,
    };
}

pub const decoders = [_]Decoder{
    .{ .name = "PNG", .matches = isPng, .decode = decodePng },
    .{ .name = "JPEG", .matches = isJpeg, .decode = decodeJpeg },
    .{ .name = "GIF", .matches = gif.isGif, .decode = decodeGif },
    .{ .name = "BMP", .matches = bmp.isBmp, .decode = decodeBmp },
    .{ .name = "WebP", .matches = webp.isWebp, .decode = decodeWebp },
};

/// What a refused image is, for the message (formats Pillow would open but this engine does not).
pub fn formatName(raw: []const u8) []const u8 {
    if (std.mem.startsWith(u8, raw, "GIF87a") or std.mem.startsWith(u8, raw, "GIF89a")) return "GIF";
    if (std.mem.startsWith(u8, raw, "BM")) return "BMP";
    if (std.mem.startsWith(u8, raw, "II*\x00") or std.mem.startsWith(u8, raw, "MM\x00*")) return "TIFF";
    return "unknown";
}

/// The decoded image as RGB; ``which`` names the decoder used (or the refused format).
pub fn rgb(gpa: Allocator, raw: []const u8, max_pixels: u64, which: *[]const u8) Error!pixels.Rgb {
    for (decoders) |d| if (d.matches(raw)) {
        which.* = d.name;
        var im = try d.decode(gpa, raw, max_pixels);
        defer im.deinit(gpa);
        return pixels.toRgb(gpa, &im);
    };
    which.* = formatName(raw);
    return error.Unsupported;
}

test "a RIFF WEBP file no decoder matches is named unknown, not WebP" {
    try std.testing.expectEqualStrings("unknown", formatName("RIFF\x10\x00\x00\x00WEBPVP8Z\x00\x00\x00\x00"));
    var which: []const u8 = "";
    try std.testing.expectError(error.Unsupported, rgb(std.testing.allocator, "RIFF\x10\x00\x00\x00WEBPVP8Z\x00\x00\x00\x00", 1 << 20, &which));
    try std.testing.expectEqualStrings("unknown", which);
}
