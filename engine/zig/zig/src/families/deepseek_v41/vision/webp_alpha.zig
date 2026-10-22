//! A WebP ALPH chunk as libwebp 1.6.0 decodes it (``alpha_dec.c``, ``filters.c``): raw or VP8L, then unfiltered.
const std = @import("std");
const vp8l = @import("webp_vp8l.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadWebp, TooManyPixels, TooLarge } || Allocator.Error;

const Filter = enum(u2) { none, horizontal, vertical, gradient };

/// The w x h alpha plane of ``chunk`` (the ALPH payload) into ``out`` (len w * h).
pub fn decode(gpa: Allocator, chunk: []const u8, w: u32, h: u32, out: []u8) Error!void {
    const n = @as(usize, w) * h;
    if (chunk.len <= 1 or out.len != n) return error.BadWebp;
    const method = chunk[0] & 3;
    const filter: Filter = @fromBackingInt(@intCast((chunk[0] >> 2) & 3));
    const pre = (chunk[0] >> 4) & 3;
    if (method > 1 or pre > 1 or chunk[0] >> 6 != 0) return error.BadWebp;
    const body = chunk[1..];
    if (method == 0) {
        if (body.len < n) return error.BadWebp;
        unfilterPlane(filter, body[0..n], out, w);
        return;
    }
    const deltas = try gpa.alloc(u8, n);
    defer gpa.free(deltas);
    try vp8l.decodeAlpha(gpa, body, w, h, deltas);
    unfilterPlane(filter, deltas, out, w);
}

/// ``WebPUnfilters[filter]`` row by row, each row predicted from the previous output row (none for the first).
fn unfilterPlane(filter: Filter, in: []const u8, out: []u8, w: u32) void {
    var y: usize = 0;
    while (y * w < in.len) : (y += 1) {
        const row = in[y * w ..][0..w];
        const dst = out[y * w ..][0..w];
        const prev: ?[]const u8 = if (y == 0) null else out[(y - 1) * w ..][0..w];
        switch (filter) {
            .none => @memcpy(dst, row),
            .horizontal => horizontal(prev, row, dst),
            .vertical => if (prev) |p| {
                for (dst, row, p) |*o, d, t| o.* = d +% t;
            } else horizontal(null, row, dst),
            .gradient => if (prev) |p| gradient(p, row, dst) else horizontal(null, row, dst),
        }
    }
}

/// ``HorizontalUnfilter_C``: the first pixel predicted from the row above (0 on the first row), then from the left.
fn horizontal(prev: ?[]const u8, in: []const u8, out: []u8) void {
    var pred: u8 = if (prev) |p| p[0] else 0;
    for (in, out) |d, *o| {
        o.* = pred +% d;
        pred = o.*;
    }
}

/// ``GradientUnfilter_C`` below the first row: left + top - top-left, clipped to 8 bits.
fn gradient(prev: []const u8, in: []const u8, out: []u8) void {
    var top: u8 = prev[0];
    var top_left = top;
    var left = top;
    for (in, out, prev) |d, *o, t| {
        top = t;
        const g = @as(i32, left) + top - top_left;
        left = d +% @as(u8, @intCast(std.math.clamp(g, 0, 255)));
        top_left = top;
        o.* = left;
    }
}

test "the three unfilters rebuild their filtered rows" {
    // a 3 x 2 plane: rows [10, 20, 30] and [15, 5, 255]
    var out: [6]u8 = undefined;
    unfilterPlane(.horizontal, &.{ 10, 10, 10, 5, 246, 250 }, &out, 3);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 15, 5, 255 }, &out);
    unfilterPlane(.vertical, &.{ 10, 10, 10, 5, 241, 225 }, &out, 3);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 15, 5, 255 }, &out);
    // gradient row 1: preds clip(10 + 10 - 10) = 10, clip(15 + 20 - 10) = 25, clip(5 + 30 - 20) = 15
    unfilterPlane(.gradient, &.{ 10, 10, 10, 5, 236, 240 }, &out, 3);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 15, 5, 255 }, &out);
}

test "a raw plane decodes and a bad header is refused" {
    var out: [4]u8 = undefined;
    try decode(std.testing.allocator, &.{ 0, 1, 2, 3, 4 }, 2, 2, &out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    try std.testing.expectError(error.BadWebp, decode(std.testing.allocator, &.{ 0, 1, 2, 3 }, 2, 2, &out));
    try std.testing.expectError(error.BadWebp, decode(std.testing.allocator, &.{ 2, 1, 2, 3, 4 }, 2, 2, &out));
    try std.testing.expectError(error.BadWebp, decode(std.testing.allocator, &.{ 0x40, 1, 2, 3, 4 }, 2, 2, &out));
}
