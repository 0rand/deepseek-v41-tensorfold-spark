//! The VP8L table budget: a meta image naming 65,536 groups is refused, and the refusal bounds the peak.
const std = @import("std");
const vp8l = @import("webp_vp8l.zig");
const Allocator = std.mem.Allocator;

/// An allocator over ``child`` that records the most bytes live at once; ``in_place`` false never grows in place.
const PeakAllocator = struct {
    child: Allocator,
    in_place: bool = true,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(p: *PeakAllocator) Allocator {
        return .{ .ptr = p, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn grew(p: *PeakAllocator, old: usize, new: usize) void {
        p.live = p.live - old + new;
        p.peak = @max(p.peak, p.live);
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const p: *PeakAllocator = @ptrCast(@alignCast(ctx));
        const r = p.child.rawAlloc(len, a, ra) orelse return null;
        p.grew(0, len);
        return r;
    }
    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const p: *PeakAllocator = @ptrCast(@alignCast(ctx));
        if (!p.in_place or !p.child.rawResize(m, a, n, ra)) return false;
        p.grew(m.len, n);
        return true;
    }
    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const p: *PeakAllocator = @ptrCast(@alignCast(ctx));
        if (!p.in_place) return null;
        const r = p.child.rawRemap(m, a, n, ra) orelse return null;
        p.grew(m.len, n);
        return r;
    }
    fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const p: *PeakAllocator = @ptrCast(@alignCast(ctx));
        p.child.rawFree(m, a, ra);
        p.grew(m.len, 0);
    }
};

/// An LSB-first bit writer, as ``BitReader`` reads.
const BitWriter = struct {
    out: std.ArrayList(u8) = .empty,
    n: u3 = 0,

    fn put(w: *BitWriter, gpa: Allocator, v: u32, bits: u5) !void {
        for (0..bits) |i| {
            if (w.n == 0) try w.out.append(gpa, 0);
            w.out.items[w.out.items.len - 1] |= (@as(u8, @truncate(v >> @intCast(i))) & 1) << w.n;
            w.n +%= 1;
        }
    }
    /// A simple code of one 1-bit symbol, 0.
    fn single(w: *BitWriter, gpa: Allocator) !void {
        try w.put(gpa, 0b0001, 4);
    }
    /// A length-coded code of ``size`` symbols: 0..255 at 8 bits, the rest unused.
    fn flat(w: *BitWriter, gpa: Allocator, size: u32) !void {
        try w.put(gpa, 0, 1);
        try w.put(gpa, 12 - 4, 4);
        for (vp8l.code_length_order[0..12]) |o| try w.put(gpa, if (o == 0 or o == 8) 1 else 0, 3);
        try w.put(gpa, 0, 1);
        for (0..size) |s| try w.put(gpa, if (s < 256) 1 else 0, 1);
    }
};

/// A ``side``-square VP8L stream naming ``(side / 4)^2`` groups of five one-symbol codes, about 2.5 bytes each.
fn manyGroups(gpa: Allocator, side: u32) ![]u8 {
    var w: BitWriter = .{};
    errdefer w.out.deinit(gpa);
    try w.put(gpa, 0x2f, 8);
    try w.put(gpa, side - 1, 14);
    try w.put(gpa, side - 1, 14);
    try w.put(gpa, 0, 4); // no alpha hint, version 0
    try w.put(gpa, 0, 2); // no transform, no colour cache
    try w.put(gpa, 1, 1); // a meta image ...
    try w.put(gpa, 0, 3); // ... at 2 bits
    try w.put(gpa, 0, 1); // its stream: no colour cache, then green, red, blue, alpha, distance
    try w.flat(gpa, vp8l.len_code_limit);
    try w.flat(gpa, 256);
    for (0..3) |_| try w.single(gpa);
    const groups = (side / 4) * (side / 4);
    for (0..groups) |i| { // the group index is red << 8 | green
        try w.put(gpa, @bitReverse(@as(u8, @truncate(i))), 8);
        try w.put(gpa, @bitReverse(@as(u8, @truncate(i >> 8))), 8);
    }
    for (0..groups * 5) |_| try w.single(gpa);
    try w.out.appendNTimes(gpa, 0, 8);
    return w.out.toOwnedSlice(gpa);
}

test "a meta image of 65,536 groups is refused at the table budget, which bounds the peak" {
    const data = try manyGroups(std.testing.allocator, 1024);
    defer std.testing.allocator.free(data);
    for ([2]bool{ true, false }) |in_place| { // an allocator that never grows in place must not raise the peak
        var pa: PeakAllocator = .{ .child = std.testing.allocator, .in_place = in_place };
        try std.testing.expectError(error.TooLarge, vp8l.decode(pa.allocator(), data, 1 << 20));
        try std.testing.expectEqual(@as(usize, 0), pa.live);
        try std.testing.expect(pa.peak <= vp8l.max_table_bytes + (2 << 20));
    }
    // the same shape at 4,096 groups decodes: every pixel group 0's literal, 0
    var pa: PeakAllocator = .{ .child = std.testing.allocator };
    const small = try manyGroups(std.testing.allocator, 256);
    defer std.testing.allocator.free(small);
    var img = try vp8l.decode(pa.allocator(), small, 1 << 20);
    defer img.deinit(pa.allocator());
    try std.testing.expectEqual(@as(u32, 256), img.w);
    for (img.px) |p| try std.testing.expectEqual(@as(u32, 0), p);
}
