//! Host tests of image prompts' buffer roles (`zig build test-dsv41-vision-plan`) on the real prefill emitter.

const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const vision_plan = @import("vision_plan.zig");
const vision_rows = @import("vision_rows.zig");
const Config = @import("config.zig").Config;

// the failing gate request's prompt rows (vgate-r1p request 4: one image and its text)
const rows: i64 = 232;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

const Fix = struct { cfg: Config = .{}, w: block.Widths, layers: [40]u32, o: block.Options = .{ .prefill_rows = rows } };

fn fixture(a: std.mem.Allocator) !Fix {
    var x: Fix = .{ .w = try widths(a), .layers = undefined };
    for (&x.layers, 0..) |*l, i| l.* = @intCast(i);
    return x;
}

fn planWith(a: std.mem.Allocator, x: *const Fix, images: bool, bias: bool) !buffers.Plan {
    var p: buffers.Plan = .{ .a = a };
    try p.add(try block_prefill.emitPrefill(a, &x.cfg, &x.w, vision_plan.planOptions(x.o, images, bias), &x.layers, rows, 0, true));
    return p;
}

test "a plan sized before images were on: the boot check names a missing image role, with and without the bias" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fixture(a);
    const off = try planWith(a, &x, false, false);
    try testing.expectEqualStrings("w.engram.keep", (try vision_plan.missing(a, &x.cfg, &x.w, x.o, &x.layers, rows, false, &off)).?);
    try testing.expect(try vision_plan.missing(a, &x.cfg, &x.w, x.o, &x.layers, rows, true, &off) != null);
    // the keep alone planned: the bias's own MoE call (image rows apart) is still missing
    const keep = try planWith(a, &x, true, false);
    try testing.expect(try vision_plan.missing(a, &x.cfg, &x.w, x.o, &x.layers, rows, false, &keep) == null);
    try testing.expect(try vision_plan.missing(a, &x.cfg, &x.w, x.o, &x.layers, rows, true, &keep) != null);
}

test "a plan sized with images on holds every role of an image segment at any split of its rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = try fixture(a);
    const on = try planWith(a, &x, true, true);
    try testing.expect(try vision_plan.missing(a, &x.cfg, &x.w, x.o, &x.layers, rows, true, &on) == null);
    // a served segment's options (vision_rows.segmentOptions): its image rows' count, 1 to all of them
    for ([_]i64{ 1, 196, rows - 1, rows }) |ni| {
        var o = x.o;
        o.image_keep = true;
        o.image_rows = ni;
        var seg: buffers.Plan = .{ .a = a };
        try seg.add(try block_prefill.emitPrefill(a, &x.cfg, &x.w, o, &x.layers, rows, 0, true));
        for (seg.sizes.keys(), seg.sizes.values()) |role, size| {
            const have = on.sizes.get(role) orelse 0;
            if (have < size) std.debug.print("image rows {d}: {s} needs {d}, planned {d}\n", .{ ni, role, size, have });
            try testing.expect(have >= size);
        }
    }
}

test "a window holding an image position is refused; text and the last vocabulary id pass" {
    try vision_rows.checkWindow(&.{ 1, 2, 3 });
    try vision_rows.checkWindow(&.{vision_rows.VBASE - 1});
    try testing.expectError(error.ImageInWindow, vision_rows.checkWindow(&.{ 5, vision_rows.VBASE }));
    try testing.expectError(error.ImageInWindow, vision_rows.checkWindow(&.{vision_rows.VBASE + 196}));
}
