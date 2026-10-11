//! Native images with the bias, split KV off (`zig build test-dsv41-knobs`): the runs and PF_TBO plan at runOptions.
const std = @import("std");
const testing = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const calls = @import("calls.zig");
const check = @import("m1_check.zig");
const fwd = @import("forward.zig");
const fp = @import("forward_prefill.zig");
const pk = @import("prod_knobs.zig");
const vision_rows = @import("vision_rows.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 8192;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

/// A host forward over layers 0-24, images and the bias on, split KV off; no runner or collective is reached.
fn forward(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, layers: []const u32, mode: pk.PrefillMode) fwd.Forward {
    const pages = @divExact(limit, block.Pool.page);
    var o: block.Options = .{ .limit = limit, .rope_rows = limit + 2048, .branches = true };
    o.pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages, .split = false };
    var f = fwd.Forward.init(a, cfg, w, o, undefined, undefined);
    f.layers = layers;
    f.prefill_state.mode = mode;
    f.images = true;
    f.image_bias = true;
    return f;
}

/// The roles of `cs` the plan does not hold at their bytes.
fn unplanned(a: std.mem.Allocator, plan: *const buffers.Plan, cs: []const calls.Call) !usize {
    var need: buffers.Plan = .{ .a = a };
    try need.add(cs);
    var gaps: usize = 0;
    for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
        if ((plan.sizes.get(role) orelse 0) >= bytes) continue;
        std.debug.print("role {s} ({d} bytes) is in no planned program\n", .{ role, bytes });
        gaps += 1;
    }
    return gaps;
}

test "native images with the bias: the plan holds the multi-segment runs' roles (runOptions), full and replay" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*x, i| x.* = @intCast(i);
    for ([_]pk.PrefillMode{ .full, .replay }) |mode| {
        var f = forward(a, &cfg, &w, &layers, mode);
        try testing.expectEqual(@as(i64, -1), vision_rows.planOptions(&f).image_rows);
        var plan: buffers.Plan = .{ .a = a };
        try fp.plan(&f, &plan, 2048);
        try testing.expect(plan.sizes.get("w.mpos64") != null);
        if (mode != .full) continue;
        // two text prompts' segments in one run (2031 + 17 rows), as the forward runs them with images on
        var ids: [17]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = 100 + @as(u32, @intCast(i));
        const sp = [_]bp.Span{ .{ .slot = 0, .start = 0, .n = 2048 - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
        const cs = try bp.emitMulti(a, &cfg, &w, vision_rows.segmentOptions(&f, &ids), &layers, &sp, .whole);
        try testing.expectEqual(@as(usize, 0), try unplanned(a, &plan, cs));
    }
}

test "native images with the bias: TF_DSV41_PF_TBO's workspace sizes and plans (runOptions)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    var layers: [25]u32 = undefined;
    for (&layers, 0..) |*x, i| x.* = @intCast(i);
    var f = forward(a, &cfg, &w, &layers, .replay);
    f.opts.pf_tbo = true;
    try testing.expect(try fp.workspaceBytes(&f, 2048) > 0);
    var plan: buffers.Plan = .{ .a = a };
    try fp.plan(&f, &plan, 2048);
    try testing.expect(plan.sizes.get("w.mpos64") != null);
}
