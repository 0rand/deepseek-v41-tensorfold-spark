//! aot-needs' AOTDIR: whether a filled AOT set holds every variant the env launches, matched as `Set.find` does.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const triton_call = @import("triton_call.zig");

const aot = cuda.aot;

pub const Cover = struct {
    gpa: std.mem.Allocator,
    specs: []const aot.Spec,
    /// the variant keys (aot_needs.Needs.reduce) no spec matches, in the order they were met
    missing: std.ArrayList([]const u8) = .empty,
    checked: usize = 0,

    pub fn deinit(c: *Cover) void {
        for (c.missing.items) |k| c.gpa.free(k);
        c.missing.deinit(c.gpa);
    }

    /// Checks a Triton call met under a new variant key `key`: some compiled variant of its function must match it.
    pub fn check(c: *Cover, call: *const calls.Call, key: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(c.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        c.checked += 1;
        if (try c.covered(a, call)) return;
        try c.missing.append(c.gpa, try c.gpa.dupe(u8, key));
    }

    fn covered(c: *const Cover, a: std.mem.Allocator, call: *const calls.Call) !bool {
        var mine: std.ArrayList(aot.Spec) = .empty;
        for (c.specs) |s| if (std.mem.eql(u8, s.@"fn", call.name)) try mine.append(a, s);
        if (mine.items.len == 0) return false;
        var dummy: u8 = 0;
        const sp = triton_call.split(a, call, mine.items, .{ .ctx = &dummy, .of = base }) catch return false;
        for (mine.items) |s| if (aot.specMatches(s, sp.args, sp.consts)) return true;
        return false;
    }

    /// A tensor at a 256-aligned buffer plus its view's offset (the runner's and the pool's bases are aligned).
    fn base(_: *anyopaque, t: calls.Tensor) anyerror!u64 {
        var numel: i64 = 1;
        for (t.shape) |n| numel *= n;
        return if (numel == 0) 0 else 0x7f00_0000_0000 + @as(u64, @intCast(t.offset));
    }

    /// The verdict on `log`: the missing keys (up to `show`) and how to fill them. Returns whether the set is complete.
    pub fn report(c: *const Cover, dir: []const u8, total: usize, log: *std.Io.Writer, show: usize) !bool {
        if (c.missing.items.len == 0) {
            try log.print("aot-needs: {s} holds all {d} variants this env launches\n", .{ dir, total });
            return true;
        }
        try log.print("aot-needs: {s} lacks {d} of {d} variants this env launches (MissingTritonVariant at boot or serving):\n", .{ dir, c.missing.items.len, total });
        for (c.missing.items[0..@min(show, c.missing.items.len)]) |k| try log.print("  {s}\n", .{k});
        if (c.missing.items.len > show) try log.print("  ... and {d} more (all in OUT)\n", .{c.missing.items.len - show});
        try log.print("aot-needs: compile them into the same set: tools/zig/triton_fill.py compile --needs OUT --have {s} --out {s} (with the fill's --py / --options / --arch)\n", .{ dir, dir });
        return false;
    }
};

// -- host tests ---------------------------------------------------------------------------------------------------

const testing = std.testing;

test "aot cover: a call is covered by a matching compiled variant, missing when its constexpr differs" {
    const json =
        \\{"kernels":[{"fn":"_k","hash":"h1","name":"_k","num_warps":4,
        \\ "params":[{"name":"X","type":"*bf16","div16":true,"nospec":false}],"consts":{"CBS":{"int":64}}}]}
    ;
    const parsed = try aot.parseSpecs(testing.allocator, json);
    defer parsed.deinit();
    var c: Cover = .{ .gpa = testing.allocator, .specs = parsed.value.kernels };
    defer c.deinit();
    const t = calls.Tensor{ .role = .{ .buf = "w.x" }, .dt = .bf16, .shape = &.{4}, .stride = &.{1}, .offset = 0 };
    const hit: calls.Call = .{ .triton = true, .name = "_k", .args = &.{ .{ .name = "X", .arg = .{ .t = t } }, .{ .name = "CBS", .arg = .{ .i = 64 } } } };
    const miss: calls.Call = .{ .triton = true, .name = "_k", .args = &.{ .{ .name = "X", .arg = .{ .t = t } }, .{ .name = "CBS", .arg = .{ .i = 0 } } } };
    const other: calls.Call = .{ .triton = true, .name = "_j", .args = &.{.{ .name = "X", .arg = .{ .t = t } }} };
    try c.check(&hit, "hit");
    try c.check(&miss, "miss");
    try c.check(&other, "other");
    try testing.expectEqual(@as(usize, 3), c.checked);
    try testing.expectEqual(@as(usize, 2), c.missing.items.len);
    try testing.expectEqualStrings("miss", c.missing.items[0]);
    try testing.expectEqualStrings("other", c.missing.items[1]);
}
