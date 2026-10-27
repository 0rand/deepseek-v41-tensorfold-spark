//! WebP files decoded to Pillow 12.3's RGB (tools/zig/dsv41_vision/gen_webp_golden.py); TF_DSV41_WEBP_GOLDEN: more.
const std = @import("std");
const testing = std.testing;
const decode = @import("decode.zig");

const fixtures = "zig/src/families/deepseek_v41/vision/fixtures/golden-webp";

/// The first record: the Pillow and libwebp versions the goldens came from.
const Generator = struct { pillow: []const u8, libwebp: []const u8 };

/// A file Pillow decodes (``size``, ``rgb_sha256``) or refuses (``refused``); or the ``generator`` record.
const Rec = struct { name: []const u8 = "", size: [2]u32 = .{ 0, 0 }, rgb_sha256: []const u8 = "", refused: bool = false, generator: ?Generator = null };

/// Every record of ``dir``/golden.jsonl; returns (equal, total).
fn run(dir: []const u8) ![2]usize {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const lines = try cwd.readFileAlloc(testing.io, try std.fmt.allocPrint(ar, "{s}/golden.jsonl", .{dir}), ar, .limited(1 << 26));
    var ok: usize = 0;
    var total: usize = 0;
    var it = std.mem.tokenizeScalar(u8, lines, '\n');
    while (it.next()) |line| {
        const r = try std.json.parseFromSliceLeaky(Rec, ar, line, .{});
        if (r.generator) |g| {
            std.debug.print("webp golden {s}: Pillow {s}, libwebp {s}\n", .{ dir, g.pillow, g.libwebp });
            continue;
        }
        total += 1;
        const raw = try cwd.readFileAlloc(testing.io, try std.fmt.allocPrint(ar, "{s}/{s}", .{ dir, r.name }), ar, .limited(1 << 28));
        var which: []const u8 = "";
        var rgb = decode.rgb(a, raw, 1 << 32, &which) catch |e| {
            if (r.refused and e == error.Undecodable) ok += 1 else std.debug.print("webp golden {s}: decode failed ({t})\n", .{ r.name, e });
            continue;
        };
        if (r.refused) {
            std.debug.print("webp golden {s}: decoded, Pillow refuses it\n", .{r.name});
            rgb.deinit(a);
            continue;
        }
        defer rgb.deinit(a);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(rgb.data, &d, .{});
        var hb: [64]u8 = undefined;
        const got = std.fmt.bufPrint(&hb, "{x}", .{d}) catch unreachable;
        if (rgb.w != r.size[0] or rgb.h != r.size[1] or !std.mem.eql(u8, got, r.rgb_sha256)) {
            std.debug.print("webp golden {s}: RGB differs from Pillow's\n", .{r.name});
        } else ok += 1;
    }
    return .{ ok, total };
}

test "WebP files decode to Pillow's RGB (committed set)" {
    const got = try run(fixtures);
    std.debug.print("webp golden {s}: {d}/{d} images equal\n", .{ fixtures, got[0], got[1] });
    try testing.expect(got[1] > 0);
    try testing.expectEqual(got[1], got[0]);
}

test "WebP files decode to Pillow's RGB (TF_DSV41_WEBP_GOLDEN)" {
    const dir = testing.environ.getPosix("TF_DSV41_WEBP_GOLDEN") orelse return error.SkipZigTest;
    const got = try run(dir);
    std.debug.print("webp golden {s}: {d}/{d} images equal\n", .{ dir, got[0], got[1] });
    try testing.expectEqual(got[1], got[0]);
}
