//! Image prompts' buffer roles (TF_DSV41_IMAGES=native): the plan's image options and its boot check.

const std = @import("std");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const Config = @import("config.zig").Config;

/// The plan's options: the keep buffer when images are on, both MoE calls at full rows with the image bias.
pub fn planOptions(o: block.Options, images: bool, bias: bool) block.Options {
    var p = o;
    p.image_keep = images;
    if (images and bias) p.image_rows = -1;
    return p;
}

/// The first role a `rows`-row image segment adds to the same text segment's that `plan` lacks or holds smaller.
pub fn missing(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, rows: i64, bias: bool, plan: *const buffers.Plan) !?[]const u8 {
    var text: buffers.Plan = .{ .a = a };
    try text.add(try block_prefill.emitPrefill(a, cfg, w, planOptions(o, false, false), layers, rows, 0, true));
    var image: buffers.Plan = .{ .a = a };
    try image.add(try block_prefill.emitPrefill(a, cfg, w, planOptions(o, true, bias), layers, rows, 0, true));
    for (image.sizes.keys(), image.sizes.values()) |role, size| {
        if ((text.sizes.get(role) orelse 0) >= size) continue;
        if ((plan.sizes.get(role) orelse 0) < size) return role;
    }
    return null;
}
