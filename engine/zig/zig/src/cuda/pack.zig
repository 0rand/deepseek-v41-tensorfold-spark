//! TF_DSV41_ARENA=pack: boot's permanent buffers bumped back to back in one growing mapping, not 2 MiB-rounded each.

const std = @import("std");
const memory = @import("memory.zig");
const arena = @import("arena.zig");

pub const Pack = struct {
    gpa: std.mem.Allocator,
    be: arena.Backend,
    alignment: usize = 512,
    prev: ?memory.Hook = null,
    open: bool = false,
    mutex: std.atomic.Mutex = .unlocked,
    /// mapped segments (base, len): one unless the backend hands out a range not following the last
    segments: std.ArrayList([2]u64) = .empty,
    cursor: u64 = 0,
    limit: u64 = 0,
    stats: struct { buffers: u64 = 0, bytes: u64 = 0, mapped: u64 = 0, freed: u64 = 0, outside: u64 = 0 } = .{},

    pub fn init(gpa: std.mem.Allocator, be: arena.Backend) Pack {
        return .{ .gpa = gpa, .be = be };
    }

    /// Under DeviceBuffer.alloc / free, over whatever hook was there (the pack must outlive every buffer it hands out).
    pub fn install(p: *Pack) void {
        p.prev = memory.currentHook();
        memory.setHook(.{ .ctx = p, .alloc = hookAlloc, .free = hookFree });
    }

    pub fn begin(p: *Pack) void {
        p.open = true;
    }

    pub fn end(p: *Pack) void {
        p.open = false;
    }

    fn lock(p: *Pack) void {
        while (!p.mutex.tryLock()) std.Thread.yield() catch {};
    }

    /// An address for `len` bytes while the phase is open; else the previous hook's (null: cuMemAlloc).
    pub fn alloc(p: *Pack, len: usize) ?u64 {
        if (len == 0) return null;
        p.lock();
        defer p.mutex.unlock();
        if (!p.open) {
            p.stats.outside += 1;
            const h = p.prev orelse return null;
            return h.alloc(h.ctx, len);
        }
        const at = std.mem.alignForward(u64, p.cursor, p.alignment);
        if (at + len > p.limit) {
            const grow = std.mem.alignForward(usize, @intCast(at + len - p.limit), p.be.granularity);
            if (p.limit != 0) if (p.be.chunk(p.be.ctx, grow)) |base| {
                p.stats.mapped += grow;
                if (base == p.limit) {
                    p.segments.items[p.segments.items.len - 1][1] += grow;
                    p.limit += grow;
                    return p.place(len);
                }
                // a range not following the last (another user of the backend mapped in between): give it back
                p.be.release(p.be.ctx, base, grow);
                p.stats.mapped -= grow;
            };
            // the first segment, or a new one: the buffer's own pages to start it
            const n = std.mem.alignForward(usize, len, p.be.granularity);
            const base = p.be.chunk(p.be.ctx, n) orelse return null;
            p.segments.append(p.gpa, .{ base, n }) catch return null;
            p.stats.mapped += n;
            p.cursor = base;
            p.limit = base + n;
        }
        return p.place(len);
    }

    fn place(p: *Pack, len: usize) u64 {
        const at = std.mem.alignForward(u64, p.cursor, p.alignment);
        p.cursor = at + len;
        p.stats.buffers += 1;
        p.stats.bytes += len;
        return at;
    }

    fn owns(p: *const Pack, x: u64) bool {
        for (p.segments.items) |s| if (x >= s[0] and x < s[0] + s[1]) return true;
        return false;
    }

    /// A packed address stays mapped until close (true: ours); any other goes to the previous hook.
    pub fn free(p: *Pack, x: u64) bool {
        p.lock();
        defer p.mutex.unlock();
        if (p.owns(x)) {
            p.stats.freed += 1;
            return true;
        }
        const h = p.prev orelse return false;
        return h.free(h.ctx, x);
    }

    /// Mapped bytes past the packed ones: the segments' tails and alignment (what packing still costs).
    pub fn slack(p: *const Pack) u64 {
        return p.stats.mapped - p.stats.bytes;
    }

    fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
        const p: *Pack = @ptrCast(@alignCast(ctx));
        return p.alloc(len);
    }

    fn hookFree(ctx: *anyopaque, x: u64) bool {
        const p: *Pack = @ptrCast(@alignCast(ctx));
        return p.free(x);
    }
};

// -- tests: a fake backend handing out contiguous ranges, and a previous hook that counts ----------------------------

const Fake = struct {
    next: u64 = 1 << 40,
    chunks: u32 = 0,
    gap: bool = false,

    fn backend(f: *Fake, g: usize) arena.Backend {
        return .{ .ctx = f, .granularity = g, .chunk = chunk, .release = release, .name = "fake" };
    }
    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.gap) f.next += 4096; // another mapping in between
        const at = f.next;
        f.next += len;
        f.chunks += 1;
        return at;
    }
    fn release(_: *anyopaque, _: u64, _: usize) void {}
};

const Prev = struct {
    allocs: u32 = 0,
    frees: u32 = 0,
    fn hook(p: *Prev) memory.Hook {
        return .{ .ctx = p, .alloc = a, .free = f };
    }
    fn a(ctx: *anyopaque, _: usize) ?u64 {
        const p: *Prev = @ptrCast(@alignCast(ctx));
        p.allocs += 1;
        return 0x7000_0000;
    }
    fn f(ctx: *anyopaque, _: u64) bool {
        const p: *Prev = @ptrCast(@alignCast(ctx));
        p.frees += 1;
        return true;
    }
};

test "pack: permanent buffers back to back, 512-aligned, under a granularity page of slack in all" {
    const t = std.testing;
    var fake: Fake = .{};
    var prev: Prev = .{};
    var p = Pack.init(t.allocator, fake.backend(2 << 20));
    defer p.segments.deinit(t.allocator);
    p.prev = prev.hook();
    p.begin();
    // the weights' mix: many tensors of odd sizes, a few large ones
    var sizes: [600]usize = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&sizes, 0..) |*s, i| s.* = if (i % 50 == 0) 300 << 20 else prng.random().intRangeAtMost(usize, 1000, 9 << 20);
    var total: u64 = 0;
    var last: u64 = 0;
    for (sizes) |s| {
        const x = p.alloc(s).?;
        try t.expectEqual(@as(u64, 0), x % 512);
        try t.expect(x >= last); // in order, never overlapping the one before
        last = x + s;
        total += s;
    }
    try t.expectEqual(@as(usize, 1), p.segments.items.len);
    try t.expect(p.slack() < (2 << 20) + 600 * 512);
    // one cuMemAlloc each would round each to 2 MiB: far more than the pack's slack
    var rounded: u64 = 0;
    for (sizes) |s| rounded += std.mem.alignForward(u64, s, 2 << 20);
    try t.expect(rounded - total > 100 * p.slack());
    // a packed buffer's free is ours and a no-op; after the phase the previous hook serves and frees
    const kept = p.stats.mapped;
    try t.expect(p.free(last - 1));
    p.end();
    try t.expectEqual(@as(?u64, 0x7000_0000), p.alloc(4096));
    try t.expect(p.free(0x7000_0000));
    try t.expectEqual(@as(u32, 1), prev.allocs);
    try t.expectEqual(@as(u32, 1), prev.frees);
    try t.expectEqual(kept, p.stats.mapped);
}

test "pack: a range that does not follow the last starts a new segment" {
    const t = std.testing;
    var fake: Fake = .{};
    var p = Pack.init(t.allocator, fake.backend(2 << 20));
    defer p.segments.deinit(t.allocator);
    p.begin();
    _ = p.alloc(3 << 20).?;
    fake.gap = true;
    const x = p.alloc(3 << 20).?;
    try t.expectEqual(@as(usize, 2), p.segments.items.len);
    try t.expect(p.owns(x) and !p.owns(1));
}
