//! Virtual token ids (GLM's ``vision_prep.derive`` / ``Registry``): span position k of an image is
//! VBASE + blake2b-64(sha256(digest || salt) || k) mod VSPAN, so every token-keyed cache tells images apart; the
//! registry bumps an image's salt when its ids would meet another image's.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const VBASE: u32 = 1 << 24;
pub const VLIMIT: u32 = (1 << 31) - 1;
pub const VSPAN: u64 = VLIMIT - VBASE;

pub fn isVid(t: u32) bool {
    return t >= VBASE and t <= VLIMIT;
}

/// ``derive(digest, n, salt)`` into ``out``.
pub fn derive(digest: *const [32]u8, salt: u32, out: []u32) void {
    var base: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(digest);
    var sb: [4]u8 = undefined;
    std.mem.writeInt(u32, &sb, salt, .little);
    h.update(&sb);
    h.final(&base);
    for (out, 0..) |*v, k| {
        var b = std.crypto.hash.blake2.Blake2b(64).init(.{});
        b.update(&base);
        var kb: [4]u8 = undefined;
        std.mem.writeInt(u32, &kb, @intCast(k), .little);
        b.update(&kb);
        var d: [8]u8 = undefined;
        b.final(&d);
        v.* = @intCast(VBASE + std.mem.readInt(u64, &d, .little) % VSPAN);
    }
}

/// Rank 0's recent ids -> the image digest they went to, at most ``cap`` ids (oldest first out), and each digest's
/// salt. Thread-safe.
pub const Registry = struct {
    gpa: Allocator,
    cap: usize = 1 << 18,
    mutex: std.Io.Mutex = .init,
    owner: std.AutoHashMapUnmanaged(u32, [32]u8) = .empty,
    order: std.ArrayList(u32) = .empty, // insertion order of ``owner`` (a ring once full)
    head: usize = 0,
    salts: std.AutoHashMapUnmanaged([32]u8, u32) = .empty,
    /// ids dropped from ``owner`` since it was last rehashed
    dead: u32 = 0,

    pub fn deinit(r: *Registry) void {
        r.owner.deinit(r.gpa);
        r.order.deinit(r.gpa);
        r.salts.deinit(r.gpa);
    }

    /// The image's ids: ``n`` of them, derived with the first salt that clashes with no other image's.
    pub fn vids(r: *Registry, io: std.Io, digest: *const [32]u8, n: usize) Allocator.Error![]u32 {
        const out = try r.gpa.alloc(u32, n);
        errdefer r.gpa.free(out);
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        var salt: u32 = r.salts.get(digest.*) orelse 0;
        var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen.deinit(r.gpa);
        while (true) : (salt += 1) {
            derive(digest, salt, out);
            seen.clearRetainingCapacity();
            var clash = false;
            for (out) |v| {
                if ((try seen.fetchPut(r.gpa, v, {})) != null) clash = true;
                if (r.owner.get(v)) |d| if (!std.mem.eql(u8, &d, digest)) {
                    clash = true;
                };
                if (clash) break;
            }
            if (!clash) break;
        }
        if (r.salts.count() >= r.cap / 16 and !r.salts.contains(digest.*)) r.salts.clearRetainingCapacity();
        try r.salts.put(r.gpa, digest.*, salt);
        for (out) |v| {
            const got = try r.owner.getOrPut(r.gpa, v);
            got.value_ptr.* = digest.*;
            if (got.found_existing) continue;
            if (r.order.items.len < r.cap) {
                try r.order.append(r.gpa, v);
            } else {
                if (r.owner.remove(r.order.items[r.head])) r.dropped();
                r.order.items[r.head] = v;
                r.head = (r.head + 1) % r.cap;
            }
        }
        return out;
    }

    /// std's tombstones never clear and keys fill 80 %: rehash at a tenth, so a tenth of the map always stays free.
    fn dropped(r: *Registry) void {
        r.dead += 1;
        if (@as(u64, r.dead) * 10 < r.owner.capacity()) return;
        r.owner.rehash(std.hash_map.AutoContext(u32){});
        r.dead = 0;
    }
};

test "virtual ids are past the vocabulary, stable, and distinct per salt" {
    var d: [32]u8 = undefined;
    @memset(&d, 7);
    var a: [5]u32 = undefined;
    var b: [5]u32 = undefined;
    derive(&d, 0, &a);
    derive(&d, 0, &b);
    try std.testing.expectEqualSlices(u32, &a, &b);
    for (a) |v| try std.testing.expect(isVid(v));
    derive(&d, 1, &b);
    try std.testing.expect(!std.mem.eql(u32, &a, &b));
}

test "a full registry's id churn leaves its owner map free slots" {
    const t = std.testing;
    var r: Registry = .{ .gpa = t.allocator, .cap = 4096 };
    defer r.deinit();
    for (0..400) |i| {
        // a new image's ids each evict the oldest once the ring is full
        var digest: [32]u8 = @splat(0);
        std.mem.writeInt(u64, digest[0..8], i, .little);
        t.allocator.free(try r.vids(t.io, &digest, 64));
        const meta = r.owner.metadata.?;
        var free: usize = 0;
        for (meta[0..r.owner.capacity()]) |x| free += @intFromBool(x.isFree());
        try t.expect(free * 10 >= r.owner.capacity());
    }
}
