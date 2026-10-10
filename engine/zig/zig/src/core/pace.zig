//! The engine's round pace against its own baseline, for /health: decode round time and prefill rows a second.
const std = @import("std");

/// A quick and a slow average of one measure; the slow one (the baseline) moves only while the quick one is near it.
pub const Track = struct {
    quick: f64 = 0,
    base: f64 = 0,
    samples: u32 = 0,
    quick_alpha: f64,
    base_alpha: f64,
    /// samples before the baseline counts
    warm: u32,

    pub fn add(t: *Track, x: f64) void {
        t.samples +|= 1;
        if (t.samples == 1) {
            t.quick = x;
            t.base = x;
            return;
        }
        t.quick += t.quick_alpha * (x - t.quick);
        // a degraded engine must not drag its own baseline along
        if (t.quick <= 2 * t.base and t.quick * 2 >= t.base) t.base += t.base_alpha * (x - t.base);
    }

    pub fn ready(t: *const Track) bool {
        return t.samples >= t.warm and t.base > 0;
    }
};

pub const Limits = struct {
    /// decode round time over its baseline that counts as slow
    decode_x: f64 = 4,
    /// prefill rows a second under its baseline by this factor counts as slow
    prefill_x: f64 = 8,
};

pub const Pace = struct {
    limits: Limits = .{},
    decode: Track = .{ .quick_alpha = 0.1, .base_alpha = 0.01, .warm = 50 },
    prefill: Track = .{ .quick_alpha = 0.3, .base_alpha = 0.05, .warm = 10 },
    /// the last round's end and length (ns), null before the first
    last_end: ?i96 = null,
    last_ns: u64 = 0,
    /// since when the pace has been slow, null while it is not
    slow_since: ?i96 = null,

    /// A round ended at `now`: its prompt rows and their seconds, and its decode seconds (0: none ran).
    pub fn round(p: *Pace, now: i96, rows: u64, piece_s: f64, decode_s: f64) void {
        p.last_ns = @intFromFloat(@max(0, piece_s + decode_s) * 1e9);
        p.last_end = now;
        if (rows > 0 and piece_s > 0) p.prefill.add(@as(f64, @floatFromInt(rows)) / piece_s);
        if (decode_s > 0) p.decode.add(decode_s);
        const slow = (p.decode.ready() and p.decode.quick > p.limits.decode_x * p.decode.base) or
            (p.prefill.ready() and p.prefill.quick * p.limits.prefill_x < p.prefill.base);
        if (!slow) p.slow_since = null else if (p.slow_since == null) p.slow_since = now;
    }

    /// Seconds the pace has been slow at `now` (0: it is not).
    pub fn slowFor(p: *const Pace, now: i96) f64 {
        const s = p.slow_since orelse return 0;
        return @as(f64, @floatFromInt(now - s)) / 1e9;
    }
};

test "pace: a degraded engine is slow against its baseline; a long context's slower rounds are not" {
    const s: i96 = std.time.ns_per_s;
    var p: Pace = .{};
    var t: i96 = 0;
    // an hour of normal rounds: 70 ms decode rounds, 4,096-row pieces at ~2,000 rows a second
    for (0..2000) |i| {
        t += s / 10;
        if (i % 20 == 0) p.round(t, 4096, 2.0, 0.07) else p.round(t, 0, 0, 0.07);
    }
    try std.testing.expectEqual(@as(f64, 0), p.slowFor(t));
    // a long context: decode rounds 2.5x, prefill 4x slower: not slow
    for (0..600) |i| {
        t += s / 4;
        if (i % 10 == 0) p.round(t, 4096, 8.0, 0.175) else p.round(t, 0, 0, 0.175);
    }
    try std.testing.expectEqual(@as(f64, 0), p.slowFor(t));
    // back to normal, then a measured stall's shape: 380 ms rounds, 4,096-row pieces of 41 s
    for (0..600) |_| {
        t += s / 10;
        p.round(t, 0, 0, 0.07);
    }
    const start = t;
    var tripped: ?i96 = null;
    for (0..2000) |i| {
        t += s / 2;
        if (i % 40 == 0) p.round(t, 4096, 41.0, 0.38) else p.round(t, 0, 0, 0.38);
        if (tripped == null and p.slowFor(t) >= 180) tripped = t;
    }
    // slow within seconds of the change, so a 3-minute rule trips in a little over 3 minutes
    try std.testing.expect(tripped != null and tripped.? - start < 200 * s);
    // the baseline stayed the healthy one
    try std.testing.expect(p.decode.base < 0.1);
}
