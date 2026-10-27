//! VP8's boolean decoder as libwebp 1.6.0 reads it (``utils/bit_reader_utils.c``, ``bit_reader_inl_utils.h``).
const std = @import("std");

/// One partition's reader: past its end one zero byte is shifted in and ``eof`` raised, then nothing more is loaded.
pub const BitReader = struct {
    value: u64 = 0,
    range: u32 = 255 - 1, // the range minus 1, in [127, 254]
    bits: i32 = -8, // valid bits left in ``value``
    buf: []const u8,
    pos: usize = 0,
    eof: bool = false,

    /// ``VP8InitBitReader``: the first bytes are loaded at once.
    pub fn init(data: []const u8) BitReader {
        var br: BitReader = .{ .buf = data };
        br.loadNewBytes();
        return br;
    }

    fn loadNewBytes(br: *BitReader) void {
        if (br.buf.len >= 8 and br.pos < br.buf.len - 7) { // ``buf < buf_max``: 7 bytes from one 8-byte load
            const in = std.mem.readInt(u64, br.buf[br.pos..][0..8], .big);
            br.pos += 7;
            br.value = (in >> 8) | (br.value << 56);
            br.bits += 56;
        } else br.loadFinalBytes();
    }

    fn loadFinalBytes(br: *BitReader) void {
        if (br.pos < br.buf.len) {
            br.bits += 8;
            br.value = br.buf[br.pos] | (br.value << 8);
            br.pos += 1;
        } else if (!br.eof) {
            br.value <<= 8;
            br.bits += 8;
            br.eof = true;
        } else br.bits = 0;
    }

    /// ``VP8GetBit``: one bit of probability ``prob`` / 256 of being 0.
    pub fn getBit(br: *BitReader, prob: u32) u1 {
        var range = br.range;
        if (br.bits < 0) br.loadNewBytes();
        const pos: u6 = @intCast(br.bits);
        const split = (range * prob) >> 8;
        const value: u32 = @truncate(br.value >> pos);
        const bit = value > split;
        if (bit) {
            range -= split;
            br.value -%= @as(u64, split + 1) << pos;
        } else range = split + 1;
        const shift: u5 = @intCast(7 ^ std.math.log2_int(u32, range));
        range <<= shift;
        br.bits -= shift;
        br.range = range - 1;
        return @intFromBool(bit);
    }

    /// ``VP8GetSigned``: ``v`` or ``-v`` on one even-probability bit.
    pub fn getSigned(br: *BitReader, v: i32) i32 {
        if (br.bits < 0) br.loadNewBytes();
        const pos: u6 = @intCast(br.bits);
        const split = br.range >> 1;
        const value: u32 = @truncate(br.value >> pos);
        const mask: i32 = @as(i32, @bitCast(split -% value)) >> 31;
        br.bits -= 1;
        br.range +%= @bitCast(mask);
        br.range |= 1;
        br.value -%= @as(u64, (split + 1) & @as(u32, @bitCast(mask))) << pos;
        return (v ^ mask) -% mask;
    }

    /// ``VP8GetValue``: ``n`` even-probability bits, most significant first.
    pub fn getValue(br: *BitReader, n: u5) u32 {
        var v: u32 = 0;
        var k = n;
        while (k > 0) {
            k -= 1;
            v |= @as(u32, br.getBit(0x80)) << k;
        }
        return v;
    }

    /// ``VP8GetSignedValue``: ``n`` bits of magnitude, then the sign.
    pub fn getSignedValue(br: *BitReader, n: u5) i32 {
        const v: i32 = @intCast(br.getValue(n));
        return if (br.getBit(0x80) == 1) -v else v;
    }
};

test "an all-zero partition reads zeros, and eof rises one byte past its end" {
    var br = BitReader.init(&.{ 0, 0 });
    try std.testing.expectEqual(@as(u32, 0), br.getValue(8));
    try std.testing.expect(!br.eof);
    _ = br.getValue(16);
    try std.testing.expect(br.eof);
    const zeros: [20]u8 = @splat(0);
    var big = BitReader.init(&zeros);
    for (0..100) |_| try std.testing.expectEqual(@as(u1, 0), big.getBit(200));
    try std.testing.expect(!big.eof);
}

test "a 0xff stream reads ones, and the signed value takes its sign bit" {
    const ones: [16]u8 = @splat(0xff);
    var br = BitReader.init(&ones);
    try std.testing.expectEqual(@as(u32, 0x7f), br.getValue(7));
    try std.testing.expectEqual(@as(i32, -3), br.getSignedValue(2));
    try std.testing.expectEqual(@as(i32, -5), br.getSigned(5));
}
