//! VP8L bit reader, Huffman tables, LZ77 prefix codes: libwebp 1.6.0 bit_reader_utils.c, huffman_utils.c, vp8l_dec.c.
const std = @import("std");

/// ``VP8LBitReader``: a 64-bit window, ``eos`` set only once a bit past the last byte was asked for.
pub const BitReader = struct {
    val: u64 = 0,
    buf: []const u8,
    pos: usize,
    bit_pos: u32 = 0,
    eos: bool = false,

    pub fn init(buf: []const u8) BitReader {
        const n = @min(buf.len, 8);
        var v: u64 = 0;
        for (buf[0..n], 0..) |b, i| v |= @as(u64, b) << @intCast(8 * i);
        return .{ .val = v, .buf = buf, .pos = n };
    }

    /// ``VP8LPrefetchBits``: the window from the current bit (the shift wraps at 64, as libwebp's mask does).
    pub fn prefetch(br: *const BitReader) u32 {
        return @truncate(br.val >> @as(u6, @truncate(br.bit_pos)));
    }

    /// ``VP8LIsEndOfStream``: a bit past the end was read (does not set ``eos``).
    pub fn isEnd(br: *const BitReader) bool {
        return br.eos or (br.pos == br.buf.len and br.bit_pos > 64);
    }

    fn setEnd(br: *BitReader) void {
        br.eos = true;
        br.bit_pos = 0;
    }

    fn shiftBytes(br: *BitReader) void {
        while (br.bit_pos >= 8 and br.pos < br.buf.len) {
            br.val >>= 8;
            br.val |= @as(u64, br.buf[br.pos]) << 56;
            br.pos += 1;
            br.bit_pos -= 8;
        }
        if (br.isEnd()) br.setEnd();
    }

    /// ``VP8LFillBitWindow`` with the x86 fast load: at least 32 bits ready after it.
    pub fn fill(br: *BitReader) void {
        if (br.bit_pos < 32) return;
        if (br.pos + 8 < br.buf.len) {
            br.val >>= 32;
            br.bit_pos -= 32;
            br.val |= @as(u64, std.mem.readInt(u32, br.buf[br.pos..][0..4], .little)) << 32;
            br.pos += 4;
            return;
        }
        br.shiftBytes();
    }

    /// ``VP8LReadBits``: ``n`` (at most 24) bits, or 0 with ``eos`` set once past the end.
    pub fn readBits(br: *BitReader, n: u5) u32 {
        if (!br.eos and n <= 24) {
            const v = br.prefetch() & ((@as(u32, 1) << n) - 1);
            br.bit_pos += n;
            br.shiftBytes();
            return v;
        }
        br.setEnd();
        return 0;
    }

    /// ``VP8LSetBitPos`` by a step: no refill, no end check.
    pub fn skip(br: *BitReader, n: u32) void {
        br.bit_pos += n;
    }
};

/// ``HuffmanCode``: a symbol and its length, or in a root entry the 2nd-level table's width and offset.
pub const Code = struct { bits: u8 = 0, value: u16 = 0 };

pub const table_bits = 8;
pub const lengths_table_bits = 7;
const max_length = 15;
/// The largest alphabet: 256 literals, 24 lengths, a 2048-entry colour cache.
pub const max_alphabet = 256 + 24 + (1 << 11);

/// ``ReadSymbol``: one symbol from a table built with 8 root bits (the caller keeps the window filled).
pub fn readSymbol(table: []const Code, br: *BitReader) u32 {
    var val = br.prefetch();
    var i: usize = val & 0xff;
    const nbits = @as(i32, table[i].bits) - table_bits;
    if (nbits > 0) {
        br.skip(table_bits);
        val = br.prefetch();
        i += table[i].value;
        i += val & ((@as(u32, 1) << @intCast(nbits)) - 1);
    }
    br.skip(table[i].bits);
    return table[i].value;
}

const code_to_plane = [120]u8{
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b,
    0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
    0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d,
    0x68, 0x02, 0x67, 0x69, 0x12, 0x1e, 0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
    0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
    0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
    0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
};

/// ``GetCopyDistance`` (lengths use the same prefix code).
pub fn copyDistance(symbol: u32, br: *BitReader) u32 {
    if (symbol < 4) return symbol + 1;
    const extra: u5 = @intCast((symbol - 2) >> 1);
    const offset = (2 + (symbol & 1)) << extra;
    return offset + br.readBits(extra) + 1;
}

/// ``PlaneCodeToDistance``: the first 120 codes are 2-D neighbourhood offsets.
pub fn planeCodeToDistance(xsize: u32, plane_code: u32) u32 {
    if (plane_code > 120) return plane_code - 120;
    const dist_code = code_to_plane[plane_code - 1];
    const yoffset: i64 = dist_code >> 4;
    const xoffset: i64 = 8 - @as(i64, dist_code & 0xf);
    const distance = yoffset * xsize + xoffset;
    return if (distance >= 1) @intCast(distance) else 1;
}

/// ``GetNextKey``: reverse(reverse(key, len) + 1, len).
fn nextKey(key: u32, len: u32) u32 {
    var step = @as(u32, 1) << @intCast(len - 1);
    while (key & step != 0) step >>= 1;
    return if (step != 0) (key & (step - 1)) + step else key;
}

fn replicate(table: []Code, step: u32, end_in: u32, code: Code) void {
    var end = end_in;
    while (true) {
        end -= step;
        table[end] = code;
        if (end == 0) break;
    }
}

/// ``NextTableBitSize``: the width of the next 2nd-level table.
fn nextTableBits(count: *const [max_length + 1]u32, len_in: u32, root_bits: u32) u32 {
    var len = len_in;
    var left: i32 = @as(i32, 1) << @intCast(len - root_bits);
    while (len < max_length) {
        left -= @intCast(count[len]);
        if (left <= 0) break;
        len += 1;
        left <<= 1;
    }
    return len - root_bits;
}

/// ``BuildHuffmanTable``: the table's total size, 0 for a bad code; ``root`` null only sizes and checks it.
fn buildTable(root: ?[]Code, root_bits: u32, lengths: []const u8, sorted: []u16) u32 {
    var total: u32 = @as(u32, 1) << @intCast(root_bits);
    var count: [max_length + 1]u32 = @splat(0);
    var offset: [max_length + 1]u32 = @splat(0);
    for (lengths) |l| {
        if (l > max_length) return 0;
        count[l] += 1;
    }
    if (count[0] == lengths.len) return 0;
    offset[1] = 0;
    for (1..max_length) |len| {
        if (count[len] > (@as(u32, 1) << @intCast(len))) return 0;
        offset[len + 1] = offset[len] + count[len];
    }
    for (lengths, 0..) |l, sym| {
        if (l == 0) continue;
        if (root != null) {
            if (offset[l] >= lengths.len) return 0;
            sorted[offset[l]] = @intCast(sym);
        }
        offset[l] += 1;
    }
    if (offset[max_length] == 1) { // one symbol: a zero-length code everywhere
        if (root) |t| replicate(t, 1, total, .{ .bits = 0, .value = sorted[0] });
        return total;
    }
    var at: usize = 0; // the current table's start in ``root``
    var low: u32 = 0xffffffff;
    const mask = total - 1;
    var key: u32 = 0;
    var num_nodes: i32 = 1;
    var num_open: i32 = 1;
    var tbits = root_bits;
    var tsize: u32 = @as(u32, 1) << @intCast(tbits);
    var sym: usize = 0;
    var len: u32 = 1;
    var step: u32 = 2;
    while (len <= root_bits) : ({
        len += 1;
        step <<= 1;
    }) {
        num_open <<= 1;
        num_nodes += num_open;
        num_open -= @intCast(count[len]);
        if (num_open < 0) return 0;
        const t = root orelse continue;
        while (count[len] > 0) : (count[len] -= 1) {
            replicate(t[key..], step, tsize, .{ .bits = @intCast(len), .value = sorted[sym] });
            sym += 1;
            key = nextKey(key, len);
        }
    }
    len = root_bits + 1;
    step = 2;
    while (len <= max_length) : ({
        len += 1;
        step <<= 1;
    }) {
        num_open <<= 1;
        num_nodes += num_open;
        num_open -= @intCast(count[len]);
        if (num_open < 0) return 0;
        while (count[len] > 0) : (count[len] -= 1) {
            if ((key & mask) != low) {
                at += tsize;
                tbits = nextTableBits(&count, len, root_bits);
                tsize = @as(u32, 1) << @intCast(tbits);
                total += tsize;
                low = key & mask;
                if (root) |t| t[low] = .{ .bits = @intCast(tbits + root_bits), .value = @intCast(at - low) };
            }
            if (root) |t| {
                const code: Code = .{ .bits = @intCast(len - root_bits), .value = sorted[sym] };
                replicate(t[at + (key >> @intCast(root_bits)) ..], step, tsize, code);
                sym += 1;
            }
            key = nextKey(key, len);
        }
    }
    if (num_nodes != 2 * @as(i32, @intCast(offset[max_length])) - 1) return 0;
    return total;
}

/// ``VP8LBuildHuffmanTable``'s sizing pass: the code's table size, 0 for a bad code.
pub fn size(root_bits: u32, lengths: []const u8) u32 {
    var sorted: [max_alphabet]u16 = undefined;
    return buildTable(null, root_bits, lengths, &sorted);
}

/// ``VP8LBuildHuffmanTable`` into ``table``, of the ``size`` the same code gave.
pub fn fill(table: []Code, root_bits: u32, lengths: []const u8) void {
    var sorted: [max_alphabet]u16 = undefined;
    @memset(table, .{});
    std.debug.assert(buildTable(table, root_bits, lengths, &sorted) == table.len);
}

/// ``size`` then ``fill`` into a new list, for tests.
fn buildList(gpa: std.mem.Allocator, list: *std.ArrayList(Code), root_bits: u32, lengths: []const u8) !u32 {
    const n = size(root_bits, lengths);
    if (n == 0) return 0;
    list.clearRetainingCapacity();
    fill(try list.addManyAsSlice(gpa, n), root_bits, lengths);
    return n;
}

/// ``HuffmanTables``: segments that never move, so growing never copies; each doubles the total, up to 256 KiB.
pub const Tables = struct {
    segments: std.ArrayList([]Code) = .empty,
    used: usize = 0,
    total: usize = 0,

    const segment_entries = 1 << 16;

    pub fn deinit(t: *Tables, gpa: std.mem.Allocator) void {
        for (t.segments.items) |s| gpa.free(s);
        t.segments.deinit(gpa);
    }

    /// ``n`` contiguous entries: the rest of the last segment, else a new one taken from ``budget`` bytes.
    pub fn take(t: *Tables, gpa: std.mem.Allocator, n: usize, budget: *usize) error{ TooLarge, OutOfMemory }![]Code {
        if (t.segments.items.len > 0) {
            const last = t.segments.items[t.segments.items.len - 1];
            if (last.len - t.used >= n) {
                defer t.used += n;
                return last[t.used..][0..n];
            }
        }
        const len = @min(@max(n, @min(t.total, segment_entries)), budget.* / @sizeOf(Code));
        if (len < n) return error.TooLarge;
        try t.segments.ensureUnusedCapacity(gpa, 1);
        const seg = try gpa.alloc(Code, len);
        budget.* -= len * @sizeOf(Code);
        t.segments.appendAssumeCapacity(seg);
        t.total += len;
        t.used = n;
        return seg[0..n];
    }
};

/// The 7-bit table of the code-length code (lengths 0..7, so it never needs a 2nd level); false for a bad code.
pub fn buildLengths(lengths: *const [19]u8, out: *[1 << lengths_table_bits]Code) bool {
    var sorted: [19]u16 = undefined;
    const total = buildTable(null, lengths_table_bits, lengths, &sorted);
    if (total != out.len) return false;
    @memset(out, .{});
    _ = buildTable(out, lengths_table_bits, lengths, &sorted);
    return true;
}

test "one-symbol and two-symbol codes" {
    var lens: [5]u8 = .{ 0, 0, 1, 0, 0 };
    var list: std.ArrayList(Code) = .empty;
    defer list.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 256), try buildList(std.testing.allocator, &list, 8, &lens));
    try std.testing.expectEqual(@as(u16, 2), list.items[17].value);
    try std.testing.expectEqual(@as(u8, 0), list.items[17].bits);
    lens = .{ 1, 0, 0, 0, 1 };
    _ = try buildList(std.testing.allocator, &list, 8, &lens);
    try std.testing.expectEqual(@as(u16, 0), list.items[0].value);
    try std.testing.expectEqual(@as(u16, 4), list.items[1].value);
    lens = .{ 1, 1, 1, 0, 0 }; // over-subscribed
    try std.testing.expectEqual(@as(u32, 0), size(8, &lens));
    lens = .{ 2, 2, 2, 0, 0 }; // incomplete
    try std.testing.expectEqual(@as(u32, 0), size(8, &lens));
}

test "two-level table: 9-bit codes behind the 8-bit root" {
    var lens: [258]u8 = @splat(8);
    lens[255] = 9;
    lens[256] = 9;
    lens[257] = 0;
    lens[0] = 8;
    var list: std.ArrayList(Code) = .empty;
    defer list.deinit(std.testing.allocator);
    // 255 codes of 8 bits + 2 of 9 bits fill the code space exactly
    try std.testing.expect(try buildList(std.testing.allocator, &list, 8, &lens) > 256);
    const data = [_]u8{ 0xff, 0x01, 0, 0, 0, 0, 0, 0, 0, 0 };
    var br = BitReader.init(&data);
    br.fill();
    try std.testing.expectEqual(@as(u32, 256), readSymbol(list.items, &br));
    try std.testing.expectEqual(@as(u32, 9), br.bit_pos);
}

test "bit reader: eos only past the last byte" {
    const data = [_]u8{ 0xa5, 0x0f };
    var br = BitReader.init(&data);
    try std.testing.expectEqual(@as(u32, 0x5), br.readBits(4));
    try std.testing.expectEqual(@as(u32, 0xfa), br.readBits(8));
    _ = br.readBits(24);
    try std.testing.expect(!br.isEnd());
    _ = br.readBits(24);
    _ = br.readBits(5);
    try std.testing.expect(br.eos);
    try std.testing.expectEqual(@as(u32, 0), br.readBits(1));
}
