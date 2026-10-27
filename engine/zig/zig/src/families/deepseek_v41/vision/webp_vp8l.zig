//! Lossless WebP (VP8L) and lossless ALPH streams as libwebp 1.6.0 decodes them (``src/dec/vp8l_dec.c``), bit exact.
const std = @import("std");
const huff = @import("webp_vp8l_huff.zig");
const xform = @import("webp_vp8l_xform.zig");
const Allocator = std.mem.Allocator;
const BitReader = huff.BitReader;
const Code = huff.Code;
const Tables = huff.Tables;
const Transform = xform.Transform;

pub const Error = error{ BadWebp, TooManyPixels, TooLarge } || Allocator.Error;

/// The most a stream's groups and tables may take (``error.TooLarge`` past it); the pixel limit does not bound them.
pub const max_table_bytes: usize = 64 << 20;

/// A decoded image in libwebp's ARGB (``a << 24 | r << 16 | g << 8 | b``), ``w * h`` pixels row by row.
pub const Argb = struct {
    w: u32,
    h: u32,
    px: []u32,

    pub fn deinit(self: *Argb, gpa: Allocator) void {
        gpa.free(self.px);
        self.* = undefined;
    }
};

pub const len_code_limit = 256 + 24;
const green = 0;
const red = 1;
const blue = 2;
const alpha = 3;
const dist = 4;
const alphabet_size = [5]u32{ len_code_limit, 256, 256, 256, 40 };
pub const code_length_order = [19]u8{ 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
const packed_size = 64;
const special_marker = 0x100;

const Code32 = struct { bits: u32 = 0, value: u32 = 0 };

/// ``HTreeGroup``: five codes (tables in ``Tables``) and the trivial-literal / packed-table shortcuts.
const Group = struct {
    htrees: [5][]const Code = @splat(&.{}),
    is_trivial_literal: bool = false,
    literal_arb: u32 = 0,
    is_trivial_code: bool = false,
    use_packed_table: bool = false,
    packed_table: [packed_size]Code32 = @splat(.{}),
};

/// ``VP8LMetadata``: one image level's colour cache, meta Huffman image and codes.
const Meta = struct {
    cache: []u32 = &.{},
    cache_shift: u5 = 0,
    mask: u32 = 0,
    bits: u5 = 0,
    xsize: u32 = 0,
    image: []u32 = &.{},
    groups: []Group = &.{},
    tables: Tables = .{},

    fn deinit(m: *Meta, gpa: Allocator) void {
        gpa.free(m.cache);
        gpa.free(m.image);
        gpa.free(m.groups);
        m.tables.deinit(gpa);
        m.* = .{};
    }

    fn group(m: *const Meta, x: u32, y: u32) *const Group {
        if (m.bits == 0) return &m.groups[0];
        return &m.groups[m.image[@as(usize, m.xsize) * (y >> m.bits) + (x >> m.bits)]];
    }

    fn cacheInsert(m: *Meta, argb: u32) void {
        m.cache[(argb *% 0x1e35a7bd) >> m.cache_shift] = argb;
    }
};

const Decoder = struct {
    gpa: Allocator,
    br: BitReader,
    transforms: [4]Transform = @splat(.{}),
    next: usize = 0,
    seen: u4 = 0,
    width: u32 = 0,
    height: u32 = 0,
    /// Bytes of ``max_table_bytes`` not yet taken by groups or table capacity (never given back).
    table_budget: usize = max_table_bytes,

    fn deinit(d: *Decoder) void {
        for (d.transforms[0..d.next]) |t| d.gpa.free(t.data);
    }

    /// ``ReadHuffmanCodeLengths``: the alphabet's code lengths through the code-length code.
    fn readCodeLengths(d: *Decoder, cl: *const [19]u8, num_symbols: u32, lengths: []u8) Error!void {
        const br = &d.br;
        var table: [1 << huff.lengths_table_bits]Code = undefined;
        if (!huff.buildLengths(cl, &table)) return error.BadWebp;
        var max_symbol: u32 = num_symbols;
        if (br.readBits(1) != 0) {
            const nbits: u5 = @intCast(2 + 2 * br.readBits(3));
            max_symbol = 2 + br.readBits(nbits);
            if (max_symbol > num_symbols) return error.BadWebp;
        }
        var symbol: u32 = 0;
        var prev: u8 = 8;
        while (symbol < num_symbols) {
            if (max_symbol == 0) break;
            max_symbol -= 1;
            br.fill();
            const p = table[br.prefetch() & ((1 << huff.lengths_table_bits) - 1)];
            br.skip(p.bits);
            const code_len = p.value;
            if (code_len < 16) {
                lengths[symbol] = @intCast(code_len);
                symbol += 1;
                if (code_len != 0) prev = @intCast(code_len);
                continue;
            }
            const slot = code_len - 16;
            const extra = [3]u5{ 2, 3, 7 };
            const offset = [3]u32{ 3, 3, 11 };
            const repeat = br.readBits(extra[slot]) + offset[slot];
            if (symbol + repeat > num_symbols) return error.BadWebp;
            const len: u8 = if (code_len == 16) prev else 0;
            @memset(lengths[symbol..][0..repeat], len);
            symbol += repeat;
        }
    }

    /// ``ReadHuffmanCode``: one code (simple or length-coded), its table built in ``tables`` when given (else empty).
    fn readCode(d: *Decoder, size: u32, lengths: []u8, tables: ?*Tables) Error![]const Code {
        const br = &d.br;
        @memset(lengths[0..size], 0);
        if (br.readBits(1) != 0) {
            const num_symbols = br.readBits(1) + 1;
            const first_bits: u5 = if (br.readBits(1) == 0) 1 else 8;
            lengths[br.readBits(first_bits)] = 1;
            if (num_symbols == 2) lengths[br.readBits(8)] = 1;
        } else {
            var cl: [19]u8 = @splat(0);
            const num_codes = br.readBits(4) + 4;
            for (code_length_order[0..num_codes]) |o| cl[o] = @intCast(br.readBits(3));
            try d.readCodeLengths(&cl, size, lengths);
        }
        if (br.eos) return error.BadWebp;
        const n = huff.size(huff.table_bits, lengths[0..size]);
        if (n == 0) return error.BadWebp;
        const t = tables orelse return &.{};
        const table = try t.take(d.gpa, n, &d.table_budget);
        huff.fill(table, huff.table_bits, lengths[0..size]);
        return table;
    }

    /// ``ReadHuffmanCodesHelper``: every group's five codes; unused groups (``mapping`` -1) are only checked.
    fn readGroups(d: *Decoder, m: *Meta, cache_bits: u5, num_groups: u32, num_max: u32, mapping: ?[]i32) Error!void {
        const cache_size: u32 = if (cache_bits > 0) @as(u32, 1) << cache_bits else 0;
        const lengths = try d.gpa.alloc(u8, len_code_limit + cache_size);
        defer d.gpa.free(lengths);
        const group_bytes = @as(usize, num_groups) * @sizeOf(Group);
        if (group_bytes > d.table_budget) return error.TooLarge;
        d.table_budget -= group_bytes;
        m.groups = try d.gpa.alloc(Group, num_groups);
        @memset(m.groups, .{});
        for (0..num_max) |i| {
            if (mapping != null and mapping.?[i] == -1) {
                for (alphabet_size, 0..) |a, j| _ = try d.readCode(a + (if (j == 0) cache_size else 0), lengths, null);
                continue;
            }
            const g = &m.groups[if (mapping) |mp| @as(usize, @intCast(mp[i])) else i];
            var total_size: u32 = 0;
            var trivial_literal = true;
            var max_bits: u32 = 0;
            for (alphabet_size, 0..) |a, j| {
                const size = a + (if (j == 0) cache_size else 0);
                g.htrees[j] = try d.readCode(size, lengths, &m.tables);
                const first = g.htrees[j][0];
                if (trivial_literal and j >= red and j <= alpha) trivial_literal = first.bits == 0;
                total_size += first.bits;
                if (j <= alpha) max_bits += std.mem.max(u8, lengths[0..size]);
            }
            g.is_trivial_literal = trivial_literal;
            if (trivial_literal) {
                const r: u32 = g.htrees[red][0].value;
                const b: u32 = g.htrees[blue][0].value;
                const a: u32 = g.htrees[alpha][0].value;
                g.literal_arb = (a << 24) | (r << 16) | b;
                const gv: u32 = g.htrees[green][0].value;
                if (total_size == 0 and gv < 256) {
                    g.is_trivial_code = true;
                    g.literal_arb |= gv << 8;
                }
            }
            g.use_packed_table = !g.is_trivial_code and max_bits < 6;
            if (g.use_packed_table) buildPacked(g);
        }
    }

    /// ``ReadHuffmanCodes``: the meta Huffman image (level 0 only), then the groups it names.
    fn readCodes(d: *Decoder, m: *Meta, xsize: u32, ysize: u32, cache_bits: u5, level0: bool) Error!void {
        const br = &d.br;
        var num_groups: u32 = 1;
        var num_max: u32 = 1;
        var mapping: ?[]i32 = null;
        defer if (mapping) |mp| d.gpa.free(mp);
        if (level0 and br.readBits(1) != 0) {
            const prec: u5 = @intCast(2 + br.readBits(3));
            m.image = try d.decodeStream(xform.subSample(xsize, prec), xform.subSample(ysize, prec), null);
            m.bits = prec;
            for (m.image) |*p| {
                p.* = (p.* >> 8) & 0xffff;
                num_max = @max(num_max, p.* + 1);
            }
            if (num_max > 1000 or num_max > @as(u64, xsize) * ysize) {
                const mp = try d.gpa.alloc(i32, num_max);
                mapping = mp;
                @memset(mp, -1);
                num_groups = 0;
                for (m.image) |*p| {
                    if (mp[p.*] == -1) {
                        mp[p.*] = @intCast(num_groups);
                        num_groups += 1;
                    }
                    p.* = @intCast(mp[p.*]);
                }
            } else num_groups = num_max;
        }
        if (br.eos) return error.BadWebp;
        try d.readGroups(m, cache_bits, num_groups, num_max, mapping);
    }

    /// ``ReadTransform``: one transform and its sub-image; each kind at most once.
    fn readTransform(d: *Decoder, xsize: *u32, ysize: u32) Error!void {
        const kind: xform.Kind = @fromBackingInt(@intCast(@as(u2, @intCast(d.br.readBits(2)))));
        const bit = @as(u4, 1) << @backingInt(kind);
        if (d.seen & bit != 0) return error.BadWebp;
        d.seen |= bit;
        const t = &d.transforms[d.next];
        t.* = .{ .kind = kind, .xsize = xsize.*, .ysize = ysize };
        d.next += 1;
        switch (kind) {
            .predictor, .cross_color => {
                t.bits = @intCast(2 + d.br.readBits(3));
                t.data = try d.decodeStream(xform.subSample(t.xsize, t.bits), xform.subSample(ysize, t.bits), null);
            },
            .color_indexing => {
                const n = d.br.readBits(8) + 1;
                const bits: u5 = if (n > 16) 0 else if (n > 4) 1 else if (n > 2) 2 else 3;
                xsize.* = xform.subSample(t.xsize, bits);
                t.bits = bits;
                t.data = try d.decodeStream(n, 1, null);
                // ``ExpandColorMap``: deltas summed, padded with zeros to every bundled index
                const map = try d.gpa.alloc(u32, @as(usize, 1) << (@as(u5, 8) >> @intCast(bits)));
                @memset(map, 0);
                map[0] = t.data[0];
                for (1..n) |i| map[i] = xform.addPixels(t.data[i], map[i - 1]);
                d.gpa.free(t.data);
                t.data = map;
            },
            .subtract_green => {},
        }
    }

    /// ``DecodeImageStream``: level 0 (``top`` set) reads the transforms and keeps its codes in ``top``, else pixels.
    fn decodeStream(d: *Decoder, xsize: u32, ysize: u32, top: ?*Meta) Error![]u32 {
        const br = &d.br;
        var txs = xsize;
        if (top != null) {
            while (br.readBits(1) != 0) try d.readTransform(&txs, ysize);
        }
        var cache_bits: u5 = 0;
        if (br.readBits(1) != 0) {
            const cb = br.readBits(4);
            if (cb < 1 or cb > 11) return error.BadWebp;
            cache_bits = @intCast(cb);
        }
        var m: Meta = .{};
        errdefer m.deinit(d.gpa);
        try d.readCodes(&m, txs, ysize, cache_bits, top != null);
        if (cache_bits > 0) {
            m.cache = try d.gpa.alloc(u32, @as(usize, 1) << cache_bits);
            @memset(m.cache, 0);
            m.cache_shift = @intCast(32 - @as(u32, cache_bits));
        }
        m.xsize = xform.subSample(txs, m.bits);
        m.mask = if (m.bits == 0) ~@as(u32, 0) else (@as(u32, 1) << m.bits) - 1;
        d.width = txs;
        d.height = ysize;
        if (top) |t| {
            t.* = m;
            return &.{};
        }
        const data = try d.gpa.alloc(u32, @as(usize, txs) * ysize);
        errdefer d.gpa.free(data);
        try d.decodeData(&m, data, txs);
        if (br.eos) return error.BadWebp;
        m.deinit(d.gpa);
        return data;
    }

    /// ``DecodeImageData``, non-incremental: literals, LZ77 copies and colour-cache hits until ``data`` is full.
    fn decodeData(d: *Decoder, m: *Meta, data: []u32, width: u32) Error!void {
        const br = &d.br;
        var src: usize = 0;
        var last_cached: usize = 0;
        var col: u32 = 0;
        var row: u32 = 0;
        const cache_limit = len_code_limit + m.cache.len;
        var g: *const Group = if (data.len > 0) m.group(0, 0) else undefined;
        outer: while (src < data.len) {
            if (col & m.mask == 0) g = m.group(col, row);
            one: {
                if (g.is_trivial_code) {
                    data[src] = g.literal_arb;
                    break :one;
                }
                br.fill();
                var code: u32 = undefined;
                if (g.use_packed_table) {
                    code = readPacked(g, br, &data[src]);
                    if (br.isEnd()) break :outer;
                    if (code == 0) break :one;
                } else code = huff.readSymbol(g.htrees[green], br);
                if (br.isEnd()) break :outer;
                if (code < 256) {
                    if (g.is_trivial_literal) {
                        data[src] = g.literal_arb | (code << 8);
                    } else {
                        const r = huff.readSymbol(g.htrees[red], br);
                        br.fill();
                        const b = huff.readSymbol(g.htrees[blue], br);
                        const a = huff.readSymbol(g.htrees[alpha], br);
                        if (br.isEnd()) break :outer;
                        data[src] = (a << 24) | (r << 16) | (code << 8) | b;
                    }
                    break :one;
                } else if (code < len_code_limit) {
                    const length = huff.copyDistance(code - 256, br);
                    const dist_symbol = huff.readSymbol(g.htrees[dist], br);
                    br.fill();
                    const distance = huff.planeCodeToDistance(width, huff.copyDistance(dist_symbol, br));
                    if (br.isEnd()) break :outer;
                    if (src < distance or data.len - src < length) return error.BadWebp;
                    for (src..src + length) |i| data[i] = data[i - distance];
                    src += length;
                    col += length;
                    while (col >= width) {
                        col -= width;
                        row += 1;
                    }
                    if (col & m.mask != 0) g = m.group(col, row);
                    if (m.cache.len > 0) {
                        while (last_cached < src) : (last_cached += 1) m.cacheInsert(data[last_cached]);
                    }
                    continue :outer;
                } else if (code < cache_limit) {
                    while (last_cached < src) : (last_cached += 1) m.cacheInsert(data[last_cached]);
                    data[src] = m.cache[code - len_code_limit];
                    break :one;
                } else return error.BadWebp;
            }
            src += 1;
            col += 1;
            if (col >= width) {
                col = 0;
                row += 1;
                if (m.cache.len > 0) {
                    while (last_cached < src) : (last_cached += 1) m.cacheInsert(data[last_cached]);
                }
            }
        }
        br.eos = br.isEnd();
        if (br.eos) return error.BadWebp;
    }

    /// ``DecodeAlphaData``: 8-bit palette-only alpha; reading past the end on the last pixel passes.
    fn decodeAlphaData(d: *Decoder, m: *Meta, data: []u8, width: u32) Error!void {
        const br = &d.br;
        var pos: usize = 0;
        var col: u32 = 0;
        var row: u32 = 0;
        var g: *const Group = if (data.len > 0) m.group(0, 0) else undefined;
        while (!br.eos and pos < data.len) {
            if (col & m.mask == 0) g = m.group(col, row);
            br.fill();
            const code = huff.readSymbol(g.htrees[green], br);
            if (code < 256) {
                data[pos] = @intCast(code);
                pos += 1;
                col += 1;
                if (col >= width) {
                    col = 0;
                    row += 1;
                }
            } else if (code < len_code_limit) {
                const length = huff.copyDistance(code - 256, br);
                const dist_symbol = huff.readSymbol(g.htrees[dist], br);
                br.fill();
                const distance = huff.planeCodeToDistance(width, huff.copyDistance(dist_symbol, br));
                if (pos < distance or data.len - pos < length) return error.BadWebp;
                for (pos..pos + length) |i| data[i] = data[i - distance];
                pos += length;
                col += length;
                while (col >= width) {
                    col -= width;
                    row += 1;
                }
                if (pos < data.len and col & m.mask != 0) g = m.group(col, row);
            } else return error.BadWebp;
            br.eos = br.isEnd();
        }
        br.eos = br.isEnd();
        if (br.eos and pos < data.len) return error.BadWebp;
    }

    /// ``ApplyInverseTransforms`` over the whole image, last read first; returns the final ``xsize * ysize`` pixels.
    fn applyTransforms(d: *Decoder, decoded: []u32) Error![]u32 {
        var cur = decoded;
        errdefer d.gpa.free(cur);
        var n = d.next;
        while (n > 0) {
            n -= 1;
            const t = &d.transforms[n];
            switch (t.kind) {
                .subtract_green => xform.addGreen(cur),
                .predictor => xform.predictorInverse(t, cur),
                .cross_color => xform.crossColorInverse(t, cur),
                .color_indexing => if (t.bits == 0) xform.colorIndexInverse(t, cur, cur) else {
                    const out = try d.gpa.alloc(u32, @as(usize, t.xsize) * t.ysize);
                    xform.colorIndexInverse(t, cur, out);
                    d.gpa.free(cur);
                    cur = out;
                },
            }
        }
        return cur;
    }
};

/// ``ReadPackedSymbols``: a whole literal from 6 bits, or the green symbol of a non-literal (returned).
fn readPacked(g: *const Group, br: *BitReader, dst: *u32) u32 {
    const code = g.packed_table[br.prefetch() & (packed_size - 1)];
    if (code.bits < special_marker) {
        br.skip(code.bits);
        dst.* = code.value;
        return 0;
    }
    br.skip(code.bits - special_marker);
    return code.value;
}

fn buildPacked(g: *Group) void {
    for (&g.packed_table, 0..) |*h, code| {
        var bits: u32 = @intCast(code);
        const hcode = g.htrees[green][bits];
        if (hcode.value >= 256) {
            h.* = .{ .bits = @as(u32, hcode.bits) + special_marker, .value = hcode.value };
            continue;
        }
        h.* = .{};
        for ([4]usize{ green, red, blue, alpha }, [4]u5{ 8, 16, 0, 24 }) |tr, shift| {
            const c = g.htrees[tr][bits];
            h.bits += c.bits;
            h.value |= @as(u32, c.value) << shift;
            bits >>= @intCast(c.bits);
        }
    }
}

/// ``Is8bOptimizable``: no colour cache and single-symbol red, blue and alpha codes in every group.
fn is8bOptimizable(m: *const Meta) bool {
    if (m.cache.len > 0) return false;
    for (m.groups) |*g| {
        inline for (.{ red, blue, alpha }) |tr| if (g.htrees[tr][0].bits > 0) return false;
    }
    return true;
}

/// Decodes a VP8L chunk payload (from the 0x2f signature byte) to ARGB; ``w * h`` over ``max_pixels`` is refused.
pub fn decode(gpa: Allocator, data: []const u8, max_pixels: u64) Error!Argb {
    if (data.len < 5 or data[0] != 0x2f or data[4] >> 5 != 0) return error.BadWebp;
    var d: Decoder = .{ .gpa = gpa, .br = BitReader.init(data) };
    defer d.deinit();
    const br = &d.br;
    if (br.readBits(8) != 0x2f) return error.BadWebp;
    const w = br.readBits(14) + 1;
    const h = br.readBits(14) + 1;
    _ = br.readBits(1);
    if (br.readBits(3) != 0 or br.eos) return error.BadWebp;
    if (@as(u64, w) * h > max_pixels) return error.TooManyPixels;
    var m: Meta = .{};
    defer m.deinit(gpa);
    _ = try d.decodeStream(w, h, &m);
    const px = try gpa.alloc(u32, @as(usize, d.width) * d.height);
    d.decodeData(&m, px, d.width) catch |e| {
        gpa.free(px);
        return e;
    };
    return .{ .w = w, .h = h, .px = try d.applyTransforms(px) };
}

/// Decodes an ALPH chunk's lossless stream (after its header byte) of ``w x h``: each pixel's green, still filtered.
pub fn decodeAlpha(gpa: Allocator, data: []const u8, w: u32, h: u32, out: []u8) Error!void {
    if (w == 0 or h == 0 or out.len != @as(u64, w) * h) return error.BadWebp;
    var d: Decoder = .{ .gpa = gpa, .br = BitReader.init(data) };
    defer d.deinit();
    var m: Meta = .{};
    defer m.deinit(gpa);
    _ = try d.decodeStream(w, h, &m);
    const n = @as(usize, d.width) * d.height;
    if (d.next == 1 and d.transforms[0].kind == .color_indexing and is8bOptimizable(&m)) {
        const idx = try gpa.alloc(u8, n);
        defer gpa.free(idx);
        try d.decodeAlphaData(&m, idx, d.width);
        xform.colorIndexAlpha(&d.transforms[0], idx, out);
        return;
    }
    const px = try gpa.alloc(u32, n);
    d.decodeData(&m, px, d.width) catch |e| {
        gpa.free(px);
        return e;
    };
    const argb = try d.applyTransforms(px);
    defer gpa.free(argb);
    for (argb, out) |p, *o| o.* = @truncate(p >> 8);
}

test {
    std.testing.refAllDecls(@This());
    _ = huff;
    _ = xform;
}
fn hex(comptime s: []const u8) [s.len / 2]u8 {
    var out: [s.len / 2]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch unreachable;
    return out;
}

test "Pillow's lossless encoder: 5x3 RGBA, exact" {
    const data = hex("2f048000105fa0a66d23281d1f105d8f3fbfe3412660814f13e9f4ff66a66d9be62c8016c3f883199b698a62dba0aedf5206" ++
        "5b211550435181047a80574006209e9bd45ba76ed35bd3df83aac836a8fbf02787022289a081381a47f43fe6b216df01");
    const want = [_]u32{
        0xffff0000, 0x8000ff00, 0x000000ff, 0x280a141e, 0xffc86432, 0x04010203, 0xffff0000, 0xffff0000,
        0x09090909, 0x00000000, 0x50323c46, 0x5d5a5b5c, 0xffff0000, 0xff010101, 0xff020202,
    };
    var img = try decode(std.testing.allocator, &data, 1 << 20);
    defer img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 5), img.w);
    try std.testing.expectEqualSlices(u32, &want, img.px);
    try std.testing.expectError(error.TooManyPixels, decode(std.testing.allocator, &data, 14));
    var n: usize = 6;
    while (n < data.len) : (n += 2) try std.testing.expectError(error.BadWebp, decode(std.testing.allocator, data[0..n], 1 << 20));
}

test "all four transforms (predictor, subtract green, colour indexing, cross colour), cache, meta codes" {
    const data = hex("2f06800010f97173ede6b8663fd1fbfdf76ff9f3ef9e9e9e4b2d00000098ceffa0ffb2b7ca412200e2be20ab1f18ac528516" ++
        "f39ff675799ffbff5ecbe1fdff397784e9aac13bdd3fee7905116130718e23ffcf611cc788c2767a7ee5fb1fc546771192a7" ++
        "76ee77eb753b348e12a9ef7fc0ff3f69b129af9ae40ad2b28e9f0023f58bb36dbbc1e4fcffefdf7fecfd6d57d7d5c7e78337" ++
        "8a281440e1f7670b4c8bc3f64872c8338bc8fdf5ffff2e6fbd86f3fd9fa89f7dfff7dfffdb7ffffdf7ffd9fe23994a1537fd" ++
        "6fcfc1ff7fbef7315a1cd10866bc8ff5fdcedddabfc95f47261211e46f76fbd97777fdaf880422c3847643d0038e6edbd9cd" ++
        "9b9f0e");
    const want = [_]u32{
        0xb8e99ab3, 0x71d23466, 0x2abbce19, 0xe3a468cc, 0x56a371c8, 0xf4ccba6c, 0xadb5541f, 0x2be8a3af,
        0xa4fbefac, 0x1bc89913, 0x34d8f4d0, 0x73c8e0c8, 0x6c0196b9, 0x110ced1f, 0xe4d13d62, 0xb8ce2a26,
        0x059cb4b3, 0xd59c8bb2, 0x95883067, 0x3febbf27, 0xbfa190b3,
    };
    var img = try decode(std.testing.allocator, &data, 1 << 20);
    defer img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 7), img.w);
    try std.testing.expectEqualSlices(u32, &want, img.px);
}

test "alpha, 8-bit path: libwebp's output, also with the last byte cut" {
    const data = hex("178046e7bd7c97a96bffcefabf82e5090d8bdf5cbeec2a3161f1a770ae7f293904b2c944ef194450008ac8892e74fff68840" ++
        "38449c6affb1cff7a20ec5de8de07b9904da94e5f9ceefd753ecef61ee0b9ddfefafefff37387f78a689028410b9b8db45cd" ++
        "2b5ed7f3fdfffe00");
    const want = [_]u8{ 197, 197, 0, 234, 197, 0, 159, 197, 0, 197, 0, 0, 234, 234, 234, 159, 197, 234, 234, 234, 197, 0, 159, 234, 159, 234, 159 };
    var out: [want.len]u8 = undefined;
    try decodeAlpha(std.testing.allocator, &data, 9, 3, &out);
    try std.testing.expectEqualSlices(u8, &want, &out);
    try decodeAlpha(std.testing.allocator, data[0 .. data.len - 1], 9, 3, &out);
    try std.testing.expectEqualSlices(u8, &want, &out);
    try std.testing.expectError(error.BadWebp, decodeAlpha(std.testing.allocator, data[0 .. data.len / 2], 9, 3, &out));
}

test "refused: a transform twice, colour-cache bits out of 1..11, a bad signature or version" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2f010000002d0000000000000000"), 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2f01000000320000000000000000"), 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2f01000000020000000000000000"), 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2e01000000020000000000000000"), 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2f01000020020000000000000000"), 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, &hex("2f010000"), 1 << 20));
}
