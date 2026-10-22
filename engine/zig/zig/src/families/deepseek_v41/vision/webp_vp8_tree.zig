//! VP8 macroblock syntax as libwebp 1.6.0 parses it: ``tree_dec.c`` (probabilities, modes), ``vp8_dec.c`` residuals.
const std = @import("std");
const tables = @import("webp_vp8_tables.zig");
const BitReader = @import("webp_vp8_bits.zig").BitReader;
const dsp = @import("webp_vp8_dsp.zig");

/// ``VP8QuantMatrix``: the [DC, AC] dequantization factors of one segment.
pub const Quant = struct { y1: [2]i32 = .{ 0, 0 }, y2: [2]i32 = .{ 0, 0 }, uv: [2]i32 = .{ 0, 0 } };

/// ``VP8MB``: a macroblock's non-zero context (4 luma bits, 4 chroma bits) and its luma DC flag.
pub const Nz = struct { nz: u8 = 0, nz_dc: u8 = 0 };

/// ``VP8MBData``: one macroblock's parsed modes and dequantized coefficients.
pub const MbData = struct {
    coeffs: [384]i16 = @splat(0),
    is_i4x4: bool = false,
    imodes: [16]u8 = @splat(0),
    uvmode: u8 = 0,
    non_zero_y: u32 = 0,
    non_zero_uv: u32 = 0,
    skip: bool = false,
    segment: u2 = 0,
};

/// The frame-wide parse state: probabilities, segment map flag, skip probability, mode and non-zero contexts.
pub const Parser = struct {
    segments: [3]u8 = .{ 255, 255, 255 },
    bands: [4 * 8 * 3 * 11]u8 = undefined,
    update_map: bool = false,
    use_skip_proba: bool = false,
    skip_p: u8 = 0,
    dqm: [4]Quant = @splat(.{}),
    intra_t: []u8, // 4 a macroblock column
    intra_l: [4]u8 = @splat(0),
    nz_info: []Nz, // the left context first, then one a column

    /// ``VP8ParseProba``: coefficient probability updates, then the skip probability.
    pub fn parseProba(ps: *Parser, br: *BitReader) void {
        for (&ps.bands, tables.coeffs_update_proba, tables.coeffs_proba0) |*p, up, def| {
            p.* = if (br.getBit(up) == 1) @intCast(br.getValue(8)) else def;
        }
        ps.use_skip_proba = br.getBit(0x80) == 1;
        if (ps.use_skip_proba) ps.skip_p = @intCast(br.getValue(8));
    }

    /// ``VP8InitScanline``: the left contexts reset at each macroblock row.
    pub fn initScanline(ps: *Parser) void {
        ps.nz_info[0] = .{};
        ps.intra_l = @splat(0);
    }

    /// ``ParseIntraMode``: segment, skip flag, luma and chroma modes of macroblock ``mb_x``.
    pub fn parseIntraMode(ps: *Parser, br: *BitReader, block: *MbData, mb_x: usize) void {
        const top = ps.intra_t[4 * mb_x ..][0..4];
        const left = &ps.intra_l;
        if (ps.update_map) {
            block.segment = if (br.getBit(ps.segments[0]) == 0)
                br.getBit(ps.segments[1])
            else
                @as(u2, br.getBit(ps.segments[2])) + 2;
        } else block.segment = 0;
        if (ps.use_skip_proba) block.skip = br.getBit(ps.skip_p) == 1;
        block.is_i4x4 = br.getBit(145) == 0;
        if (!block.is_i4x4) {
            const ymode: u8 = if (br.getBit(156) == 1)
                (if (br.getBit(128) == 1) 1 else 3)
            else
                (if (br.getBit(163) == 1) 2 else 0);
            block.imodes[0] = ymode;
            @memset(top, ymode);
            @memset(left, ymode);
        } else {
            for (0..4) |y| {
                var ymode: u8 = left[y];
                for (0..4) |x| {
                    const prob = tables.bmodes_proba[(@as(usize, top[x]) * 10 + ymode) * 9 ..][0..9];
                    var i: i8 = ymodes_intra4[br.getBit(prob[0])];
                    while (i > 0) {
                        const k: usize = @intCast(i);
                        i = ymodes_intra4[2 * k + br.getBit(prob[k])];
                    }
                    ymode = @intCast(-i);
                    top[x] = ymode;
                }
                @memcpy(block.imodes[4 * y ..][0..4], top);
                left[y] = ymode;
            }
        }
        block.uvmode = if (br.getBit(142) == 0) 0 else if (br.getBit(114) == 0) 2 else if (br.getBit(183) == 1) 1 else 3;
    }

    fn proba(ps: *const Parser, t: usize, n: usize, ctx: usize) []const u8 {
        return ps.bands[((t * 8 + tables.bands[n]) * 3 + ctx) * 11 ..][0..11];
    }

    /// ``GetCoeffsFast``: one block's tokens of type ``t`` from position ``n0``; the last non-zero position plus one.
    fn getCoeffs(ps: *const Parser, br: *BitReader, t: usize, ctx: usize, dq: [2]i32, n0: usize, out: []i16) usize {
        var n = n0;
        var p = ps.proba(t, n, ctx);
        while (n < 16) : (n += 1) {
            if (br.getBit(p[0]) == 0) return n;
            while (br.getBit(p[1]) == 0) {
                n += 1;
                if (n == 16) return 16;
                p = ps.proba(t, n, 0);
            }
            var v: i32 = 1;
            if (br.getBit(p[2]) == 0) {
                p = ps.proba(t, n + 1, 1);
            } else {
                v = largeValue(br, p);
                p = ps.proba(t, n + 1, 2);
            }
            out[zigzag[n]] = @truncate(br.getSigned(v) *% dq[@intFromBool(n > 0)]);
        }
        return 16;
    }

    /// ``ParseResiduals``: every coefficient of the macroblock at column ``mb_x``; true when all are zero.
    pub fn parseResiduals(ps: *Parser, br: *BitReader, block: *MbData, mb_x: usize) bool {
        const mb = &ps.nz_info[1 + mb_x];
        const left = &ps.nz_info[0];
        const q = ps.dqm[block.segment];
        const dst = &block.coeffs;
        @memset(dst, 0);
        var first: usize = 0;
        var ac_type: usize = 3;
        if (!block.is_i4x4) {
            var dc: [16]i16 = @splat(0);
            const ctx = @as(usize, mb.nz_dc) + left.nz_dc;
            const nz = ps.getCoeffs(br, 1, ctx, q.y2, 0, &dc);
            mb.nz_dc = @intFromBool(nz > 0);
            left.nz_dc = mb.nz_dc;
            if (nz > 1) {
                dsp.transformWht(&dc, dst);
            } else {
                const dc0: i16 = @intCast((@as(i32, dc[0]) + 3) >> 3);
                var i: usize = 0;
                while (i < 256) : (i += 16) dst[i] = dc0;
            }
            first = 1;
            ac_type = 0;
        }
        var tnz: u32 = mb.nz & 0x0f;
        var lnz: u32 = left.nz & 0x0f;
        var non_zero_y: u32 = 0;
        var at: usize = 0;
        for (0..4) |_| {
            var l: u32 = lnz & 1;
            var nz_coeffs: u32 = 0;
            for (0..4) |_| {
                const ctx = l + (tnz & 1);
                const nz = ps.getCoeffs(br, ac_type, ctx, q.y1, first, dst[at..][0..16]);
                l = @intFromBool(nz > first);
                tnz = (tnz >> 1) | (l << 7);
                nz_coeffs = nzCodeBits(nz_coeffs, nz, dst[at] != 0);
                at += 16;
            }
            tnz >>= 4;
            lnz = (lnz >> 1) | (l << 7);
            non_zero_y = (non_zero_y << 8) | nz_coeffs;
        }
        var out_t_nz: u32 = tnz;
        var out_l_nz: u32 = lnz >> 4;
        var non_zero_uv: u32 = 0;
        var ch: u5 = 0;
        while (ch < 4) : (ch += 2) {
            var nz_coeffs: u32 = 0;
            tnz = mb.nz >> (4 + @as(u3, @intCast(ch)));
            lnz = left.nz >> (4 + @as(u3, @intCast(ch)));
            for (0..2) |_| {
                var l: u32 = lnz & 1;
                for (0..2) |_| {
                    const ctx = l + (tnz & 1);
                    const nz = ps.getCoeffs(br, 2, ctx, q.uv, 0, dst[at..][0..16]);
                    l = @intFromBool(nz > 0);
                    tnz = (tnz >> 1) | (l << 3);
                    nz_coeffs = nzCodeBits(nz_coeffs, nz, dst[at] != 0);
                    at += 16;
                }
                tnz >>= 2;
                lnz = (lnz >> 1) | (l << 5);
            }
            non_zero_uv |= nz_coeffs << (4 * ch);
            out_t_nz |= (tnz << 4) << ch;
            out_l_nz |= (lnz & 0xf0) << ch;
        }
        mb.nz = @truncate(out_t_nz);
        left.nz = @truncate(out_l_nz);
        block.non_zero_y = non_zero_y;
        block.non_zero_uv = non_zero_uv;
        return (non_zero_y | non_zero_uv) == 0;
    }
};

const ymodes_intra4 = [18]i8{ 0, 1, -1, 2, -2, 3, 4, 6, -3, 5, -4, -5, -6, 7, -7, 8, -8, -9 };

const zigzag = [16]u8{ 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15 };

const cat3456 = [4][]const u8{
    &.{ 173, 148, 140 },
    &.{ 176, 155, 140, 135 },
    &.{ 180, 157, 141, 134, 130 },
    &.{ 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129 },
};

fn nzCodeBits(nz_coeffs: u32, nz: usize, dc_nz: bool) u32 {
    const code: u32 = if (nz > 3) 3 else if (nz > 1) 2 else @intFromBool(dc_nz);
    return (nz_coeffs << 2) | code;
}

/// ``GetLargeValue``: a coefficient magnitude of at least 2 (RFC 6386 section 13.2).
fn largeValue(br: *BitReader, p: []const u8) i32 {
    if (br.getBit(p[3]) == 0) {
        if (br.getBit(p[4]) == 0) return 2;
        return 3 + @as(i32, br.getBit(p[5]));
    }
    if (br.getBit(p[6]) == 0) {
        if (br.getBit(p[7]) == 0) return 5 + @as(i32, br.getBit(159));
        const v = 7 + 2 * @as(i32, br.getBit(165));
        return v + br.getBit(145);
    }
    const bit1 = br.getBit(p[8]);
    const bit0 = br.getBit(p[9 + @as(usize, bit1)]);
    const cat = 2 * @as(u3, bit1) + bit0;
    var v: i32 = 0;
    for (cat3456[cat]) |pr| v += v + br.getBit(pr);
    return v + 3 + (@as(i32, 8) << cat);
}

test "an all-zero token partition parses as skipped DC-only macroblocks" {
    var intra_t: [4]u8 = @splat(0);
    var nz_info: [2]Nz = @splat(.{});
    var ps: Parser = .{ .intra_t = &intra_t, .nz_info = &nz_info };
    ps.bands = tables.coeffs_proba0;
    const zeros: [32]u8 = @splat(0);
    var br = BitReader.init(&zeros);
    var block: MbData = .{};
    ps.parseIntraMode(&br, &block, 0);
    try std.testing.expect(block.is_i4x4);
    try std.testing.expectEqual(@as(u8, 0), block.imodes[15]);
    try std.testing.expect(ps.parseResiduals(&br, &block, 0));
    try std.testing.expectEqual(@as(u32, 0), block.non_zero_y);
}
