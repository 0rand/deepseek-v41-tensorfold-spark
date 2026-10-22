//! Lossy WebP (VP8 key frame) as libwebp 1.6.0 decodes it (``vp8_dec.c``, ``quant_dec.c``, ``frame_dec.c``) to Y, U, V.
const std = @import("std");
const BitReader = @import("webp_vp8_bits.zig").BitReader;
const tree = @import("webp_vp8_tree.zig");
const tables = @import("webp_vp8_tables.zig");
const dsp = @import("webp_vp8_dsp.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ BadWebp, TooManyPixels } || Allocator.Error;

/// The planes ``WebPDecodeYUV`` returns: Y with stride ``w``, U and V with stride ``uv_w`` and ``(h + 1) / 2`` rows.
pub const Yuv = struct {
    w: u32,
    h: u32,
    y: []u8,
    u: []u8,
    v: []u8,
    uv_w: u32,

    pub fn deinit(self: *Yuv, gpa: Allocator) void {
        gpa.free(self.y);
        gpa.free(self.u);
        gpa.free(self.v);
        self.* = undefined;
    }
};

const bps = dsp.bps;
const y_off = bps + 8; // ``Y_OFF`` .. ``V_OFF``: the blocks in the work buffer, a row and a column of context each
const u_off = y_off + bps * 16 + bps;
const v_off = u_off + 16;

/// ``VP8FInfo``: one macroblock's loop filter limits.
const FInfo = struct { limit: u8 = 0, ilevel: u8 = 0, inner: bool = false, hev_thresh: u8 = 0 };

/// ``VP8SegmentHeader`` and ``VP8FilterHeader``, the parts of the frame header read in partition 0.
const Header = struct {
    use_segment: bool = false,
    absolute_delta: bool = true,
    quantizer: [4]i32 = @splat(0),
    filter_strength: [4]i32 = @splat(0),
    simple: bool = false,
    level: i32 = 0,
    sharpness: i32 = 0,
    use_lf_delta: bool = false,
    ref_lf_delta: [4]i32 = @splat(0),
    mode_lf_delta: [4]i32 = @splat(0),
    filter_type: u2 = 0, // 0 off, 1 simple, 2 complex
};

const TopSamples = struct { y: [16]u8, u: [8]u8, v: [8]u8 };

/// ``kScan``: each 4x4 luma block's offset in the work buffer.
const scan = blk: {
    var s: [16]usize = undefined;
    for (0..16) |n| s[n] = (n & 3) * 4 + (n >> 2) * 4 * bps;
    break :blk s;
};

/// Decode a VP8 chunk payload as the demuxer hands it on (an odd size keeps its pad byte) to cropped Y, U, V planes.
pub fn decode(gpa: Allocator, data: []const u8, max_pixels: u64) Error!Yuv {
    // ``VP8GetInfo`` (from ``ParseHeadersInternal``), the chunk being the whole of ``data``
    if (data.len < 10) return error.BadWebp;
    if (data[3] != 0x9d or data[4] != 0x01 or data[5] != 0x2a) return error.BadWebp;
    const tag = @as(u32, data[0]) | @as(u32, data[1]) << 8 | @as(u32, data[2]) << 16;
    if (tag & 1 != 0 or (tag >> 1) & 7 > 3 or (tag >> 4) & 1 == 0) return error.BadWebp;
    const part0_len: usize = tag >> 5;
    if (part0_len >= data.len) return error.BadWebp;
    const w: u32 = (@as(u32, data[7]) << 8 | data[6]) & 0x3fff;
    const h: u32 = (@as(u32, data[9]) << 8 | data[8]) & 0x3fff;
    if (w == 0 or h == 0) return error.BadWebp;
    // ``VP8GetHeaders``
    const buf = data[10..];
    if (part0_len > buf.len) return error.BadWebp;
    var br = BitReader.init(buf[0..part0_len]);
    const rest = buf[part0_len..];
    _ = br.getBit(0x80); // colorspace
    _ = br.getBit(0x80); // clamp type
    var hdr: Header = .{};
    var ps: tree.Parser = .{ .intra_t = &.{}, .nz_info = &.{} };
    parseSegmentHeader(&br, &hdr, &ps);
    if (br.eof) return error.BadWebp;
    parseFilterHeader(&br, &hdr);
    if (br.eof) return error.BadWebp;
    var parts: [8]BitReader = undefined;
    const last_part: usize = (@as(usize, 1) << @intCast(br.getValue(2))) - 1;
    if (rest.len < 3 * last_part) return error.BadWebp;
    var part_start = 3 * last_part;
    for (0..last_part) |p| {
        const sz = rest[3 * p ..][0..3];
        const psize = @min(@as(usize, sz[0]) | @as(usize, sz[1]) << 8 | @as(usize, sz[2]) << 16, rest.len - part_start);
        parts[p] = BitReader.init(rest[part_start..][0..psize]);
        part_start += psize;
    }
    parts[last_part] = BitReader.init(rest[part_start..]);
    if (part_start >= rest.len) return error.BadWebp;
    parseQuant(&br, &hdr, &ps.dqm);
    _ = br.getBit(0x80); // update_proba, ignored
    ps.parseProba(&br);

    if (@as(u64, w) * h > max_pixels) return error.TooManyPixels;
    var fr = try Frame.init(gpa, w, h, hdr);
    defer fr.deinitScratch(gpa);
    errdefer fr.deinitPlanes(gpa);
    ps.intra_t = fr.intra_t;
    ps.nz_info = fr.nz_info;
    for (0..fr.mb_h) |mb_y| {
        const token_br = &parts[mb_y & last_part];
        for (0..fr.mb_w) |mb_x| ps.parseIntraMode(&br, &fr.mb_data[mb_x], mb_x);
        if (br.eof) return error.BadWebp;
        for (0..fr.mb_w) |mb_x| {
            fr.decodeMb(&ps, token_br, mb_x);
            if (token_br.eof) return error.BadWebp;
        }
        ps.initScanline();
        fr.reconstructRow(mb_y);
        if (hdr.filter_type > 0) fr.filterRow(mb_y);
    }
    return fr.crop(gpa);
}

/// ``ParseSegmentHeader`` (RFC 6386 section 9.3).
fn parseSegmentHeader(br: *BitReader, hdr: *Header, ps: *tree.Parser) void {
    hdr.use_segment = br.getBit(0x80) == 1;
    if (!hdr.use_segment) return;
    ps.update_map = br.getBit(0x80) == 1;
    if (br.getBit(0x80) == 1) {
        hdr.absolute_delta = br.getBit(0x80) == 1;
        for (&hdr.quantizer) |*q| q.* = if (br.getBit(0x80) == 1) br.getSignedValue(7) else 0;
        for (&hdr.filter_strength) |*f| f.* = if (br.getBit(0x80) == 1) br.getSignedValue(6) else 0;
    }
    if (ps.update_map) {
        for (&ps.segments) |*s| s.* = if (br.getBit(0x80) == 1) @intCast(br.getValue(8)) else 255;
    }
}

/// ``ParseFilterHeader`` (RFC 6386 section 9.4).
fn parseFilterHeader(br: *BitReader, hdr: *Header) void {
    hdr.simple = br.getBit(0x80) == 1;
    hdr.level = @intCast(br.getValue(6));
    hdr.sharpness = @intCast(br.getValue(3));
    hdr.use_lf_delta = br.getBit(0x80) == 1;
    if (hdr.use_lf_delta and br.getBit(0x80) == 1) {
        for (&hdr.ref_lf_delta) |*d| {
            if (br.getBit(0x80) == 1) d.* = br.getSignedValue(6);
        }
        for (&hdr.mode_lf_delta) |*d| {
            if (br.getBit(0x80) == 1) d.* = br.getSignedValue(6);
        }
    }
    hdr.filter_type = if (hdr.level == 0) 0 else if (hdr.simple) 1 else 2;
}

fn qclip(v: i32, m: i32) usize {
    return @intCast(std.math.clamp(v, 0, m));
}

/// ``VP8ParseQuant`` (RFC 6386 section 9.6): each segment's dequantization factors.
fn parseQuant(br: *BitReader, hdr: *const Header, dqm: *[4]tree.Quant) void {
    const base_q0: i32 = @intCast(br.getValue(7));
    var dq: [5]i32 = undefined; // y1 dc, y2 dc, y2 ac, uv dc, uv ac
    for (&dq) |*d| d.* = if (br.getBit(0x80) == 1) br.getSignedValue(4) else 0;
    for (0..4) |i| {
        var q: i32 = undefined;
        if (hdr.use_segment) {
            q = hdr.quantizer[i];
            if (!hdr.absolute_delta) q += base_q0;
        } else if (i > 0) {
            dqm[i] = dqm[0];
            continue;
        } else q = base_q0;
        const m = &dqm[i];
        m.y1 = .{ tables.dc_table[qclip(q + dq[0], 127)], tables.ac_table[qclip(q, 127)] };
        m.y2[0] = @as(i32, tables.dc_table[qclip(q + dq[1], 127)]) * 2;
        m.y2[1] = @max(8, (@as(i32, tables.ac_table[qclip(q + dq[2], 127)]) * 101581) >> 16);
        m.uv = .{ tables.dc_table[qclip(q + dq[3], 117)], tables.ac_table[qclip(q + dq[4], 127)] };
    }
}

/// ``PrecomputeFilterStrengths``: the filter limits for each segment, without and with 4x4 prediction.
fn filterStrengths(hdr: *const Header) [4][2]FInfo {
    var out: [4][2]FInfo = @splat(@splat(.{}));
    if (hdr.filter_type == 0) return out;
    for (0..4) |s| {
        var base_level = hdr.level;
        if (hdr.use_segment) {
            base_level = hdr.filter_strength[s];
            if (!hdr.absolute_delta) base_level += hdr.level;
        }
        for (0..2) |i4x4| {
            const info = &out[s][i4x4];
            var level = base_level;
            if (hdr.use_lf_delta) {
                level += hdr.ref_lf_delta[0];
                if (i4x4 == 1) level += hdr.mode_lf_delta[0];
            }
            level = std.math.clamp(level, 0, 63);
            if (level > 0) {
                var ilevel = level;
                if (hdr.sharpness > 0) {
                    ilevel >>= if (hdr.sharpness > 4) 2 else 1;
                    ilevel = @min(ilevel, 9 - hdr.sharpness);
                }
                ilevel = @max(ilevel, 1);
                info.ilevel = @intCast(ilevel);
                info.limit = @intCast(2 * level + ilevel);
                info.hev_thresh = if (level >= 40) 2 else if (level >= 15) 1 else 0;
            } else info.limit = 0;
            info.inner = i4x4 == 1;
        }
    }
    return out;
}

/// ``CheckMode``: DC prediction without the missing top or left samples on the frame's edges.
fn checkMode(mb_x: usize, mb_y: usize, mode: u8) u8 {
    if (mode != 0) return mode;
    if (mb_x == 0) return if (mb_y == 0) 6 else 5;
    return if (mb_y == 0) 4 else 0;
}

fn doTransform(bits: u32, src: []const i16, b: []u8, d: usize) void {
    switch (bits >> 30) {
        3 => dsp.transform(src, b, d),
        2 => dsp.transformAc3(src, b, d),
        1 => dsp.transformDc(src, b, d),
        else => {},
    }
}

fn doUvTransform(bits: u32, src: []const i16, b: []u8, d: usize) void {
    if (bits & 0xff == 0) return;
    if (bits & 0xaa != 0) dsp.transformUv(src, b, d) else dsp.transformDcUv(src, b, d);
}

/// The whole frame at macroblock size, reconstructed and filtered a macroblock row at a time as libwebp's cache is.
const Frame = struct {
    w: u32,
    h: u32,
    mb_w: usize,
    mb_h: usize,
    hdr: Header,
    fstrengths: [4][2]FInfo,
    y: []u8,
    u: []u8,
    v: []u8,
    intra_t: []u8,
    nz_info: []tree.Nz,
    mb_data: []tree.MbData,
    f_info: []FInfo,
    yuv_t: []TopSamples,
    yuv_b: [bps * 17 + bps * 9]u8 = @splat(0),

    fn init(gpa: Allocator, w: u32, h: u32, hdr: Header) Error!Frame {
        const mb_w = (@as(usize, w) + 15) >> 4;
        const mb_h = (@as(usize, h) + 15) >> 4;
        var fr: Frame = .{
            .w = w,
            .h = h,
            .mb_w = mb_w,
            .mb_h = mb_h,
            .hdr = hdr,
            .fstrengths = filterStrengths(&hdr),
            .y = &.{},
            .u = &.{},
            .v = &.{},
            .intra_t = &.{},
            .nz_info = &.{},
            .mb_data = &.{},
            .f_info = &.{},
            .yuv_t = &.{},
        };
        errdefer fr.deinitScratch(gpa);
        errdefer fr.deinitPlanes(gpa);
        fr.y = try gpa.alloc(u8, mb_w * 16 * mb_h * 16);
        fr.u = try gpa.alloc(u8, mb_w * 8 * mb_h * 8);
        fr.v = try gpa.alloc(u8, mb_w * 8 * mb_h * 8);
        fr.intra_t = try gpa.alloc(u8, 4 * mb_w);
        @memset(fr.intra_t, 0);
        fr.nz_info = try gpa.alloc(tree.Nz, mb_w + 1);
        @memset(fr.nz_info, .{});
        fr.mb_data = try gpa.alloc(tree.MbData, mb_w);
        @memset(fr.mb_data, .{});
        fr.f_info = try gpa.alloc(FInfo, mb_w);
        @memset(fr.f_info, .{});
        fr.yuv_t = try gpa.alloc(TopSamples, mb_w);
        return fr;
    }

    fn deinitScratch(fr: *Frame, gpa: Allocator) void {
        gpa.free(fr.intra_t);
        gpa.free(fr.nz_info);
        gpa.free(fr.mb_data);
        gpa.free(fr.f_info);
        gpa.free(fr.yuv_t);
        fr.intra_t = &.{};
        fr.nz_info = &.{};
        fr.mb_data = &.{};
        fr.f_info = &.{};
        fr.yuv_t = &.{};
    }

    fn deinitPlanes(fr: *Frame, gpa: Allocator) void {
        gpa.free(fr.y);
        gpa.free(fr.u);
        gpa.free(fr.v);
        fr.y = &.{};
        fr.u = &.{};
        fr.v = &.{};
    }

    /// ``VP8DecodeMB``: the residuals (unless skipped) and the macroblock's filter limits.
    fn decodeMb(fr: *Frame, ps: *tree.Parser, token_br: *BitReader, mb_x: usize) void {
        const block = &fr.mb_data[mb_x];
        var skip = ps.use_skip_proba and block.skip;
        if (!skip) {
            skip = ps.parseResiduals(token_br, block, mb_x);
        } else {
            const left = &ps.nz_info[0];
            const mb = &ps.nz_info[1 + mb_x];
            left.nz = 0;
            mb.nz = 0;
            if (!block.is_i4x4) {
                left.nz_dc = 0;
                mb.nz_dc = 0;
            }
            block.non_zero_y = 0;
            block.non_zero_uv = 0;
        }
        if (fr.hdr.filter_type > 0) {
            fr.f_info[mb_x] = fr.fstrengths[block.segment][@intFromBool(block.is_i4x4)];
            fr.f_info[mb_x].inner = fr.f_info[mb_x].inner or !skip;
        }
    }

    /// ``ReconstructRow``: predict and add residuals for macroblock row ``mb_y`` into the frame.
    fn reconstructRow(fr: *Frame, mb_y: usize) void {
        const b = &fr.yuv_b;
        for (0..16) |j| b[y_off + j * bps - 1] = 129;
        for (0..8) |j| {
            b[u_off + j * bps - 1] = 129;
            b[v_off + j * bps - 1] = 129;
        }
        if (mb_y > 0) {
            b[y_off - 1 - bps] = 129;
            b[u_off - 1 - bps] = 129;
            b[v_off - 1 - bps] = 129;
        } else {
            @memset(b[y_off - bps - 1 ..][0 .. 16 + 4 + 1], 127);
            @memset(b[u_off - bps - 1 ..][0 .. 8 + 1], 127);
            @memset(b[v_off - bps - 1 ..][0 .. 8 + 1], 127);
        }
        for (0..fr.mb_w) |mb_x| {
            const block = &fr.mb_data[mb_x];
            if (mb_x > 0) { // rotate in the left samples, four at a time, from row -1
                for (0..17) |r| {
                    const at = y_off + r * bps - bps;
                    @memcpy(b[at - 4 ..][0..4], b[at + 12 ..][0..4]);
                }
                for (0..9) |r| {
                    for ([2]usize{ u_off, v_off }) |o| {
                        const at = o + r * bps - bps;
                        @memcpy(b[at - 4 ..][0..4], b[at + 4 ..][0..4]);
                    }
                }
            }
            const top = &fr.yuv_t[mb_x];
            const coeffs = &block.coeffs;
            var bits = block.non_zero_y;
            if (mb_y > 0) {
                @memcpy(b[y_off - bps ..][0..16], &top.y);
                @memcpy(b[u_off - bps ..][0..8], &top.u);
                @memcpy(b[v_off - bps ..][0..8], &top.v);
            }
            if (block.is_i4x4) {
                const tr = y_off - bps + 16;
                if (mb_y > 0) {
                    if (mb_x >= fr.mb_w - 1) {
                        @memset(b[tr..][0..4], top.y[15]);
                    } else @memcpy(b[tr..][0..4], fr.yuv_t[mb_x + 1].y[0..4]);
                }
                for (1..4) |k| @memcpy(b[tr + 4 * k * bps ..][0..4], b[tr..][0..4]);
                for (0..16) |n| {
                    const dst = y_off + scan[n];
                    dsp.predLuma4(block.imodes[n], b, dst);
                    doTransform(bits, coeffs[n * 16 ..][0..16], b, dst);
                    bits <<= 2;
                }
            } else {
                dsp.predLuma16(checkMode(mb_x, mb_y, block.imodes[0]), b, y_off);
                if (bits != 0) {
                    for (0..16) |n| {
                        doTransform(bits, coeffs[n * 16 ..][0..16], b, y_off + scan[n]);
                        bits <<= 2;
                    }
                }
            }
            const uv_mode = checkMode(mb_x, mb_y, block.uvmode);
            dsp.predChroma8(uv_mode, b, u_off);
            dsp.predChroma8(uv_mode, b, v_off);
            doUvTransform(block.non_zero_uv, coeffs[16 * 16 ..][0..64], b, u_off);
            doUvTransform(block.non_zero_uv >> 8, coeffs[20 * 16 ..][0..64], b, v_off);
            if (mb_y < fr.mb_h - 1) {
                @memcpy(&top.y, b[y_off + 15 * bps ..][0..16]);
                @memcpy(&top.u, b[u_off + 7 * bps ..][0..8]);
                @memcpy(&top.v, b[v_off + 7 * bps ..][0..8]);
            }
            const ys = fr.mb_w * 16;
            const uvs = fr.mb_w * 8;
            for (0..16) |j| @memcpy(fr.y[(mb_y * 16 + j) * ys + mb_x * 16 ..][0..16], b[y_off + j * bps ..][0..16]);
            for (0..8) |j| {
                @memcpy(fr.u[(mb_y * 8 + j) * uvs + mb_x * 8 ..][0..8], b[u_off + j * bps ..][0..8]);
                @memcpy(fr.v[(mb_y * 8 + j) * uvs + mb_x * 8 ..][0..8], b[v_off + j * bps ..][0..8]);
            }
        }
    }

    /// ``FilterRow`` / ``DoFilter``: the loop filter over macroblock row ``mb_y``, left to right.
    fn filterRow(fr: *Frame, mb_y: usize) void {
        const ys = fr.mb_w * 16;
        const uvs = fr.mb_w * 8;
        for (0..fr.mb_w) |mb_x| {
            const f = fr.f_info[mb_x];
            if (f.limit == 0) continue;
            const limit: i32 = f.limit;
            const il: i32 = f.ilevel;
            const hv: i32 = f.hev_thresh;
            const yd = mb_y * 16 * ys + mb_x * 16;
            if (fr.hdr.filter_type == 1) {
                if (mb_x > 0) dsp.simpleFilter16(fr.y, yd, ys, limit + 4, false);
                if (f.inner) dsp.simpleFilter16i(fr.y, yd, ys, limit, false);
                if (mb_y > 0) dsp.simpleFilter16(fr.y, yd, ys, limit + 4, true);
                if (f.inner) dsp.simpleFilter16i(fr.y, yd, ys, limit, true);
                continue;
            }
            const ud = mb_y * 8 * uvs + mb_x * 8;
            if (mb_x > 0) {
                dsp.filterLoop(fr.y, yd, 1, ys, 16, limit + 4, il, hv, true);
                for ([2][]u8{ fr.u, fr.v }) |p| dsp.filterLoop(p, ud, 1, uvs, 8, limit + 4, il, hv, true);
            }
            if (f.inner) {
                for (1..4) |k| dsp.filterLoop(fr.y, yd + 4 * k, 1, ys, 16, limit, il, hv, false);
                for ([2][]u8{ fr.u, fr.v }) |p| dsp.filterLoop(p, ud + 4, 1, uvs, 8, limit, il, hv, false);
            }
            if (mb_y > 0) {
                dsp.filterLoop(fr.y, yd, ys, 1, 16, limit + 4, il, hv, true);
                for ([2][]u8{ fr.u, fr.v }) |p| dsp.filterLoop(p, ud, uvs, 1, 8, limit + 4, il, hv, true);
            }
            if (f.inner) {
                for (1..4) |k| dsp.filterLoop(fr.y, yd + 4 * k * ys, ys, 1, 16, limit, il, hv, false);
                for ([2][]u8{ fr.u, fr.v }) |p| dsp.filterLoop(p, ud + 4 * uvs, uvs, 1, 8, limit, il, hv, false);
            }
        }
    }

    /// The planes cropped to the frame size in place (rows move down to the narrower stride), then shrunk.
    fn crop(fr: *Frame, gpa: Allocator) Error!Yuv {
        const w: usize = fr.w;
        const uv_w = (w + 1) / 2;
        const uv_h = (@as(usize, fr.h) + 1) / 2;
        for (0..fr.h) |r| std.mem.copyForwards(u8, fr.y[r * w ..][0..w], fr.y[r * fr.mb_w * 16 ..][0..w]);
        for (0..uv_h) |r| {
            std.mem.copyForwards(u8, fr.u[r * uv_w ..][0..uv_w], fr.u[r * fr.mb_w * 8 ..][0..uv_w]);
            std.mem.copyForwards(u8, fr.v[r * uv_w ..][0..uv_w], fr.v[r * fr.mb_w * 8 ..][0..uv_w]);
        }
        fr.y = try gpa.realloc(fr.y, w * fr.h);
        fr.u = try gpa.realloc(fr.u, uv_w * uv_h);
        fr.v = try gpa.realloc(fr.v, uv_w * uv_h);
        const out: Yuv = .{ .w = fr.w, .h = fr.h, .y = fr.y, .u = fr.u, .v = fr.v, .uv_w = @intCast(uv_w) };
        fr.y = &.{};
        fr.u = &.{};
        fr.v = &.{};
        return out;
    }
};

/// A 1x1 key frame: q 0, no filter, one partition of zero tokens.
const tiny = [_]u8{ 0x50, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x01, 0x00, 0x01, 0x00 } ++ @as([14]u8, @splat(0));

test "a 1x1 frame of zero tokens decodes to the DC predictors' 128 grey" {
    var yuv = try decode(std.testing.allocator, &tiny, 1 << 20);
    defer yuv.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), yuv.uv_w);
    try std.testing.expectEqual(@as(usize, 1), yuv.y.len);
    try std.testing.expectEqual(@as(u8, 128), yuv.u[0]);
}

test "malformed and oversized frames are refused" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadWebp, decode(gpa, tiny[0..9], 1 << 20));
    var bad = tiny;
    bad[3] = 0;
    try std.testing.expectError(error.BadWebp, decode(gpa, &bad, 1 << 20));
    try std.testing.expectError(error.BadWebp, decode(gpa, tiny[0..20], 1 << 20)); // no token partition
    try std.testing.expectError(error.TooManyPixels, decode(gpa, &tiny, 0));
    for (0..tiny.len) |n| {
        if (decode(gpa, tiny[0..n], 1 << 20)) |y| {
            var yy = y;
            yy.deinit(gpa);
        } else |_| {}
    }
}
