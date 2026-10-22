//! A RIFF WEBP file as libwebp 1.6.0 accepts it: ``WebPGetFeatures`` (``webp_dec.c``) and ``WebPDemux`` (``demux.c``).
const std = @import("std");

pub const Error = error{BadWebp};

const max_payload: u64 = std.math.maxInt(u32) - 8 - 1;
const max_area: u64 = 1 << 32;
const flag_animation: u32 = 0x02;
const flag_alpha: u32 = 0x10;
const valid_flags: u32 = 0x3e;

fn le16(b: []const u8) u32 {
    return std.mem.readInt(u16, b[0..2], .little);
}

fn le24(b: []const u8) u32 {
    return std.mem.readInt(u24, b[0..3], .little);
}

fn le32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .little);
}

fn tagIs(b: []const u8, t: *const [4]u8) bool {
    return b.len >= 4 and std.mem.eql(u8, b[0..4], t);
}

/// ``VP8GetInfo``: the start code, a visible profile 0-3 key frame, the first partition inside, nonzero sizes.
pub fn vp8Info(b: []const u8, chunk_size: u64) ?[2]u32 {
    if (b.len < 10 or !std.mem.eql(u8, b[3..6], "\x9d\x01\x2a")) return null;
    const bits = le24(b);
    if (bits & 1 != 0 or (bits >> 1) & 7 > 3 or (bits >> 4) & 1 == 0 or bits >> 5 >= chunk_size) return null;
    const w = le16(b[6..]) & 0x3fff;
    const h = le16(b[8..]) & 0x3fff;
    if (w == 0 or h == 0) return null;
    return .{ w, h };
}

/// ``VP8LCheckSignature``: the 0x2f byte and a zero version in the header's last byte.
fn vp8lSignature(b: []const u8) bool {
    return b.len >= 5 and b[0] == 0x2f and b[4] >> 5 == 0;
}

/// ``VP8LGetInfo``: width, height and the alpha hint of a VP8L header.
pub fn vp8lInfo(b: []const u8) ?struct { w: u32, h: u32, alpha: bool } {
    if (!vp8lSignature(b)) return null;
    const bits = le32(b[1..]);
    return .{ .w = (bits & 0x3fff) + 1, .h = ((bits >> 14) & 0x3fff) + 1, .alpha = (bits >> 28) & 1 != 0 };
}

pub const Features = struct { w: u32, h: u32, has_alpha: bool };
const Status = enum { ok, short, bad };

/// ``WebPGetFeatures`` (``ParseHeadersInternal`` without headers): null where libwebp's status is not OK.
pub fn features(data: []const u8) ?Features {
    var f: Features = .{ .w = 0, .h = 0, .has_alpha = false };
    var vp8x = false;
    const st = parseFeatures(data, &f, &vp8x);
    if (st == .ok or (st == .short and vp8x)) return f;
    return null;
}

fn parseFeatures(data0: []const u8, f: *Features, vp8x: *bool) Status {
    var d = data0;
    if (d.len < 12) return .short;
    var riff: u64 = 0;
    if (tagIs(d, "RIFF")) {
        if (!std.mem.eql(u8, d[8..12], "WEBP")) return .bad;
        const size = le32(d[4..]);
        if (size < 12 or size > max_payload) return .bad;
        riff = size;
        d = d[12..];
    }
    if (d.len < 8) return .short;
    var animation = false;
    if (tagIs(d, "VP8X")) {
        if (le32(d[4..]) != 10) return .bad;
        if (d.len < 18) return .short;
        const flags = le32(d[8..]);
        f.w = 1 + le24(d[12..]);
        f.h = 1 + le24(d[15..]);
        if (@as(u64, f.w) * f.h >= max_area) return .bad;
        d = d[18..];
        vp8x.* = true;
        animation = flags & flag_animation != 0;
        f.has_alpha = flags & flag_alpha != 0;
    }
    if (riff == 0 and vp8x.*) return .bad;
    if (vp8x.* and animation) return .ok;
    if (d.len < 4) return .short;
    var alph = false;
    if ((riff > 0 and vp8x.*) or (riff == 0 and !vp8x.* and tagIs(d, "ALPH"))) {
        var total: u64 = 4 + 8 + 10;
        while (true) {
            if (d.len < 8) return .short;
            const size = le32(d[4..]);
            if (size > max_payload) return .bad;
            const disk = (8 + @as(u64, size) + 1) & ~@as(u64, 1);
            total += disk;
            if (riff > 0 and total > riff) return .bad;
            if (tagIs(d, "VP8 ") or tagIs(d, "VP8L")) break;
            if (d.len < disk) return .short;
            if (tagIs(d, "ALPH")) alph = true;
            d = d[@intCast(disk)..];
        }
    }
    if (d.len < 8) return finish(.short, f, alph);
    var lossless: bool = undefined;
    var compressed: u64 = undefined;
    if (tagIs(d, "VP8 ") or tagIs(d, "VP8L")) {
        const size = le32(d[4..]);
        if (riff >= 12 and size > riff - 12) return .bad;
        compressed = size;
        lossless = tagIs(d, "VP8L");
        d = d[8..];
    } else {
        lossless = vp8lSignature(d);
        compressed = d.len;
    }
    if (compressed > max_payload) return .bad;
    var w: u32 = undefined;
    var h: u32 = undefined;
    if (!lossless) {
        if (d.len < 10) return finish(.short, f, alph);
        const s = vp8Info(d, compressed) orelse return .bad;
        w = s[0];
        h = s[1];
    } else {
        if (d.len < 5) return finish(.short, f, alph);
        const s = vp8lInfo(d) orelse return .bad;
        w = s.w;
        h = s.h;
        f.has_alpha = s.alpha;
    }
    if (vp8x.* and (f.w != w or f.h != h)) return .bad;
    f.w = w;
    f.h = h;
    return finish(.ok, f, alph);
}

fn finish(st: Status, f: *Features, alph: bool) Status {
    f.has_alpha = f.has_alpha or alph;
    return st;
}

/// A chunk's offset and length (header and padded payload) in the file.
pub const Span = struct { at: usize = 0, len: usize = 0 };

pub const Frame = struct {
    num: u32 = 0,
    x: u32 = 0,
    y: u32 = 0,
    w: u32 = 0,
    h: u32 = 0,
    complete: bool = false,
    image: Span = .{},
    alpha: Span = .{},
};

/// What the first frame's decode needs from a demuxed file.
pub const Demuxed = struct { canvas_w: u32, canvas_h: u32, first: Frame };

const Parse = enum { ok, more, bad };

const Mem = struct {
    buf: []const u8,
    start: usize,
    end: usize,

    fn left(m: *const Mem) usize {
        return m.end - m.start;
    }
    fn invalid(m: *const Mem, size: u64) bool {
        return size > m.end - m.start;
    }
    fn read32(m: *Mem) u32 {
        defer m.start += 4;
        return le32(m.buf[m.start..]);
    }
    fn read24(m: *Mem) u32 {
        defer m.start += 3;
        return le24(m.buf[m.start..]);
    }
};

const Demux = struct {
    mem: Mem,
    ext: bool = false,
    flags: u32 = 0,
    canvas_w: u32 = 0,
    canvas_h: u32 = 0,
    frames: u32 = 0,
    first: Frame = .{},
    last_complete: bool = true,
    bad_frame: bool = false,

    /// ``AddFrame`` (refused after an incomplete frame) and ``IsValid*Format``'s per-frame rules (ALPH before image).
    fn add(d: *Demux, f: Frame) bool {
        if (!d.last_complete) return false;
        d.last_complete = f.complete;
        if (d.frames == 0) d.first = f;
        d.frames += 1;
        if (!f.complete or f.w == 0 or f.h == 0 or (f.alpha.len > 0 and f.alpha.at > f.image.at)) d.bad_frame = true;
        if (d.ext and !boundsOk(d, f, d.flags & flag_animation == 0)) d.bad_frame = true;
        return true;
    }
};

/// ``StoreFrame``: an optional ALPH, then one VP8 or VP8L chunk (VP8L after ALPH is an error), up to another chunk.
fn storeFrame(num: u32, min_size: u64, m: *Mem, f: *Frame) Parse {
    if (m.left() < 8 or m.left() < min_size) return .more;
    var alpha_chunks: u32 = 0;
    var image_chunks: u32 = 0;
    var status: Parse = .ok;
    while (true) {
        const at = m.start;
        const tag = m.buf[at..][0..4];
        m.start += 4;
        const size = m.read32();
        if (size > max_payload) return .bad;
        const padded = @as(u64, size) + (size & 1);
        if (m.invalid(padded)) return .bad;
        const avail: usize = @intCast(padded);
        const is_alph = std.mem.eql(u8, tag, "ALPH");
        const is_vp8l = std.mem.eql(u8, tag, "VP8L");
        const is_image = is_vp8l or std.mem.eql(u8, tag, "VP8 ");
        if (is_alph and alpha_chunks == 0) {
            alpha_chunks += 1;
            f.alpha = .{ .at = at, .len = 8 + avail };
            f.num = num;
            m.start += avail;
        } else if (is_image and is_vp8l and alpha_chunks > 0) {
            return .bad;
        } else if (is_image and image_chunks == 0) {
            const feat = features(m.buf[at .. at + 8 + avail]) orelse return .bad;
            image_chunks += 1;
            f.image = .{ .at = at, .len = 8 + avail };
            f.w = feat.w;
            f.h = feat.h;
            f.num = num;
            f.complete = status == .ok;
            m.start += avail;
        } else {
            m.start -= 8;
            return status;
        }
        if (m.start == m.end) return status;
        if (m.left() < 8) status = .more;
        if (status != .ok) return status;
    }
}

/// ``ParseSingleImage``: one frame; without the VP8X alpha flag its ALPH chunk is dropped.
fn singleImage(d: *Demux) Parse {
    if (d.frames != 0 or d.mem.invalid(8)) return .bad;
    var f: Frame = .{};
    const status = storeFrame(1, 0, &d.mem, &f);
    if (status == .bad) return .bad;
    if (d.flags & flag_alpha == 0) f.alpha = .{};
    if (!d.ext and f.w > 0 and f.h > 0) {
        d.canvas_w = f.w;
        d.canvas_h = f.h;
    }
    if (!d.add(f)) return .bad;
    return status;
}

/// ``ParseAnimationFrame``: the ANMF header, then the frame's chunks, which may not run past its payload.
fn animationFrame(d: *Demux, chunk_size: u64) Parse {
    if (d.mem.invalid(16) or chunk_size < 16) return .bad;
    var f: Frame = .{};
    f.x = 2 * d.mem.read24();
    f.y = 2 * d.mem.read24();
    const w = 1 + @as(u64, d.mem.read24());
    const h = 1 + @as(u64, d.mem.read24());
    d.mem.start += 4;
    if (w * h >= max_area) return .bad;
    const start = d.mem.start;
    var status = storeFrame(d.frames + 1, chunk_size - 16, &d.mem, &f);
    if (status != .bad and d.mem.start - start > chunk_size - 16) status = .bad;
    if (status != .bad and d.flags & flag_animation != 0 and f.num > 0) {
        if (!d.add(f)) status = .bad;
    }
    return status;
}

fn vp8xChunks(d: *Demux) Parse {
    const animation = d.flags & flag_animation != 0;
    var anim_chunks: u32 = 0;
    var status: Parse = .ok;
    while (status == .ok) {
        const tag = d.mem.buf[d.mem.start..][0..4];
        d.mem.start += 4;
        const size = d.mem.read32();
        if (size > max_payload) return .bad;
        const padded = @as(u64, size) + (size & 1);
        if (d.mem.invalid(padded)) return .bad;
        if (std.mem.eql(u8, tag, "VP8X")) {
            return .bad;
        } else if (std.mem.eql(u8, tag, "ALPH") or std.mem.eql(u8, tag, "VP8 ") or std.mem.eql(u8, tag, "VP8L")) {
            if (anim_chunks > 0 or animation) return .bad;
            d.mem.start -= 8;
            status = singleImage(d);
        } else if (std.mem.eql(u8, tag, "ANIM")) {
            if (padded < 6) return .bad;
            anim_chunks += 1;
            d.mem.start += @intCast(padded);
        } else if (std.mem.eql(u8, tag, "ANMF")) {
            if (anim_chunks == 0) return .bad;
            status = animationFrame(d, padded);
        } else {
            d.mem.start += @intCast(padded);
        }
        if (d.mem.start == d.mem.end) break;
        if (d.mem.left() < 8) status = .more;
    }
    return status;
}

/// ``ParseVP8X``: flags from the first payload byte, the canvas, any trailing VP8X payload skipped.
fn vp8xHeader(d: *Demux) Parse {
    if (d.mem.left() < 8) return .more;
    d.ext = true;
    d.mem.start += 4;
    const size = d.mem.read32();
    if (size > max_payload or size < 10) return .bad;
    const padded = @as(u64, size) + (size & 1);
    if (d.mem.invalid(padded)) return .bad;
    d.flags = d.mem.buf[d.mem.start];
    d.mem.start += 4;
    d.canvas_w = 1 + d.mem.read24();
    d.canvas_h = 1 + d.mem.read24();
    if (@as(u64, d.canvas_w) * d.canvas_h >= max_area) return .bad;
    d.mem.start += @intCast(padded - 10);
    if (d.mem.invalid(8)) return .bad;
    return vp8xChunks(d);
}

/// ``WebPDemux`` of a complete file (no partial data) with ``IsValidSimpleFormat`` / ``IsValidExtendedFormat``.
pub fn demux(raw: []const u8) Error!Demuxed {
    if (raw.len < 20 or !tagIs(raw, "RIFF") or !std.mem.eql(u8, raw[8..12], "WEBP")) return error.BadWebp;
    const riff = le32(raw[4..]);
    if (riff < 8 or riff > max_payload) return error.BadWebp;
    const end = @as(u64, riff) + 8;
    if (raw.len < end) return error.BadWebp;
    var d: Demux = .{ .mem = .{ .buf = raw[0..@intCast(end)], .start = 12, .end = @intCast(end) } };
    const tag = raw[12..16];
    const status: Parse = if (std.mem.eql(u8, tag, "VP8 ") or std.mem.eql(u8, tag, "VP8L"))
        singleImage(&d)
    else if (std.mem.eql(u8, tag, "VP8X"))
        vp8xHeader(&d)
    else
        .bad;
    if (status != .ok or d.frames == 0 or d.bad_frame or d.canvas_w == 0 or d.canvas_h == 0) return error.BadWebp;
    if (d.ext and d.flags & ~valid_flags != 0) return error.BadWebp;
    return .{ .canvas_w = d.canvas_w, .canvas_h = d.canvas_h, .first = d.first };
}

/// ``CheckFrameBounds``: a still image fills the canvas exactly; a frame lies inside it.
fn boundsOk(d: *const Demux, f: Frame, exact: bool) bool {
    if (exact) return f.x == 0 and f.y == 0 and f.w == d.canvas_w and f.h == d.canvas_h;
    return @as(u64, f.x) + f.w <= d.canvas_w and @as(u64, f.y) + f.h <= d.canvas_h;
}
