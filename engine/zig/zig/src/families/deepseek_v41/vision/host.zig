//! Rank 0's image front end (``vision_prep.Host``): settings, the virtual-id registry, a cache of preprocessed
//! images keyed by the data: URL's SHA-256 (TF_DSV41_VISION_PREP_MB), at most TF_DSV41_VISION_PREP_SLOTS requests
//! preparing at once. Prepared images are shared and counted: a cache eviction never frees one a request holds.
const std = @import("std");
const prep = @import("prep.zig");
const vids_mod = @import("vids.zig");
const parts = @import("parts.zig");
const decode = @import("decode.zig");
const fetch_mod = @import("fetch.zig");
const Allocator = std.mem.Allocator;

/// One prepared image, shared by the cache and the requests that hold it.
pub const Shared = struct {
    refs: std.atomic.Value(u32) = .init(1),
    p: prep.Prepared,

    pub fn retain(s: *Shared) *Shared {
        _ = s.refs.fetchAdd(1, .monotonic);
        return s;
    }
};

/// Where an https image URL comes from (``image_fetch.fetch_image``): null means this server fetches none.
pub const Fetcher = struct {
    ctx: *anyopaque,
    fetch: *const fn (ctx: *anyopaque, a: Allocator, url: []const u8, s: *const prep.Settings, deadline_ns: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }![]u8,
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    s: prep.Settings,
    registry: vids_mod.Registry,
    fetcher: ?Fetcher = null,
    owned_fetcher: ?*fetch_mod.Fetcher = null,
    mutex: std.Io.Mutex = .init,
    cache: std.AutoArrayHashMapUnmanaged([32]u8, *Shared) = .empty, // oldest first
    cache_bytes: u64 = 0,
    preparing: std.atomic.Value(u32) = .init(0),

    pub fn init(gpa: Allocator, io: std.Io, s: prep.Settings) Host {
        return .{ .gpa = gpa, .io = io, .s = s, .registry = .{ .gpa = gpa } };
    }

    /// TF_DSV41_VISION_FETCH=1 (the default): image URLs through prod's fetch policy (``fetch.zig``).
    pub fn enableFetch(h: *Host) Allocator.Error!void {
        if (!h.s.fetch or h.owned_fetcher != null) return;
        const f = try h.gpa.create(fetch_mod.Fetcher);
        f.* = fetch_mod.Fetcher.init(h.gpa, h.io, &h.s);
        h.owned_fetcher = f;
        h.fetcher = f.it();
    }

    pub fn deinit(h: *Host) void {
        if (h.owned_fetcher) |f| {
            f.deinit();
            h.gpa.destroy(f);
        }
        for (h.cache.values()) |e| h.release(e);
        h.cache.deinit(h.gpa);
        h.registry.deinit();
    }

    pub fn release(h: *Host, e: *Shared) void {
        if (e.refs.fetchSub(1, .acq_rel) != 1) return;
        e.p.deinit(h.gpa);
        h.gpa.destroy(e);
    }

    /// ``Host.prepare``: one URL's image, held for the caller (``release`` it).
    pub fn prepare(h: *Host, a: Allocator, url: []const u8, deadline_ns: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }!*Shared {
        var key: ?[32]u8 = null;
        if (std.mem.startsWith(u8, url, "data:")) {
            var k: [32]u8 = undefined;
            var hasher = std.crypto.hash.sha2.Sha256.init(.{});
            hasher.update(url);
            hasher.final(&k);
            key = k;
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            if (h.cache.fetchOrderedRemove(k)) |hit| { // to the newest end
                h.cache.putAssumeCapacity(k, hit.value);
                return hit.value.retain();
            }
        }
        const raw = if (key != null) try parts.dataUrl(a, url, h.s.max_bytes, problem) else try h.remote(a, url, deadline_ns, problem);
        var which: []const u8 = "";
        var rgb = decode.rgb(h.gpa, raw, h.s.max_pixels, &which) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.TooManyPixels => refuse(problem, try std.fmt.allocPrint(a, "image of more than {d} pixels (TF_DSV41_VISION_MAX_PIXELS)", .{h.s.max_pixels})),
            error.TooLarge => refuse(problem, try std.fmt.allocPrint(a, "could not decode the image ({s}): its Huffman tables need more than {d} MiB", .{ which, decode.max_table_bytes >> 20 })),
            error.Unsupported => refuse(problem, try std.fmt.allocPrint(a, "could not decode the image: {s} images are not supported by this engine (PNG, JPEG, GIF, BMP and WebP are)", .{which})),
            error.Undecodable => refuse(problem, try std.fmt.allocPrint(a, "could not decode the image ({s})", .{if (which.len > 0) which else "unknown format"})),
        };
        defer rgb.deinit(h.gpa);
        var p = prep.preprocess(h.gpa, rgb, &h.s) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.NoGrid => refuse(problem, try std.fmt.allocPrint(a, "image of {d}x{d}: no grid fits {d} positions", .{ rgb.w, rgb.h, h.s.budget() })),
            error.BadSettings => refuse(problem, "the vision settings are invalid"),
        };
        errdefer p.deinit(h.gpa);
        p.vids = try h.registry.vids(h.io, &p.digest, p.tokens());
        const e = try h.gpa.create(Shared);
        e.* = .{ .p = p };
        const k = key orelse return e;
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const got = try h.cache.getOrPut(h.gpa, k);
        if (got.found_existing) { // a racing request prepared it first: keep theirs
            const theirs = got.value_ptr.*.retain();
            h.release(e);
            return theirs;
        }
        got.value_ptr.* = e.retain();
        h.cache_bytes += e.p.patches.len * 2;
        while (h.cache_bytes > h.s.prep_mb << 20 and h.cache.count() > 1) {
            const old = h.cache.values()[0];
            h.cache.orderedRemoveAt(0);
            h.cache_bytes -= old.p.patches.len * 2;
            h.release(old);
        }
        return e;
    }

    fn remote(h: *Host, a: Allocator, url: []const u8, deadline_ns: i96, problem: *parts.Problem) error{ Refused, OutOfMemory }![]u8 {
        const colon = std.mem.indexOfScalar(u8, url, ':') orelse 0;
        const scheme = url[0..colon];
        if (!std.ascii.eqlIgnoreCase(scheme, "http") and !std.ascii.eqlIgnoreCase(scheme, "https"))
            return refuse(problem, try std.fmt.allocPrint(a, "image URL scheme '{s}': only data:, http and https are accepted", .{if (scheme.len > 0) scheme else "(none)"}));
        if (!h.s.fetch) return refuse(problem, "this server does not fetch image URLs (TF_DSV41_VISION_FETCH=0): send a data: URL");
        const f = h.fetcher orelse return refuse(problem, "this engine does not fetch image URLs yet: send the image as a data: URL");
        return f.fetch(f.ctx, a, url, &h.s, deadline_ns, problem);
    }

    /// ``Host.images``: every URL's image in order, held for the request (``releaseAll``).
    pub fn images(h: *Host, a: Allocator, urls: []const []const u8, problem: *parts.Problem) error{ Refused, OutOfMemory }![]*Shared {
        if (urls.len == 0) return &.{};
        if (urls.len > h.s.max_images) return refuse(problem, try std.fmt.allocPrint(a, "{d} images in one request: at most {d} (TF_DSV41_VISION_MAX_IMAGES)", .{ urls.len, h.s.max_images }));
        // ``self.slots.acquire(timeout=60)``: a slot within 60 s, else busy (HTTP 503)
        const give_up = std.Io.Clock.awake.now(h.io).nanoseconds + 60 * std.time.ns_per_s;
        while (true) {
            const cur = h.preparing.load(.acquire);
            if (cur < h.s.prep_slots) {
                if (h.preparing.cmpxchgWeak(cur, cur + 1, .acq_rel, .acquire) == null) break;
                continue;
            }
            if (std.Io.Clock.awake.now(h.io).nanoseconds >= give_up) {
                problem.* = .{ .message = "image processing capacity is busy; retry shortly", .busy = true };
                return error.Refused;
            }
            std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
        }
        defer _ = h.preparing.fetchSub(1, .acq_rel);
        const deadline = std.Io.Clock.awake.now(h.io).nanoseconds + @as(i96, @intFromFloat(h.s.fetch_total_s * 1e9));
        const out = try a.alloc(*Shared, urls.len);
        var n: usize = 0;
        errdefer for (out[0..n]) |e| h.release(e);
        for (urls) |u| {
            out[n] = try h.prepare(a, u, deadline, problem);
            n += 1;
        }
        return out;
    }

    pub fn releaseAll(h: *Host, held: []const *Shared) void {
        for (held) |e| h.release(e);
    }
};

fn refuse(problem: *parts.Problem, msg: []const u8) error{Refused} {
    problem.* = .{ .message = msg };
    return error.Refused;
}
