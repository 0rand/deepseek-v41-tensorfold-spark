//! A rank's TP settings from the environment (the Python engine's names where it has one): rank, master, port, backend, fail-fast knobs.
const std = @import("std");

pub const Backend = enum {
    /// NCCL for everything (the Python engine's default).
    nccl,
    /// Exchanges of at most `mailbox_max_bytes` a rank through the one-shot mailbox kernel, NCCL above that.
    mailbox,
    /// The same split between two hosts: the small exchanges over RoCE (doorbell + RDMA proxy), NCCL above.
    roce,
};

/// What `roce` does when its setup or load-time probe fails on any rank (GLM53_TF_ROCE_FALLBACK).
pub const Fallback = enum { nccl, @"error" };

pub const Error = error{BadSetting};

/// RoCE's per-write cap (roce_verbs.max_write): a setting whose slot or post would pass it fails boot, never clamped.
pub const max_rdma_write: usize = @import("roce_verbs.zig").max_write;

/// TF_DSV41_PLAN_PIN: the CPU the plan threads pin to at their first plan (auto: the highest cpu_capacity).
pub const PlanPin = union(enum) { auto, cpu: u32 };

pub const Config = struct {
    world: u32 = 2,
    rank: u32 = 0,
    /// Rank 0's address as the other ranks reach it (the CX7 address on the Sparks).
    master: []const u8 = "localhost",
    /// The rendezvous port of the Python engine (`--port`); the link listens on `link_port`.
    port: u16 = 29500,
    /// Rank 0 listens here: TF_DSV41_PLAN_PORT, else `port` + 7 (Python's plan link port).
    link_port: u16 = 29507,
    /// How long rank 0 waits for the others and they keep retrying the connect (TF_DSV41_PLAN_TIMEOUT).
    connect_timeout_ns: u64 = 300 * std.time.ns_per_s,
    /// CUDA device ordinal this rank uses.
    device: u32 = 0,
    backend: Backend = .nccl,
    mailbox_max_bytes: usize = 256 * 1024,
    /// TF_TP_ROCE_VAR_KB: `allGatherV` strides up to this go RoCE (0: the max above); other ops keep their route.
    roce_var_bytes: usize = 0,
    /// TF_TP_ROCE_CHUNKED_KB: `allGatherV` strides up to this go RoCE in <= 1 MiB ops (0: off).
    roce_chunked_bytes: usize = 0,
    /// A mailbox wait for the peer gives up after this long and poisons the transport (fail-stop).
    mailbox_timeout_ns: u64 = 120 * std.time.ns_per_s,
    failfast: bool = true,
    /// After a failure: time for the HTTP threads to answer before exit 70.
    grace_ns: u64 = std.time.ns_per_s,
    /// Rank 1's memory report period on the fate channel (also how fast it notices rank 0 hanging up).
    report_ns: u64 = std.time.ns_per_s / 2,
    nccl_lib: []const u8 = "libnccl.so.2",
    /// `name[:gid],...` RDMA devices for `roce` (null: every active Ethernet port with an IPv4 RoCE v2 GID).
    roce_hca: ?[]const u8 = null,
    /// Stripe over at most this many HCAs (auto detection only).
    roce_hcas: u8 = 2,
    roce_tc: u8 = 0,
    /// GLM53_TF_ROCE_FAST (roce.cu gather_fast_kernel) as tp_roce's `opts` word: bit 0 on, bits 8-15 pollers a flag
    /// (GLM53_TF_ROCE_POLLERS, 1-8), bits 16-31 their stagger (GLM53_TF_ROCE_STAGGER_NS, 150); local, not agreed
    roce_fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS (GLM53_TF_ROCE_BLOCKS): the largest tp_roce grid, a power of two 1..16 (one block per 16 KiB
    /// below it). 16 by default here; Python's roce.py default is 8, so a 16-row decode exchange (160 KiB) runs on 16
    /// blocks in Zig and 8 in Python. Launch shape only: the same bytes land in the same places.
    roce_blocks: u32 = 16,
    /// Pin the proxy thread (it busy-spins one core between exchanges).
    roce_cpu: ?u32 = null,
    /// `shm` runs the RoCE protocol on one host without RDMA (tests); `verbs` is the real wire.
    roce_wire: enum { verbs, shm } = .verbs,
    roce_fallback: Fallback = .nccl,
    /// TF_DSV41_PLAN_LINK=rdma (Python's RoCE host mailbox, whose follower spins on a flag): the follower polls the
    /// plan socket this long before it blocks (TF_DSV41_PLAN_SPIN_US, 5,000); 0 (tcp, nccl, unset): it blocks at once.
    plan_spin_ns: u64 = 0,
    plan_pin: ?PlanPin = null,

    /// Reads every setting from `get` (the environment in `fromEnv`); unset names keep the defaults.
    pub fn parse(get: *const fn ([]const u8) ?[]const u8) Error!Config {
        var c: Config = .{};
        c.world = try int(u32, get, "TF_TP_WORLD", c.world, 1, 64);
        c.rank = try int(u32, get, "TF_TP_RANK", c.rank, 0, c.world - 1);
        if (get("TF_TP_MASTER")) |m| c.master = m;
        c.port = try int(u16, get, "TF_TP_PORT", c.port, 1, 65535 - 8);
        c.link_port = try int(u16, get, "TF_DSV41_PLAN_PORT", c.port + 7, 1, 65535);
        c.connect_timeout_ns = try seconds(get, "TF_DSV41_PLAN_TIMEOUT", 300, 0.1, 86400);
        c.device = try int(u32, get, "TF_TP_DEVICE", c.device, 0, 63);
        // prod.env's names (GLM53_TF_*) where ours are unset: the same meaning, ours win
        const backend = alias(get, "TF_COMM_BACKEND", "GLM53_TF_COMM_BACKEND");
        if (get(backend)) |b| c.backend = std.meta.stringToEnum(Backend, std.mem.trim(u8, b, " ")) orelse {
            std.log.warn("{s}={s}: expected nccl, mailbox or roce", .{ backend, b });
            return error.BadSetting;
        };
        c.mailbox_max_bytes = 1024 * try int(usize, get, alias(get, "TF_TP_MAILBOX_MAX_KB", "GLM53_TF_ROCE_MAX_KB"), 256, 4, 16 * 1024);
        c.roce_var_bytes = 1024 * try int(usize, get, "TF_TP_ROCE_VAR_KB", 0, 0, 32 * 1024);
        if (c.roceSlotBytes() > max_rdma_write) {
            std.log.warn("TF_TP_MAILBOX_MAX_KB / GLM53_TF_ROCE_MAX_KB = {d} KiB, TF_TP_ROCE_VAR_KB = {d} KiB: an RDMA write is capped at {d} KiB; refusing to start", .{ c.mailbox_max_bytes / 1024, c.roce_var_bytes / 1024, max_rdma_write / 1024 });
            return error.BadSetting;
        }
        c.roce_chunked_bytes = 1024 * try int(usize, get, "TF_TP_ROCE_CHUNKED_KB", 0, 0, 64 * 1024);
        c.mailbox_timeout_ns = try seconds(get, "TF_TP_MAILBOX_TIMEOUT_S", 120, 0.001, 86400);
        if (get("TF_DSV41_FAILFAST")) |v| c.failfast = !(std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off"));
        c.grace_ns = try seconds(get, "TF_DSV41_FAILFAST_GRACE_S", 1.0, 0, 60);
        c.report_ns = try seconds(get, "TF_DSV41_PEER_MEM_S", 0.5, 0.01, 60);
        if (get("TF_NCCL_LIB")) |p| c.nccl_lib = p;
        c.roce_hca = get("TF_TP_ROCE_HCA");
        c.roce_hcas = try int(u8, get, "TF_TP_ROCE_HCAS", 2, 1, 2);
        c.roce_tc = try int(u8, get, "TF_TP_ROCE_TC", try int(u8, get, "NCCL_IB_TC", 0, 0, 255), 0, 255);
        if (try int(u8, get, alias(get, "TF_TP_ROCE_FAST", "GLM53_TF_ROCE_FAST"), 0, 0, 1) == 1) {
            const pollers = try int(u32, get, alias(get, "TF_TP_ROCE_POLLERS", "GLM53_TF_ROCE_POLLERS"), 1, 1, 8);
            const stagger = try int(u32, get, alias(get, "TF_TP_ROCE_STAGGER_NS", "GLM53_TF_ROCE_STAGGER_NS"), 150, 0, 65535);
            c.roce_fast = 1 | (pollers << 8) | (stagger << 16);
        }
        c.roce_blocks = try int(u32, get, alias(get, "TF_TP_ROCE_BLOCKS", "GLM53_TF_ROCE_BLOCKS"), 16, 1, 16);
        if (c.roce_blocks & (c.roce_blocks - 1) != 0) {
            std.log.warn("TF_TP_ROCE_BLOCKS={d}: expected a power of two", .{c.roce_blocks});
            return error.BadSetting;
        }
        if (get("TF_TP_ROCE_CPU")) |_| c.roce_cpu = try int(u32, get, "TF_TP_ROCE_CPU", 0, 0, 4095);
        if (get("TF_TP_ROCE_WIRE")) |w| c.roce_wire = std.meta.stringToEnum(@TypeOf(c.roce_wire), w) orelse {
            std.log.warn("TF_TP_ROCE_WIRE={s}: expected verbs or shm", .{w});
            return error.BadSetting;
        };
        const fallback = alias(get, "TF_TP_ROCE_FALLBACK", "GLM53_TF_ROCE_FALLBACK");
        if (get(fallback)) |f| c.roce_fallback = std.meta.stringToEnum(Fallback, f) orelse {
            std.log.warn("{s}={s}: expected nccl or error", .{ fallback, f });
            return error.BadSetting;
        };
        if (get("TF_DSV41_PLAN_LINK")) |l| {
            const mode = std.mem.trim(u8, l, " ");
            if (std.mem.eql(u8, mode, "rdma")) {
                c.plan_spin_ns = 1000 * try int(u64, get, "TF_DSV41_PLAN_SPIN_US", 5000, 0, 10_000_000);
            } else if (!std.mem.eql(u8, mode, "tcp") and !std.mem.eql(u8, mode, "nccl")) {
                std.log.warn("TF_DSV41_PLAN_LINK={s}: expected nccl, tcp or rdma", .{l});
                return error.BadSetting;
            }
        }
        if (get("TF_DSV41_PLAN_PIN")) |raw| {
            const v = std.mem.trim(u8, raw, " ");
            if (std.mem.eql(u8, v, "auto")) {
                c.plan_pin = .auto;
            } else if (!std.mem.eql(u8, v, "-1") and !std.mem.eql(u8, v, "off")) {
                c.plan_pin = .{ .cpu = try int(u32, get, "TF_DSV41_PLAN_PIN", 0, 0, 4095) };
            }
        }
        return c;
    }

    pub fn fromEnv() Error!Config {
        return parse(&envGet);
    }

    /// The RoCE region's slot, every RoCE post's ceiling: the small exchanges' cap or the variable-length strides'.
    pub fn roceSlotBytes(c: Config) usize {
        return @max(c.mailbox_max_bytes, c.roce_var_bytes);
    }

    /// The settings both ranks must share (everything but rank, device and timeouts), as bytes to compare.
    pub fn digest(c: Config) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(std.mem.asBytes(&c.world));
        h.update(std.mem.asBytes(&c.backend));
        h.update(std.mem.asBytes(&c.mailbox_max_bytes));
        h.update(std.mem.asBytes(&c.roce_var_bytes));
        h.update(std.mem.asBytes(&c.roce_chunked_bytes));
        h.update(std.mem.asBytes(&c.failfast));
        h.update(std.mem.asBytes(&c.roce_wire));
        h.update(std.mem.asBytes(&c.roce_fallback));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

/// `ours` when it is set, else prod.env's name for the same setting.
fn alias(get: *const fn ([]const u8) ?[]const u8, ours: []const u8, prod: []const u8) []const u8 {
    return if (get(ours) != null) ours else prod;
}

fn envGet(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const v = std.c.getenv(buf[0..name.len :0]) orelse return null;
    const s = std.mem.span(v);
    return if (s.len == 0) null else s;
}

fn int(comptime T: type, get: *const fn ([]const u8) ?[]const u8, name: []const u8, default: T, lo: T, hi: T) Error!T {
    const raw = get(name) orelse return default;
    const v = std.fmt.parseInt(T, std.mem.trim(u8, raw, " "), 10) catch {
        std.log.warn("{s}={s}: expected an integer", .{ name, raw });
        return error.BadSetting;
    };
    if (v < lo or v > hi) {
        std.log.warn("{s}={d}: expected {d}..{d}", .{ name, v, lo, hi });
        return error.BadSetting;
    }
    return v;
}

fn seconds(get: *const fn ([]const u8) ?[]const u8, name: []const u8, default: f64, lo: f64, hi: f64) Error!u64 {
    const v = if (get(name)) |raw| std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " ")) catch {
        std.log.warn("{s}={s}: expected seconds", .{ name, raw });
        return error.BadSetting;
    } else default;
    if (!(v >= lo and v <= hi)) {
        std.log.warn("{s}={d}: expected {d}..{d} s", .{ name, v, lo, hi });
        return error.BadSetting;
    }
    return @intFromFloat(v * std.time.ns_per_s);
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "defaults follow the Python engine's ports and knobs" {
    Fake.pairs = &.{ .{ "TF_TP_RANK", "1" }, .{ "TF_TP_PORT", "29600" } };
    const c = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(u32, 1), c.rank);
    try std.testing.expectEqual(@as(u16, 29607), c.link_port);
    try std.testing.expectEqual(Backend.nccl, c.backend);
    try std.testing.expectEqual(std.time.ns_per_s, c.grace_ns);
    try std.testing.expectEqual(std.time.ns_per_s / 2, c.report_ns);
}

test "bad settings are refused and the digest ignores the rank" {
    Fake.pairs = &.{.{ "TF_TP_RANK", "2" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{.{ "TF_COMM_BACKEND", "rdma" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_ROCE_HCA", "rocep1s0f1,roceP2p1s0f1" }, .{ "NCCL_IB_TC", "106" } };
    const r = try Config.parse(&Fake.get);
    try std.testing.expectEqual(Backend.roce, r.backend);
    try std.testing.expectEqual(@as(u8, 106), r.roce_tc);
    try std.testing.expectEqualStrings("rocep1s0f1,roceP2p1s0f1", r.roce_hca.?);
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "mailbox" }, .{ "TF_TP_MAILBOX_MAX_KB", "64" }, .{ "TF_DSV41_FAILFAST_GRACE_S", "0.25" } };
    const a = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(usize, 64 * 1024), a.mailbox_max_bytes);
    try std.testing.expectEqual(std.time.ns_per_s / 4, a.grace_ns);
    var b = a;
    b.rank = 1;
    try std.testing.expectEqualSlices(u8, &a.digest(), &b.digest());
    b.mailbox_max_bytes = 1024;
    try std.testing.expect(!std.mem.eql(u8, &a.digest(), &b.digest()));
    try std.testing.expectEqual(@as(usize, 0), a.roce_var_bytes);
    var v = a;
    v.roce_var_bytes = 10 << 20;
    try std.testing.expect(!std.mem.eql(u8, &a.digest(), &v.digest()));
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "TF_TP_ROCE_VAR_KB", "1024" } };
    try std.testing.expectEqual(@as(usize, 1 << 20), (try Config.parse(&Fake.get)).roce_var_bytes);
    Fake.pairs = &.{.{ "TF_TP_ROCE_VAR_KB", "65536" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
}

test "boot refuses any setting that asks for an RDMA write past 1 MiB" {
    for ([_][2][]const u8{ .{ "TF_TP_ROCE_VAR_KB", "10240" }, .{ "TF_TP_ROCE_VAR_KB", "1025" }, .{ "GLM53_TF_ROCE_MAX_KB", "2048" }, .{ "TF_TP_MAILBOX_MAX_KB", "1028" } }) |kv| {
        Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, kv };
        try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    }
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "roce" }, .{ "GLM53_TF_ROCE_MAX_KB", "1024" }, .{ "TF_TP_ROCE_VAR_KB", "1024" } };
    try std.testing.expectEqual(@as(usize, 1 << 20), (try Config.parse(&Fake.get)).roceSlotBytes());
}

test "prod.env's names: GLM53_TF_* where ours are unset, the plan link's spin and pin" {
    Fake.pairs = &.{ .{ "GLM53_TF_COMM_BACKEND", "roce" }, .{ "GLM53_TF_ROCE_MAX_KB", "1024" }, .{ "GLM53_TF_ROCE_FALLBACK", "error" }, .{ "TF_DSV41_PLAN_LINK", "rdma" }, .{ "TF_DSV41_PLAN_PIN", "auto" } };
    const p = try Config.parse(&Fake.get);
    try std.testing.expectEqual(Backend.roce, p.backend);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), p.mailbox_max_bytes);
    try std.testing.expectEqual(Fallback.@"error", p.roce_fallback);
    try std.testing.expectEqual(@as(u64, 5000 * 1000), p.plan_spin_ns);
    try std.testing.expectEqual(PlanPin.auto, p.plan_pin.?);
    // ours win; tcp blocks at once; an explicit CPU; off
    Fake.pairs = &.{ .{ "TF_COMM_BACKEND", "nccl" }, .{ "GLM53_TF_COMM_BACKEND", "roce" }, .{ "TF_TP_MAILBOX_MAX_KB", "64" }, .{ "GLM53_TF_ROCE_MAX_KB", "1024" }, .{ "TF_DSV41_PLAN_LINK", "tcp" }, .{ "TF_DSV41_PLAN_PIN", "7" } };
    const o = try Config.parse(&Fake.get);
    try std.testing.expectEqual(Backend.nccl, o.backend);
    try std.testing.expectEqual(@as(usize, 64 * 1024), o.mailbox_max_bytes);
    try std.testing.expectEqual(@as(u64, 0), o.plan_spin_ns);
    try std.testing.expectEqual(PlanPin{ .cpu = 7 }, o.plan_pin.?);
    Fake.pairs = &.{ .{ "TF_DSV41_PLAN_LINK", "rdma" }, .{ "TF_DSV41_PLAN_SPIN_US", "200" }, .{ "TF_DSV41_PLAN_PIN", "-1" } };
    const s = try Config.parse(&Fake.get);
    try std.testing.expectEqual(@as(u64, 200 * 1000), s.plan_spin_ns);
    try std.testing.expect(s.plan_pin == null);
    Fake.pairs = &.{.{ "TF_DSV41_PLAN_LINK", "udp" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    // the RoCE grid cap: 16 unset, Python's name, ours over it, powers of two only
    Fake.pairs = &.{};
    try std.testing.expectEqual(@as(u32, 16), (try Config.parse(&Fake.get)).roce_blocks);
    Fake.pairs = &.{.{ "GLM53_TF_ROCE_BLOCKS", "8" }};
    try std.testing.expectEqual(@as(u32, 8), (try Config.parse(&Fake.get)).roce_blocks);
    Fake.pairs = &.{ .{ "TF_TP_ROCE_BLOCKS", "4" }, .{ "GLM53_TF_ROCE_BLOCKS", "8" } };
    try std.testing.expectEqual(@as(u32, 4), (try Config.parse(&Fake.get)).roce_blocks);
    Fake.pairs = &.{.{ "TF_TP_ROCE_BLOCKS", "12" }};
    try std.testing.expectError(error.BadSetting, Config.parse(&Fake.get));
    Fake.pairs = &.{};
}
