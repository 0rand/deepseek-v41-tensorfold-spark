//! `tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host`: one rank of a TP=2 GPU test.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const tp_kernel = @import("tp_kernel");
const checks = @import("gpu_checks.zig");
const bench = @import("gpu_bench.zig");

const max_bytes = (4 << 20) + 4096;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host\n", .{});
        return 2;
    }
    const cfg = try tp.Config.fromEnv();
    if (std.mem.eql(u8, args[1], "roce-host")) return @import("roce_host.zig").run(cfg, init.gpa);
    if (std.mem.eql(u8, args[1], "failfast") and (!cfg.failfast or args.len > 5) and !hostOnlyAllowed(cfg)) {
        std.debug.print("REFUSED: failfast with TF_DSV41_FAILFAST=0 or a hold (T2) is host-only: TF_TP_ALLOW_T2_DEVBOX=1 and no RDMA wire\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    const t0 = tp.sock.nowNs();
    var s: tp.Session = undefined;
    try s.open(&driver, cfg, tp_kernel.mailbox);
    const up_ms = @as(f64, @floatFromInt(tp.sock.nowNs() - t0)) / 1e6;
    std.debug.print("INFO rank {d}/{d} on {s} (sm_{d}), backend {t}, mailbox kernel {d} B, bootstrap + NCCL init {d:.0} ms\n", .{ cfg.rank, cfg.world, try ctx.name(&name_buf), try ctx.capability(), s.comm().kind(), tp_kernel.mailbox.len, up_ms });
    if (std.mem.eql(u8, args[1], "suite")) return suite(&s, &driver, init.gpa, cfg);
    if (std.mem.eql(u8, args[1], "failfast") and args.len >= 4 and std.mem.eql(u8, args[3], "transport")) {
        const n = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 50;
        const hold = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 0;
        return transport(&s, &driver, init.gpa, cfg, try std.fmt.parseInt(u32, args[2], 10), n, hold);
    }
    if (std.mem.eql(u8, args[1], "failfast") and args.len == 4) return failfast(&s, &driver, init.gpa, cfg, try std.fmt.parseInt(u32, args[2], 10), args[3]);
    std.debug.print("unknown command {s}\n", .{args[1]});
    return 2;
}

fn suite(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, max_bytes);
    defer c.deinit();
    var backends: [2]struct { comm: tp.Collective, label: []const u8 } = undefined;
    var n_backends: usize = 1;
    backends[0] = .{ .comm = s.nccl.iface(), .label = "nccl" };
    if (s.comm().kind() != .nccl) {
        backends[1] = .{ .comm = s.comm(), .label = if (s.roce != null) "roce" else "mailbox" };
        n_backends = 2;
    }
    const sizes = [_]usize{ 6, 1000, 4096, 65536, 256 * 1024, 256 * 1024 + 16, 1 << 20, 4 << 20 };
    for (backends[0..n_backends]) |b| {
        for (sizes, 0..) |n, i| try c.collectives(b.comm, b.label, n, 100 * i);
        try c.pointToPoint(b.comm, b.label, 64 * 1024, 7);
        for ([_]usize{ 4096, 65536, 1 << 20 }) |n| try c.graphs(b.comm, b.label, n, 4);
        try b.comm.barrier();
    }
    for (backends[0..n_backends]) |b| {
        try bench.sweep(&c, b.comm, b.label, 4096, 4 << 20);
        try bench.decodeWindow(&c, b.comm, b.label, 86, 8192);
    }
    // the stop: rank 0 marks it expected first, then both meet and close
    if (s.fate) |*f| f.expectStop();
    try s.comm().barrier();
    std.debug.print("SUMMARY rank {d}: {d} passed, {d} failed\n", .{ cfg.rank, c.passes, c.failures });
    s.close();
    return if (c.failures == 0) 0 else 1;
}

fn realNow() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}

/// Exchanges until the victim fails; the survivor is left waiting inside one, which only the fate channel ends.
fn failfast(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, victim: u32, mode: []const u8) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, 1 << 20);
    defer c.deinit();
    const comm = s.comm();
    var k: u32 = 0;
    while (true) : (k += 1) {
        if (k == 50 and cfg.rank == victim) {
            std.debug.print("INJECT {s} rank {d} at {d:.6}\n", .{ mode, cfg.rank, realNow() });
            if (std.mem.eql(u8, mode, "kill")) _ = std.c.raise(.KILL);
            s.fatal("window", error.Injected);
            return 3; // not reached: the process exits 70
        }
        try comm.allReduce(c.in.ptr, c.out.ptr, 2048, .bf16, .sum, c.stream.handle);
        c.stream.synchronize() catch |e| {
            std.debug.print("rank {d}: stream error {t} after the peer failed\n", .{ cfg.rank, e });
        };
        comm.check() catch |e| {
            std.debug.print("rank {d}: transport reports {t} at exchange {d}; waiting for the fate channel\n", .{ cfg.rank, e, k });
            while (true) tp.sock.sleepNs(std.time.ns_per_s);
        };
    }
}

/// The failed-transport path with no RDMA fault (N sums, an NCCL meet, the victim's `Roce.inject`, its close).
fn transport(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, victim: u32, n: u32, hold_s: u32) !u8 {
    const r = s.roce orelse {
        std.debug.print("failfast transport needs TF_COMM_BACKEND=roce (the session runs {t})\n", .{s.comm().kind()});
        return 2;
    };
    var c = try checks.Ctx.init(d, gpa, cfg.rank, 4096);
    defer c.deinit();
    const comm = s.comm();
    for (0..n) |_| {
        try comm.allReduce(c.in.ptr, c.out.ptr, 2048, .bf16, .sum, c.stream.handle);
        try c.stream.synchronize();
        try comm.check();
    }
    try s.nccl.iface().barrier();
    std.debug.print("rank {d}: {d} clean exchanges; both ranks met over NCCL\n", .{ cfg.rank, n });
    // the inject only on an idle link: every rank's posts completed, else no rank injects
    const t_idle = tp.sock.nowNs();
    while (!r.linkIdle() and tp.sock.nowNs() - t_idle < 2 * std.time.ns_per_s) tp.sock.sleepNs(std.time.ns_per_ms);
    var line: [384]u8 = undefined;
    std.debug.print("rank {d}: link {s} before the inject: {s}\n", .{ cfg.rank, if (r.linkIdle()) "idle" else "BUSY", r.describeLink(&line) });
    if (try idleRanks(s, &c, r.linkIdle()) != cfg.world) {
        std.debug.print("REFUSED rank {d}: a link was not idle; no inject, a clean stop\n", .{cfg.rank});
        if (s.fate) |*f| f.expectStop();
        try s.nccl.iface().barrier();
        s.close();
        return 2;
    }
    if (cfg.rank == victim) {
        std.debug.print("INJECT transport rank {d} at {d:.6}\n", .{ cfg.rank, realNow() });
        r.inject();
        comm.check() catch |e| std.debug.print("rank {d}: transport reports {t}\n", .{ cfg.rank, e });
    } else if (s.fate != null) {
        while (true) tp.sock.sleepNs(std.time.ns_per_s); // the fate channel ends this rank (exit 70)
    }
    const kept0 = tp.roce.Roce.kept.load(.monotonic);
    s.close(); // fail-fast on: exit 70 from here
    const kept = tp.roce.Roce.kept.load(.monotonic) - kept0;
    const want: u32 = @intFromBool(cfg.rank == victim);
    const live = cfg.rank != victim or mapped(r.region);
    std.debug.print("rank {d}: closed; regions kept {d} (want {d}){s}\n", .{ cfg.rank, kept, want, if (cfg.rank != victim) "" else if (live) ", region still mapped" else ", region NOT mapped" });
    if (hold_s > 0) {
        std.debug.print("HOLD rank {d} pid {d} for {d} s\n", .{ cfg.rank, std.c.getpid(), hold_s });
        tp.sock.sleepNs(@as(u64, hold_s) * std.time.ns_per_s);
    }
    const ok = kept == want and live;
    std.debug.print("{s} rank {d}\n", .{ if (ok) "PASS" else "FAIL", cfg.rank });
    return if (ok) 0 else 1;
}

fn mapped(region: []const u8) bool {
    var vec: [1]u8 = undefined;
    return std.os.linux.errno(std.os.linux.mincore(@constCast(region.ptr), 4096, &vec)) == .SUCCESS;
}

/// T2's forms (fail-fast off, a hold): only with TF_TP_ALLOW_T2_DEVBOX=1 and no RDMA wire (no RoCE, or its shm wire).
fn hostOnlyAllowed(cfg: tp.Config) bool {
    const flag = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_TP_ALLOW_T2_DEVBOX") orelse ""), "1");
    return flag and (cfg.backend != .roce or cfg.roce_wire == .shm);
}

/// The ranks whose link was idle, summed over NCCL (one f32 a rank).
fn idleRanks(s: *tp.Session, c: *checks.Ctx, idle: bool) !u32 {
    var v: [1]f32 = .{@floatFromInt(@intFromBool(idle))};
    try c.in.uploadAsync(0, std.mem.asBytes(&v), c.stream.handle);
    try s.nccl.iface().allReduce(c.in.ptr, c.out.ptr, 1, .f32, .sum, c.stream.handle);
    try c.stream.synchronize();
    try c.out.download(0, std.mem.asBytes(&v));
    return @intFromFloat(v[0]);
}
