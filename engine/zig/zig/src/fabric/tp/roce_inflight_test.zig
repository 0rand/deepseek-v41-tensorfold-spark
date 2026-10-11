//! VerbsWire.post on fake verbs whose completions lag: bytes in flight stay at most 2 MiB, every write at most 1 MiB.
const std = @import("std");
const verbs = @import("verbs");
const abi = verbs.abi;
const rv = @import("roce_verbs.zig");
const Roce = @import("roce.zig").Roce;

const slot: u64 = 1 << 20;
const send_off = 8192;

/// The fake provider: what the HCAs hold (posted, not completed), counted from the requests themselves.
const Fake = struct {
    var region: []const u8 = &.{};
    var lag: u32 = 1;
    var polls: u64 = 0;
    var queue: [2][512]u64 = undefined;
    var head: [2]u32 = .{ 0, 0 };
    var len: [2]u32 = .{ 0, 0 };
    var inflight: u64 = 0;
    var max_inflight: u64 = 0;
    var max_write: u64 = 0;
    var outside: u64 = 0;
    var bytes: u64 = 0;

    fn reset(l: u32) void {
        lag = l;
        polls = 0;
        head = .{ 0, 0 };
        len = .{ 0, 0 };
        inflight = 0;
        max_inflight = 0;
        max_write = 0;
        outside = 0;
        bytes = 0;
    }

    fn postSend(qp: *abi.Qp, wr: *abi.SendWr, _: *?*abi.SendWr) callconv(.c) c_int {
        const q = qp.qp_num;
        var sum: u64 = 0;
        var at: ?*abi.SendWr = wr;
        while (at) |w| : (at = w.next) {
            if (w.send_flags & abi.send_inline == 0) {
                const sge = w.sg_list[0];
                const lo = @intFromPtr(region.ptr);
                if (sge.addr < lo + send_off or sge.addr + sge.length > lo + send_off + 2 * slot) outside += 1;
                max_write = @max(max_write, sge.length);
                sum += sge.length;
            }
            if (w.send_flags & abi.send_signaled != 0) {
                queue[q][(head[q] + len[q]) % 512] = sum;
                len[q] += 1;
                inflight += sum;
                bytes += sum;
                max_inflight = @max(max_inflight, inflight);
                sum = 0;
            }
        }
        return 0;
    }

    /// One completion every `lag` polls of a queue pair's CQ: the HCA finishes writes far slower than they are posted.
    fn pollCq(cq: *abi.Cq, _: c_int, wc: [*]abi.Wc) callconv(.c) c_int {
        const q = cq.handle;
        polls += 1;
        if (polls % lag != 0 or len[q] == 0) return 0;
        inflight -= queue[q][head[q]];
        head[q] = (head[q] + 1) % 512;
        len[q] -= 1;
        wc[0] = std.mem.zeroes(abi.Wc);
        wc[0].status = abi.wc_success;
        return 1;
    }
};

const Hw = struct {
    ctx: [2]abi.Context = undefined,
    qp: [2]abi.Qp = undefined,
    cq: [2]abi.Cq = undefined,
    mr: [2]abi.Mr = undefined,

    fn wire(hw: *Hw, region: []u8) rv.VerbsWire {
        var w: rv.VerbsWire = .{ .v = undefined, .traffic_class = 0, .recv_off = send_off + 2 * slot, .flag_off = 4096, .send_off = send_off, .slot_bytes = slot, .region = region };
        for (0..2) |i| {
            hw.ctx[i].ops = .{ ._before_poll = @splat(null), .poll_cq = Fake.pollCq, ._before_post = @splat(null), .post_send = Fake.postSend, .post_recv = undefined, ._after = @splat(null) };
            hw.qp[i].qp_num = @intCast(i);
            hw.cq[i].handle = @intCast(i);
            hw.mr[i].lkey = 7;
            w.hcas[i] = .{ .ctx = &hw.ctx[i], .pd = undefined, .mr = &hw.mr[i], .cq = &hw.cq[i], .qp = &hw.qp[i], .gid_index = 0, .gid = @splat(0), .mtu = 0, .lid = 0 };
        }
        w.n_hca = 2;
        return w;
    }
};

// Every chunked stride of split_roce_test (R x 512 x 584, union blocks) posted back to back in <= 1 MiB ops.
test "chunked posts: at most 2 MiB in flight over both HCAs and no write past 1 MiB, the faulting stride included" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    var strides: [24]u64 = undefined;
    var n: usize = 0;
    for ([_]u64{ 1, 2, 3, 4, 6, 16, 32, 64 }) |r| {
        strides[n] = r * 512 * 584;
        n += 1;
    }
    for ([_]u64{ 1795, 1796, (128 << 20) / (3 * 584) }) |m| {
        strides[n] = m * 584;
        n += 1;
    }
    for ([_]u32{ 1, 3, 17 }) |lag| {
        Fake.reset(lag);
        var hw: Hw = .{};
        var w = hw.wire(region);
        var seq: u32 = 0;
        var want: u64 = 0;
        for (strides[0..n]) |stride| {
            var off: u64 = 0;
            while (off < stride) : (off += slot) {
                const part = @min(slot, stride - off);
                seq += 1;
                const padded: u32 = @intCast(std.mem.alignForward(u64, part, 16));
                try w.post(seq & 1, seq, padded);
                want += padded;
            }
        }
        while (w.inflight() > 0) try w.reap();
        try std.testing.expectEqual(want, Fake.bytes);
        try std.testing.expectEqual(@as(u64, 0), Fake.inflight);
        try std.testing.expectEqual(@as(u64, 0), Fake.outside);
        // the ruling's numbers, not roce_verbs.zig's constants: 1 MiB a write, 2 MiB in flight
        try std.testing.expect(Fake.max_write <= 1 << 20);
        try std.testing.expect(Fake.max_inflight <= 2 << 20);
        // the bound is reached, not avoided: posts outran completions
        try std.testing.expect(Fake.max_inflight > 1 << 20);
        try std.testing.expectEqual(Fake.max_inflight, w.peak.load(.monotonic));
    }
}

const Halts = struct {
    var halted: [2]bool = .{ false, false };
    var flushed_before_halt: u32 = 0;
    var flushes: u32 = 0;

    fn modifyQp(qp: *abi.Qp, attr: *abi.QpAttr, mask: c_int) callconv(.c) c_int {
        if (attr.qp_state == abi.qps_err and mask == abi.mask.state) halted[qp.qp_num] = true;
        return 0;
    }

    /// ibv_query_qp as the device answers it: ERR once halted, else RTS.
    fn queryQp(qp: *abi.Qp, attr: *abi.QpAttr, _: c_int, _: *abi.QpInitAttr) callconv(.c) c_int {
        attr.qp_state = if (halted[qp.qp_num]) abi.qps_err else abi.qps_rts;
        return 0;
    }

    /// No request completes on its own; a halted QP flushes what it holds (error status, as the HCA reports it).
    fn pollCq(cq: *abi.Cq, _: c_int, wc: [*]abi.Wc) callconv(.c) c_int {
        const q = cq.handle;
        if (Fake.len[q] == 0 or !halted[q]) return 0;
        if (!halted[0] or !halted[1]) flushed_before_halt += 1;
        Fake.inflight -= Fake.queue[q][Fake.head[q]];
        Fake.head[q] = (Fake.head[q] + 1) % 512;
        Fake.len[q] -= 1;
        flushes += 1;
        wc[0] = std.mem.zeroes(abi.Wc);
        wc[0].status = 5; // IBV_WC_WR_FLUSH_ERR
        return 1;
    }
};

test "a failed transport: the QPs go to error and every posted write is flushed before teardown; abort halts" {
    const region = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(4096), Roce.regionBytes(slot));
    defer std.testing.allocator.free(region);
    Fake.region = region;
    Fake.reset(1);
    Halts.halted = .{ false, false };
    Halts.flushed_before_halt = 0;
    Halts.flushes = 0;
    var hw: Hw = .{};
    var w = hw.wire(region);
    w.v.api.ibv_modify_qp = Halts.modifyQp;
    w.v.query_qp = Halts.queryQp;
    for (&hw.ctx) |*c| c.ops.poll_cq = Halts.pollCq;
    // two 1 MiB ops posted, neither completed: 2 MiB in flight at the HCAs, the link busy
    try std.testing.expect(w.idle());
    try w.post(1, 1, slot);
    try w.post(0, 2, slot);
    try std.testing.expectEqual(@as(u64, 2 * slot), w.inflight());
    try std.testing.expect(!w.idle());
    try std.testing.expect(w.quiesce(std.time.ns_per_s));
    try std.testing.expect(Halts.halted[0] and Halts.halted[1]);
    try std.testing.expectEqual(@as(u32, 0), Halts.flushed_before_halt);
    try std.testing.expectEqual(@as(u32, 4), Halts.flushes);
    try std.testing.expectEqual(@as(u64, 0), w.inflight());
    try std.testing.expectEqual(@as(u64, 0), Fake.inflight);
    // the line tf-tp-test's transport run reads: the device's states and every reap, all of them flushes
    var line: [256]u8 = undefined;
    try std.testing.expect(w.idle());
    try std.testing.expectEqualStrings("HCA 0 QP ERR; posted 2, completed 2; HCA 1 QP ERR; posted 2, completed 2; reaped 4 (flushed 4), still posted 0", w.report(&line));
    // Roce.abort (the fate hook) halts the HCAs too
    Halts.halted = .{ false, false };
    @memset(region, 0);
    var r: Roce = .{ .me = 0, .slot_bytes = slot, .timeout_ns = 1, .region = region, .mapping = region, .wire = .{ .verbs = w }, .cpu = null };
    r.abort();
    try std.testing.expect(Halts.halted[0] and Halts.halted[1]);
    try std.testing.expect(r.failed());
}

test "a QP whose inline capacity cannot take the 4-byte flag post is refused" {
    try std.testing.expect(!rv.inlineFits(.{ .max_send_wr = 1, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 3 }));
    try std.testing.expect(rv.inlineFits(.{ .max_send_wr = 1, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 4 }));
}

/// /proc/self/smaps' VmFlags of the mapping that starts at `at`.
fn vmFlags(at: usize, buf: []u8) ![]const u8 {
    const fd = std.c.open("/proc/self/smaps", .{});
    if (fd < 0) return error.NoSmaps;
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    // the mapping holding `at`: once its flags match a neighbour's, the kernel may merge the two, so it can start below
    var lines = std.mem.splitScalar(u8, buf[0..n], '\n');
    var inside = false;
    while (lines.next()) |l| {
        const sp = std.mem.indexOfScalar(u8, l, ' ') orelse l.len;
        if (std.mem.indexOfScalar(u8, l[0..sp], '-')) |dash| {
            const lo = std.fmt.parseInt(usize, l[0..dash], 16) catch continue;
            const hi = std.fmt.parseInt(usize, l[dash + 1 .. sp], 16) catch continue;
            inside = lo <= at and at < hi;
        }
        if (inside and std.mem.startsWith(u8, l, "VmFlags:")) return l;
    }
    return error.NotFound;
}

test "the region is DONTFORK and NOHUGEPAGE before registration" {
    const len = Roce.regionBytes(slot);
    const m = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(m);
    const buf = try std.testing.allocator.alloc(u8, 4 << 20);
    defer std.testing.allocator.free(buf);
    // smaps: dc = MADV_DONTFORK, nh = MADV_NOHUGEPAGE
    const before = try vmFlags(@intFromPtr(m.ptr), buf);
    try std.testing.expect(std.mem.indexOf(u8, before, " dc") == null and std.mem.indexOf(u8, before, " nh") == null);
    try rv.prepareRegion(m);
    const flags = try vmFlags(@intFromPtr(m.ptr), buf);
    try std.testing.expect(std.mem.indexOf(u8, flags, " dc") != null);
    try std.testing.expect(std.mem.indexOf(u8, flags, " nh") != null);
}
