//! Split KV's Exchange on RoCE's protocol at prod's settings (hybrid routes, 2 stripes): each post inside slot and MR.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const tp = @import("tp");
const split = @import("split.zig");
const roce = tp.roce;
const rv = tp.roce_verbs;
const hybrid = tp.hybrid;

const page = 256;
const pts = 16;
const rb = 584; // the comp rows' bytes on the model
const K = 512;
const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = rb, .split = true },
    .{ .name = "comp.1", .ratio = 1, .row_bytes = rb, .split = true },
};
const slot_lens = [_]u64{ 15 * page + 7, 9 * page, 13 * page + 100 };

/// The serving profile as the launcher passes it (ZIG_COMM_BACKEND -> TF_COMM_BACKEND); TF_TP_ROCE_VAR_KB unset.
fn prodEnv(name: []const u8) ?[]const u8 {
    const table = .{ .{ "TF_TP_WORLD", "2" }, .{ "TF_COMM_BACKEND", "roce" }, .{ "GLM53_TF_ROCE_MAX_KB", "1024" }, .{ "GLM53_TF_ROCE_FAST", "1" } };
    inline for (table) |kv| if (std.mem.eql(u8, name, kv[0])) return kv[1];
    return null;
}

/// Prod's settings plus chunked gathers: strides past 1 MiB, up to 20 MiB, go RoCE as <= 1 MiB ops.
fn chunkEnv(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_TP_ROCE_CHUNKED_KB")) return "20480";
    return prodEnv(name);
}

/// The faulting run's settings: the profile's plus TF_TP_ROCE_VAR_KB=10240 (boot must refuse it).
fn varEnv(name: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_TP_ROCE_VAR_KB")) return "10240";
    return prodEnv(name);
}

/// What one rank's posts did: count, the largest, and ranges a post would have taken outside its slot or the MR.
const Seen = struct {
    posts: u64 = 0,
    max_padded: u32 = 0,
    outside: u64 = 0,
    mr_len: u64,
    slot: u64,

    fn seen(ctx: *anyopaque, l: rv.Layout, slot: u32, padded: u32, s: []const rv.Stripe) void {
        const v: *Seen = @ptrCast(@alignCast(ctx));
        v.posts += 1;
        v.max_padded = @max(v.max_padded, padded);
        // the expected layout from roce.zig's region size, not from the wire's own numbers
        if (l.len != v.mr_len or l.slot != v.slot or l.recv != l.send + 2 * v.slot or l.send + 4 * v.slot != v.mr_len) v.outside += 1;
        var total: u64 = 0;
        for (s) |x| {
            const lo_send = l.send + slot * v.slot;
            const lo_recv = l.recv + slot * v.slot;
            if (x.local < lo_send or x.local + x.len > lo_send + v.slot or x.local + x.len > v.mr_len) v.outside += 1;
            if (x.remote < lo_recv or x.remote + x.len > lo_recv + v.slot or x.remote + x.len > v.mr_len) v.outside += 1;
            if (x.flag < l.flags or x.flag + 4 > l.send) v.outside += 1;
            total += x.len;
        }
        if (total != padded) v.outside += 1;
    }
};

/// A Session's `roce` collective on host ranks: hybrid.zig's routes pick RoCE (host side) or NCCL (host collective).
const Routed = struct {
    nccl: tp.Collective,
    r: *roce.Roce,
    cfg: tp.Config,
    roce_ops: u64 = 0,
    nccl_ops: u64 = 0,
    big_roce: u64 = 0,
    chunked: u64 = 0,

    fn self_(p: *anyopaque) *Routed {
        return @ptrCast(@alignCast(p));
    }
    fn rankOf(p: *anyopaque) u32 {
        return self_(p).nccl.rank();
    }
    fn worldOf(p: *anyopaque) u32 {
        return self_(p).nccl.world();
    }
    fn allReduce(p: *anyopaque, s: u64, r: u64, n: usize, t: tp.DType, op: tp.Op, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.allReduce(s, r, n, t, op, st);
    }
    fn took(h: *Routed, n: usize) void {
        h.roce_ops += 1;
        if (n > 1 << 20) h.big_roce += 1;
    }
    fn allGather(p: *anyopaque, s: u64, r: u64, count: usize, t: tp.DType, st: tp.collective.Stream) tp.collective.Error!void {
        const h = self_(p);
        const n = count * t.size();
        if (!hybrid.smallRoute(h.cfg.mailbox_max_bytes, n, h.r.fits(n))) {
            h.nccl_ops += 1;
            return h.nccl.allGather(s, r, count, t, st);
        }
        h.took(n);
        const in: [*]const u8 = @ptrFromInt(s);
        const out: [*]u8 = @ptrFromInt(r);
        h.r.hostGather(in[0..n], out[0 .. 2 * n]) catch return error.BackendFailed;
    }
    fn single(h: *Routed, n: usize) bool {
        return hybrid.varRoute(h.cfg.mailbox_max_bytes, h.cfg.roce_var_bytes, n, h.r.fits(n));
    }
    fn varFits(p: *anyopaque, n: usize) bool {
        const h = self_(p);
        const one = h.single(n);
        return one or hybrid.chunkRoute(h.cfg.roce_chunked_bytes, n, one, h.r.chunk_abi);
    }
    fn varAgreed(p: *anyopaque) tp.collective.Error!bool {
        return varFits(p, 16);
    }
    fn allGatherV(p: *anyopaque, s: u64, r: u64, stride: usize, lens: u64, t: tp.DType, st: tp.collective.Stream) tp.collective.Error!void {
        const h = self_(p);
        const n = stride * t.size();
        if (!varFits(p, n)) return allGather(p, s, r, stride, t, st);
        const l: [*]const i32 = @ptrFromInt(lens);
        const in: [*]const u8 = @ptrFromInt(s);
        const out: [*]u8 = @ptrFromInt(r);
        if (!h.single(n)) {
            h.roce_ops += 1;
            h.chunked += 1;
            return h.r.hostGatherVChunked(in[0..n], out[0 .. 2 * n], .{ @intCast(l[0]), @intCast(l[1]) }) catch error.BackendFailed;
        }
        h.took(n);
        h.r.hostGatherV(in[0..n], out[0 .. 2 * n], .{ @intCast(l[0]), @intCast(l[1]) }) catch return error.BackendFailed;
    }
    fn exchange(p: *anyopaque, s: u64, r: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.exchange(s, r, n, t, peer, st);
    }
    fn send(p: *anyopaque, b: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.send(b, n, t, peer, st);
    }
    fn recv(p: *anyopaque, b: u64, n: usize, t: tp.DType, peer: u32, st: tp.collective.Stream) tp.collective.Error!void {
        return self_(p).nccl.recv(b, n, t, peer, st);
    }
    fn barrier(p: *anyopaque) tp.collective.Error!void {
        return self_(p).nccl.barrier();
    }
    fn check(p: *anyopaque) tp.collective.Error!void {
        self_(p).r.check() catch |e| return e;
        return self_(p).nccl.check();
    }
    fn abort(p: *anyopaque) void {
        self_(p).r.abort();
        self_(p).nccl.abort();
    }

    const vtable: tp.collective.Collective.VTable = .{ .kind = .hybrid, .rank = rankOf, .world = worldOf, .all_reduce = allReduce, .all_gather = allGather, .exchange = exchange, .send = send, .recv = recv, .barrier = barrier, .check = check, .abort = abort, .all_gather_v = allGatherV, .var_fits = varFits, .var_agreed = varAgreed };

    fn iface(h: *Routed) tp.Collective {
        return .{ .ptr = h, .vtable = &vtable };
    }
};

fn value(f: usize, s: usize, t: usize, i: usize) u8 {
    return @truncate(f *% 101 +% s *% 61 +% t *% 7 +% i *% 3 +% 1);
}

const Ctx = struct {
    get: *const fn ([]const u8) ?[]const u8,
    port: u16,
    lfd: tp.sock.Fd,
    seen: [2]Seen = undefined,
    big_roce: [2]u64 = .{ 0, 0 },
    roce_ops: [2]u64 = .{ 0, 0 },
    chunked: [2]u64 = .{ 0, 0 },
};

fn rank(comm: tp.Collective, me: u32, ctx: *Ctx) anyerror!void {
    const gpa = testing.allocator;
    const cfg = try tp.Config.parse(ctx.get);
    try testing.expectEqual(tp.config.Backend.roce, cfg.backend);
    var b = if (me == 0) try tp.Bootstrap.acceptOn(ctx.lfd, 2, 10 * std.time.ns_per_s) else try tp.Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = ctx.port, .connect_timeout_ns = 10 * std.time.ns_per_s });
    defer b.close();
    const slot = std.mem.alignForward(usize, cfg.roceSlotBytes(), 4096);
    ctx.seen[me] = .{ .mr_len = roce.Roce.regionBytes(slot), .slot = slot };
    var tap: roce.PostTap = .{ .ctx = &ctx.seen[me], .seen = Seen.seen };
    const r = try roce.Roce.init(null, &b, "", .{ .wire = .shm, .shm_stripes = 2, .tap = &tap, .max_bytes = cfg.roceSlotBytes(), .timeout_ns = 20 * std.time.ns_per_s });
    defer r.deinit();
    // no chunked op without the ABI marker; the host side then moves `RoceArgs.chunk` ops as the kernel does
    try testing.expectError(error.Unsupported, r.allGatherVChunked(1 << 12, 1 << 24, 2 << 20, 1 << 12, null));
    r.chunk_abi = cfg.roce_chunked_bytes > 0;
    var routed: Routed = .{ .nccl = comm, .r = r, .cfg = cfg };
    const c = routed.iface();

    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 64 * page, 2, me);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    var slots: [3]*sessions.Slot = undefined;
    for (&slots, slot_lens) |*s, n| {
        s.* = try pool.newSlot(pts * page);
        try s.*.ensure(n);
    }
    for (fams, 0..) |f, fi| for (slots, 0..) |sl, si| for (sl.mapped(), 0..) |pg, k| {
        if (!pool.owns(pg)) continue;
        const v = host.pageOf(@intCast(fi), pool.localPage(pg));
        const per = page / f.ratio;
        for (0..per) |row| for (0..rb) |i| {
            v[row * rb + i] = value(fi, si, k * per + row, i);
        };
    };
    var tables: [3 * pts]i32 = undefined;
    for (slots, 0..) |s, si| for (0..pts) |k| {
        tables[si * pts + k] = @intCast(s.localTableAt(@intCast(k)));
    };
    var hk: split.HostKernels = .{};
    const x = try split.Exchange.init(c, hk.kernels(), .on);
    try testing.expect(x.var_ok);
    var prng = std.Random.DefaultPrng.init(42); // the same selections on both ranks
    const rnd = prng.random();

    // windows of R rows (ROWS_CAP 64), both families: R 3 is 897,024 B a stride (RoCE), R 4 1,196,032 (past 1 MiB)
    for (fams, 0..) |f, fi| for ([_]u32{ 1, 2, 3, 4, 6, 16, 32, 64 }) |R| {
        const n = R * K;
        const sel = try gpa.alloc(i32, n);
        defer gpa.free(sel);
        const rslot = try gpa.alloc(i32, R);
        defer gpa.free(rslot);
        for (rslot, 0..) |*s, i| s.* = @intCast(i % 3);
        for (sel, 0..) |*t, i| {
            const rows: u32 = @intCast(slot_lens[@intCast(rslot[i / K])] / f.ratio);
            t.* = if (rnd.uintLessThan(u32, 9) == 0) -1 else @intCast(rnd.uintLessThan(u32, rows));
        }
        const send = try gpa.alloc(u8, n * rb);
        defer gpa.free(send);
        const recv = try gpa.alloc(u8, 2 * n * rb);
        defer gpa.free(recv);
        const tok = try gpa.alloc(i32, n);
        defer gpa.free(tok);
        const got = try gpa.alloc(u8, n * rb);
        defer gpa.free(got);
        var lens: [split.max_world]i32 = @splat(-7);
        const psh: u32 = std.math.log2_int(u32, page / f.ratio);
        const da: split.DenseArgs = .{ .sel = @intFromPtr(sel.ptr), .rows = R, .k = K, .table = @intFromPtr(&tables), .pts = pts, .rslot = @intFromPtr(rslot.ptr), .psh = psh, .base = @intFromPtr(host.tensors[fi].ptr), .row_bytes = rb, .world = 2, .send = @intFromPtr(send.ptr), .tok = @intFromPtr(tok.ptr) };
        for (0..2) |pass| {
            @memset(recv, 0xEE);
            if (pass == 0) try x.window(split.PackArgs.of(da, me, @intFromPtr(&lens)), @intFromPtr(recv.ptr), null) else try x.dense(da, @intFromPtr(recv.ptr), null);
            for (sel, tok, 0..) |t, o, i| {
                if (t < 0) continue;
                const s: usize = @intCast(rslot[i / K]);
                const row = recv[@as(usize, @intCast(o)) * rb ..][0..rb];
                for (row, 0..) |byte, j| if (byte != value(fi, s, @intCast(t), j)) {
                    std.debug.print("fam {d} R {d} pass {d}: entry {d} wrong\n", .{ fi, R, pass, i });
                    return error.TestUnexpectedResult;
                };
                // the window's (packed, chunked past 1 MiB) row byte-identical to the dense (NCCL) gather's
                if (pass == 0) @memcpy(got[i * rb ..][0..rb], row) else try testing.expectEqualSlices(u8, got[i * rb ..][0..rb], row);
            }
        }
    };

    // union blocks (allGather of M rows): the 1 MiB edge (1,795 rows = 1,048,280 B) and the 128 MiB cap's block
    const max_m: u32 = (128 << 20) / (3 * rb);
    const rows_buf = try gpa.alloc(u8, @as(usize, max_m) * rb);
    defer gpa.free(rows_buf);
    for (rows_buf, 0..) |*v, i| v.* = @truncate(i *% 13 +% me);
    const phys = try gpa.alloc(u32, max_m);
    defer gpa.free(phys);
    for (phys, 0..) |*p, i| p.* = @intCast(max_m - 1 - i);
    for ([_]u32{ 1, 1795, 1796, max_m }) |m| {
        const send = try gpa.alloc(u8, @as(usize, m) * rb);
        defer gpa.free(send);
        const recv = try gpa.alloc(u8, 2 * @as(usize, m) * rb);
        defer gpa.free(recv);
        try x.unionBlock(@intFromPtr(rows_buf.ptr), rb, @intFromPtr(phys.ptr), m, @intFromPtr(send.ptr), @intFromPtr(recv.ptr), null);
        for (0..2) |o| for (0..m) |i| {
            const want: u8 = @truncate(@as(usize, phys[i]) * rb *% 13 +% o);
            if (recv[(o * m + i) * rb] != want) return error.TestUnexpectedResult;
        };
    }
    try c.check();
    try b.agree("done", "split roce test end");
    ctx.big_roce[me] = routed.big_roce;
    ctx.roce_ops[me] = routed.roce_ops;
    ctx.chunked[me] = routed.chunked;
}

fn runAt(get: *const fn ([]const u8) ?[]const u8) !Ctx {
    const l = try tp.bootstrap.testListener();
    defer tp.sock.close(l.fd);
    var ctx: Ctx = .{ .get = get, .port = l.port, .lfd = l.fd };
    try tp.host.run(2, &ctx, rank);
    return ctx;
}

test "prod's settings: every split exchange's RoCE post fits the 1 MiB slot and its MR; past 1 MiB goes to NCCL" {
    const ctx = try runAt(prodEnv);
    for (ctx.seen, ctx.big_roce, ctx.roce_ops) |s, big, ops| {
        try testing.expectEqual(@as(u64, 1 << 20), s.slot);
        try testing.expect(ops > 0 and s.posts >= ops);
        try testing.expectEqual(@as(u64, 0), s.outside);
        try testing.expect(s.max_padded <= 1 << 20);
        try testing.expectEqual(@as(u64, 0), big);
    }
}

test "TF_TP_ROCE_CHUNKED_KB: strides past 1 MiB go RoCE as <= 1 MiB writes, byte-identical to the dense gather" {
    // one op's strides stay single; off (0), past the bound or without the kernel's ABI marker: NCCL
    try testing.expect(!hybrid.chunkRoute(20 << 20, 2 << 20, true, true));
    try testing.expect(!hybrid.chunkRoute(0, 2 << 20, false, true));
    try testing.expect(!hybrid.chunkRoute(20 << 20, (20 << 20) + 1, false, true));
    try testing.expect(!hybrid.chunkRoute(20 << 20, 2 << 20, false, false));
    try testing.expect(hybrid.chunkRoute(20 << 20, 20 << 20, false, true));
    const ctx = try runAt(chunkEnv);
    for (ctx.seen, ctx.big_roce, ctx.chunked) |s, big, chunked| {
        try testing.expectEqual(@as(u64, 1 << 20), s.slot);
        // R 4..64 windows of both families (fam 1 R 32: the faulting run's 9,568,256 B stride; R 64: 19,136,512 B)
        try testing.expectEqual(@as(u64, 10), chunked);
        try testing.expectEqual(@as(u64, 0), s.outside);
        try testing.expect(s.max_padded <= 1 << 20);
        try testing.expectEqual(@as(u64, 0), big);
    }
}

test "TF_TP_ROCE_VAR_KB=10240 (the faulting run's setting) fails boot; a RoCE slot past the write cap is refused too" {
    try testing.expectError(error.BadSetting, tp.Config.parse(varEnv));
    const l = try tp.bootstrap.testListener();
    defer tp.sock.close(l.fd);
    var ctx: Big = .{ .port = l.port, .lfd = l.fd };
    try tp.host.run(2, &ctx, bigSlot);
    for (ctx.errs) |e| try testing.expectEqual(@as(?anyerror, error.WriteTooLarge), e);
}

const Big = struct { port: u16, lfd: tp.sock.Fd, errs: [2]?anyerror = .{ null, null } };

fn bigSlot(_: tp.Collective, me: u32, ctx: *Big) anyerror!void {
    var b = if (me == 0) try tp.Bootstrap.acceptOn(ctx.lfd, 2, 10 * std.time.ns_per_s) else try tp.Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = ctx.port, .connect_timeout_ns = 10 * std.time.ns_per_s });
    defer b.close();
    if (roce.Roce.init(null, &b, "", .{ .wire = .shm, .max_bytes = (1 << 20) + 4096 })) |r| r.deinit() else |e| {
        ctx.errs[me] = e;
    }
}

test "the RoCE post guard: no write past 1 MiB whatever the slot; oversize, misaligned or foreign-slot posts refused before a verb" {
    const big_slot: u64 = 10240 << 10;
    const l: rv.Layout = .{ .send = 8192, .recv = 8192 + 2 * big_slot, .flags = 4096, .slot = big_slot, .len = roce.Roce.regionBytes(big_slot) };
    var out: [rv.max_hcas]rv.Stripe = undefined;
    // fam 1, R 32's 9,568,256 B stride (the faulting one) and 1 MiB + 16: refused although the slot holds them
    try testing.expectError(error.WriteTooLarge, rv.stripes(l, 2, 1, 32 * K * rb, &out));
    try testing.expectError(error.WriteTooLarge, rv.stripes(l, 2, 0, (1 << 20) + 16, &out));
    try rv.stripes(l, 2, 1, 1 << 20, &out);
    try testing.expectEqual(@as(u64, l.send + big_slot), out[0].local);
    try testing.expectEqual(l.recv + big_slot + (1 << 20), out[1].remote + out[1].len);
    // a 512 KiB slot: slot 0 past its end stays inside the MR but writes slot 1; only the slot bounds refuse it
    const s5: u64 = 512 << 10;
    const m: rv.Layout = .{ .send = 8192, .recv = 8192 + 2 * s5, .flags = 4096, .slot = s5, .len = roce.Roce.regionBytes(s5) };
    try testing.expectError(error.OutOfRegion, rv.stripes(m, 2, 0, @intCast(s5 + 16), &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(m, 2, 1, @intCast(s5 + 16), &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(m, 2, 0, 24, &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(m, 2, 2, 16, &out));
    try testing.expectError(error.OutOfRegion, rv.stripes(m, 3, 0, 16, &out));
    // a layout whose registered length is short of its slots: the post refused, not posted past the MR
    var short = m;
    short.len -= 16;
    try testing.expectError(error.OutOfRegion, rv.stripes(short, 2, 1, @intCast(s5), &out));
    // a peer whose slot or region differs is refused at connect
    const mine: rv.Record = .{ .n_hca = 1, .slot_bytes = s5, .region_len = m.len };
    var peer = mine;
    peer.slot_bytes = 1 << 20;
    try testing.expectError(error.BadRecord, rv.VerbsWire.pairing(mine, peer));
}

const Keep = struct { port: u16, lfd: tp.sock.Fd, fail: bool, region: [2]usize = .{ 0, 0 }, len: usize = 0 };

fn keepRank(_: tp.Collective, me: u32, ctx: *Keep) anyerror!void {
    var b = if (me == 0) try tp.Bootstrap.acceptOn(ctx.lfd, 2, 10 * std.time.ns_per_s) else try tp.Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = ctx.port, .connect_timeout_ns = 10 * std.time.ns_per_s });
    defer b.close();
    const r = try roce.Roce.init(null, &b, "", .{ .wire = .shm, .max_bytes = 1 << 20 });
    ctx.region[me] = @intFromPtr(r.mapping.ptr);
    ctx.len = r.mapping.len;
    try b.agree("open", "keep test open");
    if (ctx.fail and me == 1) r.abort();
    r.deinit();
}

/// Whether [at, at + len) is still mapped in this process.
fn mapped(at: usize, len: usize) bool {
    var vec: [4096]u8 = undefined;
    const rc = std.os.linux.mincore(@ptrFromInt(at), @min(len, 4096 * 4096), &vec);
    return std.os.linux.errno(rc) == .SUCCESS;
}

test "deinit frees a healthy transport's region and keeps a failed one's registered and mapped until exit" {
    inline for (.{ false, true }) |fail| {
        const l = try tp.bootstrap.testListener();
        defer tp.sock.close(l.fd);
        var ctx: Keep = .{ .port = l.port, .lfd = l.fd, .fail = fail };
        const kept0 = roce.Roce.kept.load(.monotonic);
        try tp.host.run(2, &ctx, keepRank);
        const kept = roce.Roce.kept.load(.monotonic) - kept0;
        try testing.expectEqual(@as(u32, if (fail) 1 else 0), kept);
        try testing.expectEqual(fail, mapped(ctx.region[1], ctx.len));
        try testing.expect(!mapped(ctx.region[0], ctx.len));
    }
}
