//! The RoCE backend between two hosts (the Python engine's roce.py / roce.cpp): the tp_roce kernel stages and rings a doorbell, a proxy thread moves the slot over a wire.
const std = @import("std");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const rv = @import("roce_verbs.zig");
const HostRegister = @import("mailbox.zig").HostRegister;
const posix = std.posix;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Error = collective.Error;

pub const max_blocks = 16;
const threads = 512;
const page = std.heap.page_size_min;
const flag_stride = 128;
// region: ctrl [4 KiB], flags [2 slots][2 HCAs] at 128 B, send [2][slot], recv [2][slot] (mailbox.cu's tp_roce)
const ctrl_off = 0;
const flag_off = 4096;
const send_off = 8192;
// ctrl words (mailbox.cu): doorbell, padded bytes per slot, completed, abort, proxy failure, a failed wait's record
const w_seq = 0;
const w_slot_bytes = 4;
const w_completed = 8;
const w_abort = 12;
const w_proxy_failed = 13;
const w_failed = 16;
/// Model graphs leave sub-millisecond gaps: the proxy stays hot this many idle polls, then naps 20 us at a time.
const idle_spins: u64 = 20_000_000;

const Mode = enum(u32) { gather = 0, sum_f32 = 1, sum_bf16 = 2, exchange = 3 };

/// mailbox.cu's `RoceArgs`, field for field.
const RoceArgs = extern struct {
    in: DevicePtr,
    out: DevicePtr,
    send_slots: DevicePtr,
    recv_slots: DevicePtr,
    flags: DevicePtr,
    ctrl: DevicePtr,
    state: DevicePtr,
    nbytes: u64,
    slot_bytes: u64,
    timeout_ns: u64,
    rank: u32,
    mode: u32,
    n_hca: u32,
    /// GLM53_TF_ROCE_FAST (config.roce_fast; 0: the classic path)
    opts: u32 = 0,
    /// gather only: device int32 [2] of each rank's bytes (0: every shard is `nbytes`)
    lens: DevicePtr = 0,
    /// Chunked gathers: this op's byte offset into each rank's shard (it moves at most a slot from there; 0 otherwise).
    chunk: u64 = 0,
};

pub const WireKind = enum {
    /// RDMA writes over the HCAs (the Sparks' CX7).
    verbs,
    /// The proxy copies into the peer's region in shared memory: the same protocol on one host without RDMA
    /// (tests and the single-GPU gate of the kernel's doorbell path).
    shm,
};

pub const Options = struct {
    wire: WireKind = .verbs,
    hcas: []const rv.HcaSpec = &.{},
    traffic_class: u8 = 0,
    cpu: ?u32 = null,
    max_bytes: usize = 256 * 1024,
    timeout_ns: u64 = 120 * std.time.ns_per_s,
    /// GLM53_TF_ROCE_FAST: tp_roce's `opts` word (config.roce_fast)
    fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS: the largest grid (config.roce_blocks)
    blocks: u32 = 16,
    /// The shm wire's stripes, as HCAs (the Sparks stripe over 2): its posts take the verbs wire's stripes and guard.
    shm_stripes: u32 = 1,
    /// Tests: sees every shm post's stripes before the copy.
    tap: ?*PostTap = null,
};

/// A test's view of each post (shm wire): slot, padded bytes and the stripes the wire is about to move.
pub const PostTap = struct {
    ctx: *anyopaque,
    seen: *const fn (ctx: *anyopaque, l: rv.Layout, slot: u32, padded: u32, s: []const rv.Stripe) void,
};

/// The shm wire: the peer's region is mapped here; a post copies the slot, then stores the flag (x86 / Arm stores
/// after the copy, release).
const ShmWire = struct {
    peer: []u8,
    local: []u8,
    slot_bytes: usize,
    stripes: u32 = 1,
    tap: ?*PostTap = null,

    fn post(w: *ShmWire, slot: u32, seq: u32, padded: u32) !void {
        const l: rv.Layout = .{ .send = send_off, .recv = send_off + 2 * w.slot_bytes, .flags = flag_off, .slot = w.slot_bytes, .len = w.local.len };
        var plan: [rv.max_hcas]rv.Stripe = undefined;
        try rv.stripes(l, w.stripes, slot, padded, &plan);
        if (w.tap) |t| t.seen(t.ctx, l, slot, padded, plan[0..w.stripes]);
        for (plan[0..w.stripes]) |s| {
            @memcpy(w.peer[s.remote..][0..s.len], w.local[s.local..][0..s.len]);
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(w.peer[s.flag..].ptr));
            flag.store(seq, .release);
        }
    }
};

const Wire = union(WireKind) {
    verbs: rv.VerbsWire,
    shm: ShmWire,

    fn post(w: *Wire, slot: u32, seq: u32, padded: u32) !void {
        switch (w.*) {
            .verbs => |*v| try v.post(slot, seq, padded),
            .shm => |*s| try s.post(slot, seq, padded),
        }
    }

    fn reap(w: *Wire) !void {
        switch (w.*) {
            .verbs => |*v| try v.reap(),
            .shm => {},
        }
    }

    fn hcas(w: *const Wire) u32 {
        return switch (w.*) {
            .verbs => |v| v.n_hca,
            .shm => |s| s.stripes,
        };
    }
};

pub const Roce = struct {
    me: u32,
    slot_bytes: usize,
    timeout_ns: u64,
    /// GLM53_TF_ROCE_FAST: the kernel's `opts` word
    fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS: the largest grid
    max_grid: u32 = 16,
    /// This rank's region (for shm: half of the mapping in `mapping`).
    region: []align(page) u8,
    mapping: []align(page) u8,
    wire: Wire,
    cpu: ?u32,
    // device side (absent in host-only use: tests and `tf-tp-test roce-host`)
    d: ?*const cuda.Driver = null,
    hr: ?HostRegister = null,
    dev: DevicePtr = 0,
    state: ?cuda.DeviceBuffer = null,
    module: ?cuda.Module = null,
    func: ?cuda.Function = null,
    /// The kernel image takes `RoceArgs.chunk` (it exports tp_roce_chunk_abi): TF_TP_ROCE_CHUNKED_KB needs it.
    chunk_abi: bool = false,
    /// `allGatherVChunked`'s ops launched (tf-kv-split-test's report).
    chunk_ops: u64 = 0,
    // the proxy thread
    thread: ?std.Thread = null,
    running: std.atomic.Value(bool) = .init(false),
    last_seq: u32 = 0,
    err_buf: [160]u8 = undefined,
    err_len: usize = 0,
    // host emulation of the kernel's side (tests)
    host_epoch: u32 = 0,
    aborted: std.atomic.Value(bool) = .init(false),

    pub fn regionBytes(slot: usize) usize {
        return send_off + 4 * slot;
    }

    /// Both ranks call it after the bootstrap (two ranks). Any rank failing makes every rank return
    /// `error.RoceUnavailable` (the setup is collective, as in roce.py); `image` empty: host-only (no kernel).
    pub fn init(d: ?*const cuda.Driver, b: *Bootstrap, image: []const u8, o: Options) !*Roce {
        if (b.world != 2) return error.Unsupported;
        if (o.max_bytes > rv.max_write) return error.WriteTooLarge; // a slot no post could fill in one write
        const slot = std.mem.alignForward(usize, o.max_bytes, 4096);
        const len = regionBytes(slot);
        const r = try std.heap.page_allocator.create(Roce);
        errdefer std.heap.page_allocator.destroy(r);
        r.* = .{ .me = b.rank, .slot_bytes = slot, .timeout_ns = o.timeout_ns, .fast = o.fast, .max_grid = o.blocks, .region = undefined, .mapping = undefined, .wire = undefined, .cpu = o.cpu };
        // 1. the regions and the wire, each rank alone; every rank learns whether all did
        const ok = r.openWire(b, o, len) catch |e| blk: {
            std.log.warn("[tensorfold] tp roce: rank {d} setup failed: {t}", .{ b.rank, e });
            break :blk false;
        };
        if (!try agreeAll(b, ok, "RoCE setup")) {
            if (ok) r.closeWire();
            return error.RoceUnavailable;
        }
        errdefer r.closeWire();
        // 2. the verbs connection records, through the bootstrap
        const connected = r.connectWire(b) catch |e| blk: {
            std.log.warn("[tensorfold] tp roce: rank {d} connect failed: {t}", .{ b.rank, e });
            break :blk false;
        };
        if (!try agreeAll(b, connected, "RoCE connect")) return error.RoceUnavailable;
        // 3. the device side
        if (d) |drv| {
            const dev_ok = r.openDevice(drv, image) catch |e| blk: {
                std.log.warn("[tensorfold] tp roce: rank {d} device setup failed: {t}", .{ b.rank, e });
                break :blk false;
            };
            if (!try agreeAll(b, dev_ok, "RoCE device setup")) {
                if (dev_ok) r.closeDevice();
                return error.RoceUnavailable;
            }
        }
        r.running.store(true, .release);
        r.thread = try std.Thread.spawn(.{}, proxyLoop, .{r});
        try b.agree("proxy", "RoCE proxy start");
        return r;
    }

    fn agreeAll(b: *Bootstrap, ok: bool, _: []const u8) !bool {
        var blobs: [2][]const u8 = undefined;
        var buf: [2]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(ok)}, &buf, &blobs);
        return all[0][0] == 1 and all[1][0] == 1;
    }

    fn openWire(r: *Roce, b: *Bootstrap, o: Options, len: usize) !bool {
        switch (o.wire) {
            .verbs => {
                r.mapping = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
                r.region = r.mapping;
                errdefer posix.munmap(r.mapping);
                try rv.prepareRegion(r.mapping);
                r.wire = .{ .verbs = try rv.VerbsWire.open(o.hcas, r.region, .{ .send = send_off, .recv = send_off + 2 * r.slot_bytes, .flags = flag_off, .slot = r.slot_bytes }, o.traffic_class) };
            },
            .shm => {
                r.mapping = try sharedPair(b, len);
                r.region = @alignCast(r.mapping[r.me * len ..][0..len]);
                r.wire = .{ .shm = .{ .peer = r.mapping[(1 - r.me) * len ..][0..len], .local = r.region, .slot_bytes = r.slot_bytes, .stripes = o.shm_stripes, .tap = o.tap } };
            },
        }
        return true;
    }

    fn connectWire(r: *Roce, b: *Bootstrap) !bool {
        switch (r.wire) {
            .shm => return true,
            .verbs => |*v| {
                const mine = v.record();
                var blobs: [2][]const u8 = undefined;
                var buf: [2 * @sizeOf(rv.Record)]u8 = undefined;
                const all = try b.allGather(std.mem.asBytes(&mine), &buf, &blobs);
                var peer: rv.Record = undefined;
                if (all[1 - r.me].len != @sizeOf(rv.Record)) return error.BadRecord;
                @memcpy(std.mem.asBytes(&peer), all[1 - r.me]);
                try v.connect(peer);
                var line: [512]u8 = undefined;
                std.log.info("[tensorfold] tp roce: rank {d} registered {s}", .{ r.me, v.describe(&line) });
                return true;
            },
        }
    }

    fn openDevice(r: *Roce, d: *const cuda.Driver, image: []const u8) !bool {
        if (image.len == 0) return error.NoKernel;
        r.d = d;
        r.hr = try HostRegister.open();
        try d.check(r.hr.?.register(r.region.ptr, r.region.len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&r.dev, r.region.ptr, 0), "cuMemHostGetDevicePointer");
        r.state = try cuda.DeviceBuffer.alloc(d, 16);
        try r.state.?.fill8(0, null);
        r.module = try cuda.Module.load(d, image);
        r.func = try r.module.?.function("tp_roce");
        r.chunk_abi = if (r.module.?.function("tp_roce_chunk_abi")) |_| true else |_| false;
        return true;
    }

    fn closeDevice(r: *Roce) void {
        if (r.module) |*m| m.unload();
        if (r.state) |*s| s.free();
        if (r.hr) |*h| {
            _ = h.unregister(r.region.ptr);
            h.lib.close();
        }
        r.module = null;
        r.state = null;
        r.hr = null;
    }

    fn closeWire(r: *Roce) void {
        switch (r.wire) {
            .verbs => |*v| v.close(),
            .shm => {},
        }
        posix.munmap(r.mapping);
    }

    /// Regions a failed transport left registered and mapped (the process exit is their only teardown).
    pub var kept: std.atomic.Value(u32) = .init(0);

    /// HCAs and GPU quiet, then unregister and unmap; a failed transport's region is kept until exit.
    pub fn deinit(r: *Roce) void {
        const was_failed = r.failed();
        r.word(w_abort).store(1, .release);
        r.running.store(false, .release);
        if (r.thread) |t| t.join();
        const quiet = r.quiesce();
        if (was_failed or !quiet) {
            const why = if (was_failed) "failed" else "did not quiesce";
            std.log.warn("[tensorfold] tp roce: rank {d}: the transport {s}; region 0x{x} ({d} B) kept registered until exit", .{ r.me, why, @intFromPtr(r.region.ptr), r.region.len });
            _ = kept.fetchAdd(1, .monotonic);
            return;
        }
        r.closeDevice();
        r.closeWire();
        std.heap.page_allocator.destroy(r);
    }

    /// The QPs to error with every posted request reaped, and the GPU's work on the region done.
    fn quiesce(r: *Roce) bool {
        const wire = switch (r.wire) {
            .verbs => |*v| v.quiesce(rv.quiesce_ns),
            .shm => true,
        };
        const gpu = if (r.d) |d| d.api.cuCtxSynchronize() == cuda.abi.success else true;
        var line: [384]u8 = undefined;
        std.log.info("[tensorfold] tp roce: rank {d} quiesced ({s}): {s}; GPU sync {s}", .{ r.me, if (wire) "drained" else "NOT drained", r.report(&line), if (gpu) "done" else "FAILED" });
        return wire and gpu;
    }

    fn report(r: *const Roce, buf: []u8) []const u8 {
        return switch (r.wire) {
            .verbs => |*v| v.report(buf),
            .shm => "shm wire",
        };
    }

    fn word(r: *const Roce, i: usize) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(r.region[ctrl_off + 4 * i ..].ptr));
    }

    fn fail(r: *Roce, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&r.err_buf, fmt, args) catch "proxy failure";
        r.err_len = s.len;
        std.log.err("[tensorfold] tp roce proxy: {s}", .{s});
        r.word(w_proxy_failed).store(1, .release);
        r.word(w_abort).store(1, .release); // waits on this rank end now instead of at the timeout
        r.halt();
    }

    /// The HCAs stop: nothing of this rank's goes out and none of the peer's writes lands while the ranks go down.
    fn halt(r: *Roce) void {
        switch (r.wire) {
            .verbs => |*v| {
                const ok = v.halt();
                var line: [384]u8 = undefined;
                std.log.warn("[tensorfold] tp roce: rank {d} halted{s}: {s}", .{ r.me, if (ok) "" else " (a QP refused ERR)", v.report(&line) });
            },
            .shm => {},
        }
    }

    /// Every post of this rank's completed (shm: nothing is ever in flight).
    pub fn linkIdle(r: *const Roce) bool {
        return switch (r.wire) {
            .verbs => |*v| v.idle(),
            .shm => true,
        };
    }

    /// Each HCA's posts and completions (tf-tp-test's idle check).
    pub fn describeLink(r: *const Roce, buf: []u8) []const u8 {
        return r.report(buf);
    }

    /// tf-tp-test's `failfast RANK transport`: the proxy's failure path (`fail`), here with no op in flight.
    pub fn inject(r: *Roce) void {
        r.fail("failure injected by tf-tp-test (no RDMA fault)", .{});
    }

    /// The transport failed or was aborted: a device may still write to its region.
    pub fn failed(r: *const Roce) bool {
        if (r.aborted.load(.acquire) or r.word(w_abort).load(.acquire) != 0) return true;
        return r.word(w_proxy_failed).load(.acquire) != 0 or r.word(w_failed).load(.acquire) != 0;
    }

    /// roce.cpp's main_loop: spin on the doorbell, post every new sequence (at most two pending: a peer is never
    /// more than one op ahead), reap completions while idle, nap only after a long idle stretch.
    fn proxyLoop(r: *Roce) void {
        if (r.cpu) |cpu| pin(cpu);
        var idle: u64 = 0;
        while (r.running.load(.monotonic)) {
            if (r.word(w_abort).load(.monotonic) != 0) return;
            const seq = r.word(w_seq).load(.acquire);
            if (seq == r.last_seq) {
                idle += 1;
                if (idle % 64 == 0) r.wire.reap() catch |e| return r.fail("completion: {t}", .{e});
                if (idle >= idle_spins) sock.sleepNs(20 * std.time.ns_per_us) else std.atomic.spinLoopHint();
                continue;
            }
            idle = 0;
            var pending = seq -% r.last_seq;
            if (pending > 2) return r.fail("doorbell skipped {d} ops (posted {d}, now {d})", .{ pending, r.last_seq, seq });
            var s = r.last_seq +% 1;
            while (pending > 0) : ({
                s +%= 1;
                pending -= 1;
            }) {
                const padded = r.word(w_slot_bytes + (s & 1)).load(.acquire);
                if (padded == 0 or padded % 16 != 0 or padded > r.slot_bytes) return r.fail("op {d} of {d} bytes", .{ s, padded });
                r.wire.post(s & 1, s, padded) catch |e| return r.fail("post of op {d}: {t}", .{ s, e });
                r.last_seq = s;
            }
            r.wire.reap() catch |e| return r.fail("completion: {t}", .{e});
        }
    }

    pub fn fits(r: *const Roce, n: usize) bool {
        return n <= r.slot_bytes;
    }

    fn launch(r: *Roce, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        return r.launchAt(mode, in, out, n, lens, 0, stream);
    }

    fn launchAt(r: *Roce, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, lens: DevicePtr, chunk: u64, stream: collective.Stream) Error!void {
        if (r.aborted.load(.acquire)) return error.Aborted;
        if (n > r.slot_bytes and (lens == 0 or !r.chunk_abi)) return error.Unsupported;
        if (n == 0) return;
        const d = r.d orelse return error.Unsupported;
        var a: RoceArgs = .{
            .in = in,
            .out = out,
            .send_slots = r.dev + send_off,
            .recv_slots = r.dev + send_off + 2 * r.slot_bytes,
            .flags = r.dev + flag_off,
            .ctrl = r.dev + ctrl_off,
            .state = r.state.?.ptr,
            .nbytes = n,
            .slot_bytes = r.slot_bytes,
            .timeout_ns = r.timeout_ns,
            .rank = r.me,
            .mode = @intFromEnum(mode),
            .n_hca = r.wire.hcas(),
            .opts = r.fast,
            .lens = lens,
            .chunk = chunk,
        };
        var params = [1]?*anyopaque{@ptrCast(&a)};
        const blocks = @min(@import("mailbox.zig").blocksFor(@min(n, r.slot_bytes)), r.max_grid);
        d.check(d.api.cuLaunchKernel(r.func.?.handle, blocks, 1, 1, threads, 1, 1, 0, stream, &params, null), "cuLaunchKernel tp_roce") catch return error.BackendFailed;
    }

    pub fn allGatherBytes(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return r.launch(.gather, s, o, n, 0, stream);
    }

    /// `allGatherBytes` of n-byte strides where rank r stages and posts only lens[r] bytes (device int32 [2], the same on both
    /// ranks; padded to 16, at least 16: the proxy posts no empty op) and the peer copies lens[r] of them to out + r x n.
    pub fn allGatherV(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        if (n > r.slot_bytes) return error.Unsupported; // one op: past a slot is allGatherVChunked's
        if (lens == 0) return error.Invalid;
        return r.launch(.gather, s, o, n, lens, stream);
    }

    /// `allGatherV` past the slot: op c moves at most a slot of each shard from c x slot (empty on both ranks: no op).
    pub fn allGatherVChunked(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        if (lens == 0) return error.Invalid;
        if (!r.chunk_abi) return error.Unsupported;
        var off: usize = 0;
        while (off < n) : (off += r.slot_bytes) {
            try r.launchAt(.gather, s, o, n, lens, off, stream);
            r.chunk_ops += 1;
        }
    }

    /// The most bytes the verbs wire has had posted and not completed (shm: none in flight).
    pub fn peakInflight(r: *const Roce) u64 {
        return switch (r.wire) {
            .verbs => |*w| w.peak.load(.monotonic),
            .shm => 0,
        };
    }

    pub fn exchangeBytes(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return r.launch(.exchange, s, o, n, 0, stream);
    }

    pub fn sum(r: *Roce, s: DevicePtr, o: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const mode: Mode = switch (t) {
            .f32 => .sum_f32,
            .bf16 => .sum_bf16,
            else => return error.Unsupported,
        };
        return r.launch(mode, s, o, count * t.size(), 0, stream);
    }

    /// Ends this rank's waits and the proxy (the peer learns through the fate channel and aborts its own).
    pub fn abort(r: *Roce) void {
        r.aborted.store(true, .release);
        r.word(w_abort).store(1, .release);
        r.halt();
    }

    pub fn check(r: *Roce) Error!void {
        if (r.word(w_failed).load(.acquire) != 0) {
            std.log.warn("tp roce: rank {d}'s wait for seq {d} failed ({s} after {d} us; HCA {d}, flag held {d}); doorbell {d}, posted {d}, completed {d}, abort {d}, proxy failed {d}, flags {d} {d}", .{ r.me, r.word(w_failed + 1).load(.acquire), if (r.word(w_failed + 4).load(.acquire) == 1) "aborted" else "timed out", r.word(w_failed + 5).load(.acquire), r.word(w_failed + 3).load(.acquire), r.word(w_failed + 2).load(.acquire), r.word(w_seq).load(.acquire), r.last_seq, r.word(w_completed).load(.acquire), r.word(w_abort).load(.acquire), r.word(w_proxy_failed).load(.acquire), r.flagWord(0), r.flagWord(1) });
            return error.PeerFailed;
        }
        if (r.word(w_proxy_failed).load(.acquire) != 0) return error.BackendFailed;
        if (r.aborted.load(.acquire) or r.word(w_abort).load(.acquire) != 0) return error.Aborted;
    }

    fn flagWord(r: *const Roce, slot: usize) u32 {
        const p: *std.atomic.Value(u32) = @ptrCast(@alignCast(r.region[flag_off + slot * 2 * flag_stride ..].ptr));
        return p.load(.acquire);
    }

    pub fn proxyError(r: *const Roce) []const u8 {
        return r.err_buf[0..r.err_len];
    }

    /// The kernel's side of one op done by the host (tests, and `tf-tp-test roce-host` on hosts without a GPU check):
    /// out <- [rank 0's in, rank 1's in] (gather) through the same slots, doorbell, wire and flags.
    pub fn hostGather(r: *Roce, in: []const u8, out: []u8) Error!void {
        return r.hostGatherV(in, out, null);
    }

    /// `hostGather` with the kernel's variable lengths (`allGatherV`): rank k's first lens[k] bytes of its n land at out + k x n.
    pub fn hostGatherV(r: *Roce, in: []const u8, out: []u8, lens: ?[2]u32) Error!void {
        if (in.len > r.slot_bytes) return error.Invalid;
        return r.hostGatherAt(in, out, lens, 0);
    }

    /// `allGatherVChunked`'s ops done by the host (the kernel's side of each, as `hostGatherV`).
    pub fn hostGatherVChunked(r: *Roce, in: []const u8, out: []u8, lens: [2]u32) Error!void {
        var off: usize = 0;
        while (off < in.len) : (off += r.slot_bytes) try r.hostGatherAt(in, out, lens, off);
    }

    /// One op from shard offset `off`: each rank's bytes past it, at most a slot; an op empty on both ranks is skipped.
    fn hostGatherAt(r: *Roce, in: []const u8, out: []u8, lens: ?[2]u32, off: usize) Error!void {
        const n = in.len;
        if (n == 0 or out.len < 2 * n) return error.Invalid;
        if (lens) |l| if (l[0] > n or l[1] > n) return error.Invalid;
        const full: [2]usize = if (lens) |l| .{ l[0], l[1] } else .{ n, n };
        const mine: usize = @min(full[r.me] -| off, r.slot_bytes);
        const theirs: usize = @min(full[1 - r.me] -| off, r.slot_bytes);
        if (off > 0 and mine == 0 and theirs == 0) return;
        const seq = r.host_epoch +% 1;
        const slot: usize = seq & 1;
        @memcpy(r.region[send_off + slot * r.slot_bytes ..][0..mine], in[off..][0..mine]);
        r.word(w_slot_bytes + slot).store(@intCast(@max(16, std.mem.alignForward(usize, mine, 16))), .release);
        r.word(w_seq).store(seq, .release);
        @memcpy(out[r.me * n + off ..][0..mine], in[off..][0..mine]);
        const t0 = sock.nowNs();
        for (0..r.wire.hcas()) |h| {
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(r.region[flag_off + (slot * 2 + h) * flag_stride ..].ptr));
            while (flag.load(.acquire) != seq) {
                if (r.word(w_abort).load(.monotonic) != 0) return error.Aborted;
                if (sock.nowNs() - t0 > r.timeout_ns) return error.Timeout;
                std.atomic.spinLoopHint();
            }
        }
        @memcpy(out[(1 - r.me) * n + off ..][0..theirs], r.region[send_off + 2 * r.slot_bytes + slot * r.slot_bytes ..][0..theirs]);
        r.host_epoch = seq;
    }
};

/// One shared mapping holding both ranks' regions (rank 0 creates and sizes the file before anyone learns its name).
fn sharedPair(b: *Bootstrap, len: usize) ![]align(page) u8 {
    var name: [64]u8 = undefined;
    var name_len: u8 = 0;
    var fd: std.c.fd_t = -1;
    defer if (fd >= 0) {
        _ = std.c.close(fd);
    };
    if (b.rank == 0) blk: {
        const s = std.fmt.bufPrint(&name, "/dev/shm/tf-tp-roce-{d}-{x}", .{ std.c.getpid(), sock.nowNs() }) catch break :blk;
        name[s.len] = 0;
        fd = std.c.open(name[0..s.len :0], .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd >= 0 and std.c.ftruncate(fd, @intCast(2 * len)) == 0) name_len = @intCast(s.len);
    }
    // an empty name tells the other rank that rank 0 failed (it must not wait for a name that never comes)
    try b.broadcast(std.mem.asBytes(&name_len), std.mem.asBytes(&name_len));
    if (name_len == 0) return error.SharedMemoryUnavailable;
    try b.broadcast(name[0..name_len], name[0..name_len]);
    name[name_len] = 0;
    if (b.rank != 0) {
        fd = std.c.open(name[0..name_len :0], .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SharedMemoryUnavailable;
    }
    const m = try posix.mmap(null, 2 * len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
    try b.agree("mapped", "RoCE shm wire");
    if (b.rank == 0) _ = std.c.unlink(name[0..name_len :0]);
    return m;
}

fn pin(cpu: u32) void {
    var set: std.os.linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    std.os.linux.sched_setaffinity(0, &set) catch std.log.warn("tp roce: could not pin the proxy to CPU {d}", .{cpu});
}
