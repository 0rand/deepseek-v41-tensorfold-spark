//! Engram's table rows from the kit's packed per-rank shards on local NVMe (Python engram_host.ShardRows, prefetch.py,
//! engram_gate.py's reader): the local `engram_host.Source` (R3).
//!
//! - **File:** `<dir>/engram-l<layer>-r<rank>of<world>.bin`, a 4,096-byte header of 6 little-endian u64 (magic
//!   `DSV41EN1`, layer, lo, hi, total rows, row bytes), then rows lo .. hi - 1 of 264 bytes: 256 e4m3 values and 8
//!   UE8M0 scales (one a 32). Read with O_DIRECT (buffered where the filesystem refuses it, the pages dropped).
//! - **Bits:** a value is `bf16(e4m3 x 2^(scale - 127))` by torch's CPU rules (`dequant(...).to(bfloat16)`, what
//!   `blocks.engram_rows` gives), through one 65,536-entry table [scale << 8 | byte]; its SHA-256 equals the table
//!   torch builds (fixtures/engram-rows-ref.txt). A row's bytes are the file's whatever the path (cache, prefetched,
//!   read on demand), so replies never depend on I/O.
//! - **Async:** `issue` puts a batch's sector-aligned extents (holes up to 4 KiB merged, Python's `extents`) on an
//!   io_uring and returns; `rows` takes cached records, waits for its batches' completions and reads the rest. One
//!   thread (the engine's): the kernel does the I/O.
//! - **No io_uring** (Docker's default seccomp profile refuses `io_uring_setup`, as in prod's containers): the extents go
//!   to a persistent pool of reader threads (`pread`), as Python's engram_native does for the same reason;
//!   TF_DSV41_ENGRAM_QD threads (Python's name and default, 64). `issue` returns at once as with io_uring, and a miss
//!   reads its extents in parallel. TF_DSV41_ENGRAM_QD=0: the reads run serially at `rows` / `issue` (the old path).
//! - **Cache:** an LRU of `cap` records a layer (Python's 65,536 = 17 MB). Completed batches enter it after the call
//!   that waited for them has converted its rows (no record moves under a reader); bulk batches (over cap / 4 rows,
//!   a prompt's) are read through.

const std = @import("std");
const linux = std.os.linux;
const eh = @import("engram_host.zig");

pub const magic: u64 = 0x31344E4531565344;
pub const header_bytes: u64 = 4096;
pub const sector: u64 = 512;
pub const row_bytes: u64 = 264;
pub const values = 256;
pub const merge_gap: u64 = 4096;
const page = 4096;

pub const Header = struct {
    layer: u64,
    lo: u64,
    hi: u64,
    total: u64,
    row_bytes: u64,

    /// The header's six words, checked against the file's size (Python read_header).
    pub fn parse(b: *const [48]u8, size: u64) !Header {
        var w: [6]u64 = undefined;
        for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, b[8 * i ..][0..8], .little);
        if (w[0] != magic) return error.NotEngramShard;
        const h: Header = .{ .layer = w[1], .lo = w[2], .hi = w[3], .total = w[4], .row_bytes = w[5] };
        if (h.row_bytes != row_bytes or h.hi < h.lo or size != header_bytes + (h.hi - h.lo) * h.row_bytes) return error.BadEngramShard;
        return h;
    }
};

/// torch's float8_e4m3fn -> float32 (c10 fp8e4m3fn_to_fp32_value, bit for bit).
pub fn e4m3(b: u8) f32 {
    const w: u32 = @as(u32, b) << 24;
    const sign = w & 0x8000_0000;
    const nonsign = w & 0x7FFF_FFFF;
    var shift: u32 = @clz(nonsign);
    shift = if (shift > 4) shift - 4 else 0;
    const inf_nan: u32 = @bitCast((@as(i32, @bitCast(nonsign +% 0x0100_0000)) >> 8) & 0x7F80_0000);
    const zero: u32 = @bitCast(@as(i32, @bitCast(nonsign -% 1)) >> 31);
    const r = sign | ((((nonsign << @intCast(shift)) >> 4) +% ((0x78 -% shift) << 23) | inf_nan) & ~zero);
    return @bitCast(r);
}

/// torch's CPU float32 -> bfloat16 (its vectorized conversion, what `.to(bfloat16)` of a rows tensor runs): round to
/// nearest even, NaN -> 0xFFFF. A NaN needs an e4m3 NaN code or scale 255, which no packed table holds; c10's scalar
/// rule gives 0x7FC0 there, so only these 514 of 65,536 entries depend on torch's code path.
pub fn bf16(x: f32) u16 {
    if (std.math.isNan(x)) return 0xFFFF;
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7FFF + ((u >> 16) & 1))) >> 16);
}

/// The table [scale << 8 | byte] -> bf16 bits of e4m3(byte) x 2^(scale - 127) (scale 0: x 0).
pub fn fillTable(t: *[65536]u16) void {
    for (0..256) |s| {
        const mul: f32 = @bitCast(@as(u32, @intCast(s)) << 23);
        for (0..256) |b| t[s << 8 | b] = bf16(e4m3(@intCast(b)) * mul);
    }
}

/// One extent of a batch: file bytes [off, off + len) into the batch buffer at `at`, its rows `first .. + count`.
const Extent = struct { off: u64, len: u64, at: usize, first: u32, count: u32 };

const Batch = struct {
    table: u32,
    /// sorted unique global rows
    rows: []u64,
    ext: []Extent,
    buf: []align(page) u8,
    left: u32,
    failed: bool = false,
    in_use: bool = true,
    /// a `rows` call took at least half its rows (a bulk batch is read through once read); `taken`: this call's count
    read: bool = false,
    taken: u32 = 0,
    /// `rows` calls on its table that settled it unread: a bulk batch issued a prompt segment ahead
    /// (TF_DSV41_PREFETCH_AHEAD) waits through one of them, then a wrong guess is dropped
    unread: u8 = 0,

    /// The record of `rows[k]` (k found by the caller).
    fn record(b: *const Batch, k: usize, lo: u64) []const u8 {
        var a: usize = 0;
        var z: usize = b.ext.len;
        while (z - a > 1) {
            const m = (a + z) / 2;
            if (b.ext[m].first <= k) a = m else z = m;
        }
        const e = b.ext[a];
        const at = e.at + @as(usize, @intCast(header_bytes + (b.rows[k] - lo) * row_bytes - e.off));
        return b.buf[at..][0..row_bytes];
    }
};

const Entry = union(enum) { ready: u32, pending: u32 };
const none: u32 = std.math.maxInt(u32);

/// One layer's shard and record cache.
const Table = struct {
    fd: linux.fd_t,
    direct: bool,
    h: Header,
    size: u64,
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    /// cap records, their ids, and the LRU links (head: most recent)
    recs: []u8,
    ids: []u64,
    prev: []u32,
    next: []u32,
    used: u32 = 0,
    head: u32 = none,
    tail: u32 = none,
    /// keys removed since the map was last rehashed: each left a tombstone that a missing key's probe walks past
    dead: u32 = 0,

    fn unlink(t: *Table, s: u32) void {
        if (t.prev[s] != none) t.next[t.prev[s]] = t.next[s] else t.head = t.next[s];
        if (t.next[s] != none) t.prev[t.next[s]] = t.prev[s] else t.tail = t.prev[s];
    }

    fn front(t: *Table, s: u32) void {
        t.prev[s] = none;
        t.next[s] = t.head;
        if (t.head != none) t.prev[t.head] = s;
        t.head = s;
        if (t.tail == none) t.tail = s;
    }

    /// A slot for a new record: an unused one, else the least recent (its id leaves the map).
    fn take(t: *Table) u32 {
        if (t.used < t.ids.len) {
            t.used += 1;
            return t.used - 1;
        }
        const s = t.tail;
        t.unlink(s);
        t.drop(t.ids[s]);
        return s;
    }

    /// std's tombstones never clear and keys fill 80 %: rehash at a tenth, so a tenth of the map always stays free.
    fn drop(t: *Table, x: u64) void {
        if (!t.map.remove(x)) return;
        t.dead += 1;
        if (@as(u64, t.dead) * 10 < t.map.capacity()) return;
        t.map.rehash(std.hash_map.AutoContext(u64){});
        t.dead = 0;
    }
};

pub const Stats = struct { hits: u64 = 0, waited: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0, wait_ns: u64 = 0 };

/// One extent for a pool thread: file bytes [off, off + buf.len) into `buf` (the batch's own buffer), batch `bi`.
const Job = struct { bi: u32, fd: linux.fd_t, size: u64, direct: bool, buf: []u8, off: u64 };

/// The reader threads without io_uring. `mu` guards the queue and every batch's `left` / `failed` and the batches'
/// list itself (a worker settles `batches.items[bi]` while the engine thread may append a batch).
const Pool = struct {
    mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    work: std.c.pthread_cond_t = .{},
    done: std.c.pthread_cond_t = .{},
    jobs: std.ArrayList(Job) = .empty,
    head: usize = 0,
    stop: bool = false,
    threads: []std.Thread = &.{},

    fn lock(p: *Pool) void {
        _ = std.c.pthread_mutex_lock(&p.mu);
    }

    fn unlock(p: *Pool) void {
        _ = std.c.pthread_mutex_unlock(&p.mu);
    }
};

/// Linux native AIO (TF_DSV41_ENGRAM_AIO=1): a batch's extents in one io_submit, reaped with io_getevents on the
/// thread that waits for them; no reader threads to wake. Docker's default seccomp profile allows these syscalls
/// (it refuses io_uring_setup). Reads O_DIRECT shards asynchronously; a buffered shard's reads complete at submit.
const Aio = struct {
    ctx: u64 = 0,
    in_flight: u32 = 0,
    /// TF_DSV41_ENGRAM_POLL=1: a wait polls the completions (io_getevents with a zero timeout) instead of sleeping in
    /// the kernel, up to `poll_ns`, then sleeps: the waiting thread is awake when its reads land (no wake-up), and
    /// the CPU spins only while a caller waits for reads in flight
    poll: bool = false,

    pub const poll_ns: u64 = 20 * std.time.ns_per_ms;

    pub const depth: u32 = 1024;

    const Iocb = extern struct {
        data: u64,
        key: u32 = 0,
        rw_flags: i32 = 0,
        opcode: u16 = 0, // IOCB_CMD_PREAD
        reqprio: i16 = 0,
        fildes: u32,
        buf: u64,
        nbytes: u64,
        offset: i64,
        reserved2: u64 = 0,
        flags: u32 = 0,
        resfd: u32 = 0,
    };

    const Event = extern struct { data: u64, obj: u64, res: i64, res2: i64 };

    fn init() !Aio {
        var a: Aio = .{ .poll = pollEnv() };
        const rc = linux.syscall2(.io_setup, depth, @intFromPtr(&a.ctx));
        if (linux.errno(rc) != .SUCCESS) return error.AioRefused;
        return a;
    }

    pub fn deinit(a: *Aio) void {
        _ = linux.syscall1(.io_destroy, a.ctx);
    }
};

/// TF_DSV41_ENGRAM_POLL: 0 (default) | 1 (with TF_DSV41_ENGRAM_AIO=1).
pub fn pollEnv() bool {
    const v = std.c.getenv("TF_DSV41_ENGRAM_POLL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_DSV41_ENGRAM_AIO: 0 (default) | 1.
pub fn aioEnv() bool {
    const v = std.c.getenv("TF_DSV41_ENGRAM_AIO") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_DSV41_ENGRAM_QD: the reader threads without io_uring (Python engram_native's knob, default 64; 0: none).
pub fn poolThreads() u32 {
    const v = std.c.getenv("TF_DSV41_ENGRAM_QD") orelse return 64;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64;
}

pub const Rows = struct {
    gpa: std.mem.Allocator,
    tables: []Table,
    ring: ?linux.IoUring,
    /// no io_uring: the reader threads (null: serial reads on the engine thread)
    pool: ?*Pool = null,
    /// no io_uring, TF_DSV41_ENGRAM_AIO=1: native AIO (in place of the pool)
    aio: ?Aio = null,
    batches: std.ArrayList(Batch) = .empty,
    lut: *[65536]u16,
    cap: u32,
    stats: Stats = .{},
    /// scratch reused by `rows` (sorted unique ids, their records)
    uniq: std.ArrayList(u64) = .empty,
    recs: std.ArrayList([*]const u8) = .empty,
    want: std.ArrayList(u32) = .empty,

    /// Opens rank `rank` of `world`'s shard of every Engram layer in `layer_ids` under `dir`, `cap` records a layer.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, rank: u32, world: u32, layer_ids: []const u32, cap: u32) !*Rows {
        if (cap < 4096) return error.EngramCacheTooSmall;
        const r = try gpa.create(Rows);
        errdefer gpa.destroy(r);
        r.* = .{ .gpa = gpa, .tables = try gpa.alloc(Table, layer_ids.len), .ring = null, .lut = try gpa.create([65536]u16), .cap = cap };
        fillTable(r.lut);
        var made: usize = 0;
        errdefer {
            for (r.tables[0..made]) |*t| r.closeTable(t);
            gpa.free(r.tables);
            gpa.destroy(r.lut);
        }
        for (layer_ids, r.tables) |l, *t| {
            var nb: [4096]u8 = undefined;
            const p = try std.fmt.bufPrint(nb[0 .. nb.len - 1], "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, l, rank, world });
            nb[p.len] = 0;
            const path = nb[0..p.len :0];
            t.* = try openTable(gpa, path, cap);
            made += 1;
            if (t.h.layer != l) return error.BadEngramShard;
        }
        // TF_DSV41_ENGRAM_URING=0: the reader threads even where io_uring works (a container that allows it, A/B)
        const uring_off = if (std.c.getenv("TF_DSV41_ENGRAM_URING")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        r.ring = (if (uring_off) error.PermissionDenied else linux.IoUring.init(256, 0)) catch |e| blk: {
            if (aioEnv()) {
                if (Aio.init()) |a| {
                    r.aio = a;
                    std.log.scoped(.dsv41).info("engram: no io_uring ({t}); rows read by native AIO (TF_DSV41_ENGRAM_AIO, {d} in flight), waits {s}", .{ e, Aio.depth, if (a.poll) "polled (TF_DSV41_ENGRAM_POLL)" else "asleep in the kernel" });
                    break :blk null;
                } else |ae| std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_AIO=1 but io_setup failed ({t}): the reader threads", .{ae});
            }
            const n = poolThreads();
            if (n > 0) {
                try r.startPool(n);
                std.log.scoped(.dsv41).info("engram: no io_uring ({t}); rows read by {d} threads (TF_DSV41_ENGRAM_QD)", .{ e, n });
            } else std.log.scoped(.dsv41).warn("engram: no io_uring ({t}) and TF_DSV41_ENGRAM_QD=0; rows are read when a window needs them", .{e});
            break :blk null;
        };
        return r;
    }

    /// Native AIO in place of io_uring or the pool (tests, the probe): every read async through one context.
    pub fn startAio(r: *Rows) !void {
        if (r.ring) |*g| g.deinit();
        r.ring = null;
        r.stopPool();
        if (r.aio == null) r.aio = try Aio.init();
    }

    /// The reader threads (no io_uring): `n` workers waiting for extents.
    pub fn startPool(r: *Rows, n: u32) !void {
        if (r.pool != null or r.ring != null or n == 0) return error.EngramPool;
        const p = try r.gpa.create(Pool);
        errdefer r.gpa.destroy(p);
        p.* = .{};
        p.threads = try r.gpa.alloc(std.Thread, n);
        var made: usize = 0;
        errdefer {
            p.lock();
            p.stop = true;
            p.unlock();
            _ = std.c.pthread_cond_broadcast(&p.work);
            for (p.threads[0..made]) |t| t.join();
            r.gpa.free(p.threads);
        }
        r.pool = p;
        errdefer r.pool = null;
        for (p.threads) |*t| {
            t.* = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{r});
            made += 1;
        }
    }

    /// The reader threads stopped (their queue drained); reads are serial afterwards.
    pub fn stopPool(r: *Rows) void {
        const p = r.pool orelse return;
        p.lock();
        p.stop = true;
        p.unlock();
        _ = std.c.pthread_cond_broadcast(&p.work);
        for (p.threads) |t| t.join();
        r.gpa.free(p.threads);
        p.jobs.deinit(r.gpa);
        r.gpa.destroy(p);
        r.pool = null;
    }

    /// A reader thread: extents off the queue until `stop` (the queue drained first).
    fn worker(r: *Rows) void {
        const p = r.pool.?;
        while (true) {
            p.lock();
            while (p.head == p.jobs.items.len and !p.stop) _ = std.c.pthread_cond_wait(&p.work, &p.mu);
            if (p.head == p.jobs.items.len) {
                p.unlock();
                return;
            }
            const j = p.jobs.items[p.head];
            p.head += 1;
            if (p.head == p.jobs.items.len) {
                p.jobs.clearRetainingCapacity();
                p.head = 0;
            }
            p.unlock();
            const ok = if (readAt(j.fd, j.size, j.direct, j.buf, j.off)) true else |_| false;
            p.lock();
            const b = &r.batches.items[j.bi];
            if (!ok) b.failed = true;
            b.left -= 1;
            const last = b.left == 0;
            p.unlock();
            if (last) _ = std.c.pthread_cond_broadcast(&p.done);
        }
    }

    fn openTable(gpa: std.mem.Allocator, path: [:0]const u8, cap: u32) !Table {
        var direct = true;
        var rc = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, 0);
        if (linux.errno(rc) != .SUCCESS) {
            direct = false;
            rc = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(rc) != .SUCCESS) {
                std.log.scoped(.dsv41).err("engram: cannot open {s}", .{path});
                return error.FileNotFound;
            }
        }
        const fd: linux.fd_t = @intCast(rc);
        errdefer _ = linux.close(fd);
        var st: linux.Statx = undefined;
        if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != .SUCCESS) return error.StatFailed;
        const hb = try gpa.alignedAlloc(u8, .fromByteUnits(page), page);
        defer gpa.free(hb);
        try preadAll(fd, hb, 0);
        if (!direct) _ = linux.fadvise(fd, 0, page, linux.POSIX_FADV.DONTNEED);
        const h = try Header.parse(hb[0..48], st.size);
        const t: Table = .{
            .fd = fd,
            .direct = direct,
            .h = h,
            .size = st.size,
            .recs = try gpa.alloc(u8, cap * row_bytes),
            .ids = try gpa.alloc(u64, cap),
            .prev = try gpa.alloc(u32, cap),
            .next = try gpa.alloc(u32, cap),
        };
        return t;
    }

    fn closeTable(r: *Rows, t: *Table) void {
        _ = linux.close(t.fd);
        t.map.deinit(r.gpa);
        r.gpa.free(t.recs);
        r.gpa.free(t.ids);
        r.gpa.free(t.prev);
        r.gpa.free(t.next);
    }

    pub fn close(r: *Rows) void {
        // every read in flight lands before its buffer goes
        for (r.batches.items, 0..) |*b, i| if (b.in_use) {
            r.wait(@intCast(i)) catch {};
            b.read = true;
        };
        r.settle(null);
        r.stopPool();
        if (r.ring) |*g| g.deinit();
        if (r.aio) |*a| a.deinit();
        for (r.tables) |*t| r.closeTable(t);
        r.gpa.free(r.tables);
        r.batches.deinit(r.gpa);
        r.uniq.deinit(r.gpa);
        r.recs.deinit(r.gpa);
        r.want.deinit(r.gpa);
        r.gpa.destroy(r.lut);
        r.gpa.destroy(r);
    }

    /// engram_host's interface to these rows.
    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }

    fn issueFn(p: *anyopaque, li: usize, idx: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, idx);
    }

    fn rowsFn(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, idx, dim, out);
    }

    /// Sorted unique `idx` into r.uniq (checked against the shard's range).
    fn unique(r: *Rows, t: *const Table, idx: []const i64) !void {
        r.uniq.clearRetainingCapacity();
        try r.uniq.ensureTotalCapacity(r.gpa, idx.len);
        for (idx) |i| {
            if (i < 0 or @as(u64, @intCast(i)) < t.h.lo or @as(u64, @intCast(i)) >= t.h.hi) return error.EngramRowOutsideShard;
            r.uniq.appendAssumeCapacity(@intCast(i));
        }
        std.mem.sort(u64, r.uniq.items, {}, std.sort.asc(u64));
        var m: usize = 0;
        for (r.uniq.items) |x| if (m == 0 or r.uniq.items[m - 1] != x) {
            r.uniq.items[m] = x;
            m += 1;
        };
        r.uniq.shrinkRetainingCapacity(m);
    }

    /// Starts reading layer `li`'s rows `idx` that are neither cached nor in flight; returns at once.
    pub fn issue(r: *Rows, li: usize, idx: []const i64) !void {
        const t = &r.tables[li];
        try r.unique(t, idx);
        var m: usize = 0;
        for (r.uniq.items) |x| if (!t.map.contains(x)) {
            r.uniq.items[m] = x;
            m += 1;
        };
        if (m == 0) return;
        r.stats.issued += m;
        _ = try r.start(@intCast(li), r.uniq.items[0..m], false);
    }

    /// A batch for sorted unique `rows` of table `ti` (none cached or in flight): its extents submitted. `urgent`: a read
    /// the caller waits for now (the pool runs it before queued read-ahead).
    fn start(r: *Rows, ti: u32, rows_in: []const u64, urgent: bool) !u32 {
        const t = &r.tables[ti];
        const gpa = r.gpa;
        var ext: std.ArrayList(Extent) = .empty;
        errdefer ext.deinit(gpa);
        var total: usize = 0;
        for (rows_in, 0..) |x, k| {
            const s = header_bytes + (x - t.h.lo) * row_bytes;
            const s0 = s / sector * sector;
            const s1 = (s + row_bytes + sector - 1) / sector * sector;
            if (ext.items.len > 0) {
                const e = &ext.items[ext.items.len - 1];
                if (s0 <= e.off + e.len + merge_gap) {
                    const end = @max(e.off + e.len, s1);
                    total += @intCast(end - e.off - e.len);
                    e.len = end - e.off;
                    e.count += 1;
                    continue;
                }
            }
            total = std.mem.alignForward(usize, total, page);
            try ext.append(gpa, .{ .off = s0, .len = s1 - s0, .at = total, .first = @intCast(k), .count = 1 });
            total += @intCast(s1 - s0);
        }
        // extents keep their buffer offsets page aligned (O_DIRECT)
        const buf = try gpa.alignedAlloc(u8, .fromByteUnits(page), std.mem.alignForward(usize, total, page));
        errdefer gpa.free(buf);
        const owned = try gpa.dupe(u64, rows_in);
        errdefer gpa.free(owned);
        const b: Batch = .{ .table = ti, .rows = owned, .ext = try ext.toOwnedSlice(gpa), .buf = buf, .left = 0 };
        if (r.pool) |p| {
            // the batch's slot and its extents queued under the pool's lock (a worker settles batches.items[bi])
            try t.map.ensureUnusedCapacity(gpa, @intCast(owned.len));
            p.lock();
            const queued = r.queue(p, t, b, urgent);
            p.unlock();
            const bi = try queued;
            _ = std.c.pthread_cond_broadcast(&p.work);
            for (owned) |x| t.map.putAssumeCapacity(x, .{ .pending = bi });
            return bi;
        }
        const bi = try r.slotFor(b);
        for (owned) |x| try t.map.put(gpa, x, .{ .pending = bi });
        const bp = &r.batches.items[bi];
        r.stats.reads += bp.ext.len;
        for (bp.ext) |e| r.stats.bytes += e.len;
        if (r.ring) |*g| {
            for (bp.ext, 0..) |e, k| {
                const sqe = g.read(@as(u64, bi) << 32 | k, t.fd, .{ .buffer = bp.buf[e.at..][0..@intCast(e.len)] }, e.off) catch blk: {
                    // a full submission queue: submit what is queued, take finished reads, then the slot is free
                    _ = try g.submit();
                    try r.reap(0);
                    break :blk try g.read(@as(u64, bi) << 32 | k, t.fd, .{ .buffer = bp.buf[e.at..][0..@intCast(e.len)] }, e.off);
                };
                _ = sqe;
                bp.left += 1;
            }
            _ = try g.submit();
        } else if (r.aio != null) {
            try r.aioSubmit(bi, t);
        } else {
            for (bp.ext) |e| try r.readExtent(t, bp.buf[e.at..][0..@intCast(e.len)], e.off);
        }
        return bi;
    }

    fn slotFor(r: *Rows, b: Batch) !u32 {
        for (r.batches.items, 0..) |*x, i| if (!x.in_use) {
            x.* = b;
            return @intCast(i);
        };
        try r.batches.append(r.gpa, b);
        return @intCast(r.batches.items.len - 1);
    }

    /// A batch `b` in a free slot, its extents on the pool's queue (`p` locked by the caller): at its end, or `urgent` at
    /// its head (a window's misses go before a prompt segment's read-ahead).
    fn queue(r: *Rows, p: *Pool, t: *const Table, b: Batch, urgent: bool) !u32 {
        try p.jobs.ensureUnusedCapacity(r.gpa, b.ext.len);
        const bi = try r.slotFor(b);
        const bp = &r.batches.items[bi];
        r.stats.reads += bp.ext.len;
        const n = bp.ext.len;
        const at = if (urgent) p.head else p.jobs.items.len;
        if (urgent) {
            // one shift of the queued jobs past the head, then this batch's in their place
            const old = p.jobs.items.len;
            p.jobs.items.len += n;
            std.mem.copyBackwards(Job, p.jobs.items[at + n ..], p.jobs.items[at..old]);
        } else p.jobs.items.len += n;
        for (bp.ext, p.jobs.items[at..][0..n]) |e, *j| {
            r.stats.bytes += e.len;
            j.* = .{ .bi = bi, .fd = t.fd, .size = t.size, .direct = t.direct, .buf = bp.buf[e.at..][0..@intCast(e.len)], .off = e.off };
        }
        bp.left = @intCast(bp.ext.len);
        return bi;
    }

    fn readExtent(r: *Rows, t: *const Table, buf: []u8, off: u64) !void {
        _ = r;
        return readAt(t.fd, t.size, t.direct, buf, off);
    }

    /// Batch `bi`'s extents on the AIO context (all of them, reaping finished reads while the context is full).
    fn aioSubmit(r: *Rows, bi: u32, t: *const Table) !void {
        const a = &r.aio.?;
        const bp = &r.batches.items[bi];
        var cbs: [64]Aio.Iocb = undefined;
        var ptrs: [64]*Aio.Iocb = undefined;
        var k: usize = 0;
        while (k < bp.ext.len) {
            if (a.in_flight == Aio.depth) try r.reap(1);
            const n = @min(bp.ext.len - k, cbs.len, Aio.depth - a.in_flight);
            for (cbs[0..n], ptrs[0..n], bp.ext[k..][0..n], 0..) |*cb, *pp, e, j| {
                cb.* = .{ .data = @as(u64, bi) << 32 | (k + j), .fildes = @intCast(t.fd), .buf = @intFromPtr(bp.buf[e.at..].ptr), .nbytes = e.len, .offset = @intCast(e.off) };
                pp.* = cb;
            }
            const rc = linux.syscall3(.io_submit, a.ctx, n, @intFromPtr(&ptrs));
            switch (linux.errno(rc)) {
                .SUCCESS => {},
                .AGAIN, .INTR => {
                    try r.reap(if (a.in_flight > 0) 1 else 0);
                    continue;
                },
                else => return error.EngramReadFailed,
            }
            bp.left += @intCast(rc);
            a.in_flight += @intCast(rc);
            k += rc;
        }
    }

    /// Takes finished AIO reads (waiting for at least `at_least`).
    fn aioReap(r: *Rows, at_least: u32) !void {
        const a = &r.aio.?;
        var evs: [64]Aio.Event = undefined;
        var need = @min(at_least, a.in_flight);
        const zero: linux.timespec = .{ .sec = 0, .nsec = 0 };
        const t0 = if (a.poll and need > 0) nowNs() else 0;
        var polling = a.poll and need > 0;
        while (true) {
            // polling: whatever has landed, at once; else the kernel's wait for `need`
            if (polling and nowNs() - t0 > Aio.poll_ns) polling = false;
            const rc = if (polling)
                linux.syscall5(.io_getevents, a.ctx, 0, evs.len, @intFromPtr(&evs), @intFromPtr(&zero))
            else
                linux.syscall5(.io_getevents, a.ctx, need, evs.len, @intFromPtr(&evs), 0);
            switch (linux.errno(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => return error.EngramReadFailed,
            }
            for (evs[0..rc]) |ev| {
                const b = &r.batches.items[@intCast(ev.data >> 32)];
                const e = b.ext[@intCast(ev.data & 0xFFFF_FFFF)];
                const t = &r.tables[b.table];
                const want = @min(e.len, t.size - e.off);
                if (ev.res < 0 or @as(u64, @intCast(ev.res)) < want) b.failed = true;
                if (!t.direct) _ = linux.fadvise(t.fd, @intCast(e.off), @intCast(e.len), linux.POSIX_FADV.DONTNEED);
                b.left -= 1;
                a.in_flight -= 1;
            }
            need -|= @intCast(rc);
            if (need == 0) return;
            if (polling and rc == 0) std.atomic.spinLoopHint();
        }
    }

    /// Takes finished reads off the ring (waiting for at least `at_least`).
    fn reap(r: *Rows, at_least: u32) !void {
        if (r.aio != null) return r.aioReap(at_least);
        const g = &(r.ring orelse return);
        var cqes: [64]linux.io_uring_cqe = undefined;
        var need = at_least;
        while (true) {
            const n = try g.copy_cqes(&cqes, need);
            for (cqes[0..n]) |c| {
                const b = &r.batches.items[@intCast(c.user_data >> 32)];
                const e = b.ext[@intCast(c.user_data & 0xFFFF_FFFF)];
                const t = &r.tables[b.table];
                const want = @min(e.len, t.size - e.off);
                if (c.res < 0 or @as(u64, @intCast(c.res)) < want) b.failed = true;
                if (!t.direct) _ = linux.fadvise(t.fd, @intCast(e.off), @intCast(e.len), linux.POSIX_FADV.DONTNEED);
                b.left -= 1;
            }
            need -|= n;
            if (need == 0) return;
        }
    }

    /// Blocks until batch `bi`'s reads are done.
    fn wait(r: *Rows, bi: u32) !void {
        if (r.pool) |p| {
            p.lock();
            defer p.unlock();
            while (r.batches.items[bi].left > 0) _ = std.c.pthread_cond_wait(&p.done, &p.mu);
            if (r.batches.items[bi].failed) return error.EngramReadFailed;
            return;
        }
        while (r.batches.items[bi].left > 0) try r.reap(1);
        if (r.batches.items[bi].failed) return error.EngramReadFailed;
    }

    /// Finished batches into the caches (bulk ones read through), their buffers freed; `li`: the table whose `rows`
    /// call settles (an unread bulk batch of it waits through one such call), null: none.
    fn settle(r: *Rows, li: ?usize) void {
        // the pool's workers write `left` / `failed`: settled under its lock (the batches' list stays put)
        if (r.pool) |p| p.lock();
        defer if (r.pool) |p| p.unlock();
        for (r.batches.items) |*b| {
            if (!b.in_use or b.left > 0) continue;
            const t = &r.tables[b.table];
            const keep = !b.failed and b.rows.len <= r.cap / 4;
            // a settle on another table never drops it: a segment reads layer 1, then 14, so the next segment's layer
            // 14 batch sees this segment's layer-14 read and the next one's layer-1 read before its own
            if (!keep and !b.failed and !b.read and (b.unread < 1 or li != b.table)) {
                if (li == b.table) b.unread += 1;
                continue;
            }
            for (b.rows, 0..) |x, k| {
                if (!keep) {
                    t.drop(x);
                    continue;
                }
                const s = t.take();
                @memcpy(t.recs[s * row_bytes ..][0..row_bytes], b.record(k, t.h.lo));
                t.ids[s] = x;
                t.front(s);
                // the entry is this batch's (a row is in one batch at most; evictions only remove ready entries)
                t.map.getPtr(x).?.* = .{ .ready = s };
            }
            r.gpa.free(b.rows);
            r.gpa.free(b.ext);
            r.gpa.free(b.buf);
            b.in_use = false;
        }
    }

    /// Layer `li`'s rows `idx` (global row ids this rank owns, any order, duplicates allowed) as bf16 [idx.len, dim].
    pub fn rows(r: *Rows, li: usize, idx: []const i64, dim: usize, out: []u16) !void {
        if (dim > values or dim % 32 != 0 or out.len != idx.len * dim) return error.BadEngramRows;
        const t = &r.tables[li];
        try r.unique(t, idx);
        const u = r.uniq.items;
        try r.recs.resize(r.gpa, u.len);
        r.want.clearRetainingCapacity();
        // cached records now; in-flight ones after their batches; the rest read as one batch
        var miss: std.ArrayList(u64) = .empty;
        defer miss.deinit(r.gpa);
        for (u, 0..) |x, k| {
            if (t.map.get(x)) |e| switch (e) {
                .ready => |s| {
                    r.recs.items[k] = t.recs[s * row_bytes ..].ptr;
                    t.unlink(s);
                    t.front(s);
                    r.stats.hits += 1;
                },
                .pending => |bi| {
                    r.batches.items[bi].taken += 1;
                    if (std.mem.indexOfScalar(u32, r.want.items, bi) == null) try r.want.append(r.gpa, bi);
                    r.stats.waited += 1;
                },
            } else try miss.append(r.gpa, x);
        }
        if (miss.items.len > 0) {
            r.stats.misses += miss.items.len;
            const bi = try r.start(@intCast(li), miss.items, true);
            r.batches.items[bi].read = true;
            try r.want.append(r.gpa, bi);
        }
        const t0 = nowNs();
        for (r.want.items) |bi| {
            try r.wait(bi);
            const b = &r.batches.items[bi];
            if (2 * @as(usize, b.taken) >= b.rows.len) b.read = true;
            b.taken = 0;
        }
        r.stats.wait_ns += nowNs() - t0;
        for (u, 0..) |x, k| switch (t.map.get(x).?) {
            .ready => {},
            .pending => |bi| {
                const b = &r.batches.items[bi];
                r.recs.items[k] = b.record(std.sort.binarySearch(u64, b.rows, x, orderU64).?, t.h.lo).ptr;
            },
        };
        // every position: its record's values through the table (scale byte of each 32 values, then the e4m3 byte)
        for (idx, 0..) |i, p| {
            const rec = r.recs.items[std.sort.binarySearch(u64, u, @as(u64, @intCast(i)), orderU64).?];
            const o = out[p * dim ..][0..dim];
            for (o, 0..) |*v, j| v.* = r.lut[@as(usize, rec[values + j / 32]) << 8 | rec[j]];
        }
        r.settle(li);
    }
};

fn orderU64(a: u64, b: u64) std.math.Order {
    return std.math.order(a, b);
}

/// An extent's bytes [off, off + buf.len) of a shard of `size` bytes (the last one may end past the file: the read
/// stops at its end), the pages dropped when the file is not O_DIRECT.
fn readAt(fd: linux.fd_t, size: u64, direct: bool, buf: []u8, off: u64) !void {
    const want = @min(buf.len, size - off);
    var done: usize = 0;
    while (done < want) {
        const rc = linux.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.EngramReadFailed,
        }
        if (rc == 0) return error.EngramShortRead;
        done += rc;
    }
    if (!direct) _ = linux.fadvise(fd, @intCast(off), @intCast(buf.len), linux.POSIX_FADV.DONTNEED);
}

fn preadAll(fd: linux.fd_t, out: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const rc = linux.pread(fd, out[done..].ptr, out.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.EngramReadFailed,
        }
        if (rc == 0) return error.EngramShortRead;
        done += rc;
    }
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
