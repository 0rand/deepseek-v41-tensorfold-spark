//! A rank's boot memory: the rope tables uploaded from a scratch read, and the per-step trace (TF_DSV41_BOOT_TRACE).

const std = @import("std");

const log = std.log.scoped(.dsv41);
const GiB: f64 = 1 << 30;
const MiB: f64 = 1 << 20;

/// rope.bin's header bytes, then the main and the compressed table (f32 [rows, 64] each)
pub const rope_header = 16;

/// Reads `path` into scratch memory, gives each table's `want` bytes to `sink.table(i, bytes)`, then `sink.done()`.
pub fn uploadRope(gpa: std.mem.Allocator, io: std.Io, path: []const u8, want: usize, sink: anytype) !void {
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const rope = try std.Io.Dir.cwd().readFileAlloc(io, path, scratch.allocator(), .unlimited);
    const half = (rope.len -| rope_header) / 2;
    if (half < want) {
        log.err("{s} holds {d} positions, the limit needs {d}", .{ path, half / 256, want / 256 });
        return error.RopeTooShort;
    }
    for (0..2) |i| try sink.table(i, rope[rope_header + i * half ..][0..want]);
    try sink.done();
}

/// A "Key:   N kB" line's value in bytes (/proc/meminfo, /proc/self/status); null when the key is absent.
pub fn field(text: []const u8, key: []const u8) ?u64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        var f = std.mem.tokenizeAny(u8, line[key.len..], " \t");
        const n = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return n * 1024;
    }
    return null;
}

fn procField(path: [*:0]const u8, key: []const u8) ?u64 {
    var buf: [8192]u8 = undefined;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return field(buf[0..@intCast(n)], key);
}

const malloc_trim = @extern(?*const fn (usize) callconv(.c) c_int, .{ .name = "malloc_trim", .linkage = .weak });

/// One boot step's figures: MemAvailable, the process's device and pinned bytes, its driver allocations, RssAnon.
pub const Figures = struct {
    avail: ?u64,
    device: u64,
    allocs: u64,
    pinned: u64,
    anon: ?u64,
};

fn gib(v: ?u64) f64 {
    return if (v) |x| @as(f64, @floatFromInt(x)) / GiB else -1;
}

/// The step's log line; `drop` is MemAvailable at the previous step less now (GiB; null on the first step).
pub fn stepLine(buf: []u8, name: []const u8, x: Figures, drop: ?f64) []const u8 {
    return std.fmt.bufPrint(buf, "boot trace: {s}: MemAvailable {d:.2} GiB (drop {d:.3}), device {d:.2} GiB in {d} driver allocations, pinned {d:.3} GiB, RssAnon {d:.2} GiB", .{
        name, gib(x.avail), drop orelse 0, gib(x.device), x.allocs, gib(x.pinned), gib(x.anon),
    }) catch buf;
}

/// TF_DSV41_BOOT_TRACE=1: a line a boot step, the buffer plan's totals, and malloc_trim's effect at the end.
pub const Trace = struct {
    on: bool = false,
    last: ?u64 = null,

    pub fn fromEnv() Trace {
        const v = std.c.getenv("TF_DSV41_BOOT_TRACE") orelse return .{};
        return .{ .on = std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1") };
    }

    /// `u`: cuda.usage(false) (device, host, allocs).
    pub fn step(t: *Trace, name: []const u8, u: anytype) void {
        if (!t.on) return;
        const x: Figures = .{
            .avail = procField("/proc/meminfo", "MemAvailable:"),
            .device = u.device,
            .allocs = u.allocs,
            .pinned = u.host,
            .anon = procField("/proc/self/status", "RssAnon:"),
        };
        const drop: ?f64 = if (t.last != null and x.avail != null) gib(t.last) - gib(x.avail) else null;
        var buf: [256]u8 = undefined;
        log.info("{s}", .{stepLine(&buf, name, x, drop)});
        t.last = x.avail;
    }

    /// `totals`: buffers.Plan.totals() (roles and bytes by scope: persistent, window, layer).
    pub fn plan(t: *const Trace, totals: anytype) void {
        if (!t.on) return;
        const b = totals.bytes;
        log.info("boot trace: buffer plan: persistent {d} roles {d:.1} MiB, window {d} roles {d:.1} MiB, layer {d} roles {d:.1} MiB", .{
            totals.roles[0], @as(f64, @floatFromInt(b[0])) / MiB, totals.roles[1], @as(f64, @floatFromInt(b[1])) / MiB, totals.roles[2], @as(f64, @floatFromInt(b[2])) / MiB,
        });
    }

    /// glibc's free heap returned to the system, logged as RssAnon before and after (a no-op without glibc).
    pub fn trim(t: *const Trace) void {
        if (!t.on) return;
        const f = malloc_trim orelse return log.info("boot trace: malloc_trim: not in this libc", .{});
        const before = procField("/proc/self/status", "RssAnon:");
        _ = f(0);
        const after = procField("/proc/self/status", "RssAnon:");
        log.info("boot trace: malloc_trim: RssAnon {d:.3} -> {d:.3} GiB", .{ gib(before), gib(after) });
    }
};

test "the rope tables reach the sink byte for byte and the read is freed" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const rows = 5;
    const half = rows * 64 * 4;
    const data = try gpa.alloc(u8, rope_header + 2 * half);
    defer gpa.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    try tmp.dir.writeFile(io, .{ .sub_path = "rope.bin", .data = data });
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/rope.bin", .{&tmp.sub_path});
    defer gpa.free(path);
    const Sink = struct {
        got: [2][]const u8 = undefined,
        copies: [2][4 * 64 * 4]u8 = undefined,
        fn table(s: *@This(), i: usize, bytes: []const u8) !void {
            @memcpy(s.copies[i][0..bytes.len], bytes);
            s.got[i] = s.copies[i][0..bytes.len];
        }
        fn done(_: *@This()) !void {}
    };
    var sink: Sink = .{};
    const want = 4 * 64 * 4;
    try uploadRope(gpa, io, path, want, &sink);
    try std.testing.expectEqualSlices(u8, data[rope_header..][0..want], sink.got[0]);
    try std.testing.expectEqualSlices(u8, data[rope_header + half ..][0..want], sink.got[1]);
}

test "proc fields in bytes, the step line" {
    const meminfo = "MemTotal:       125481616 kB\nMemFree:         5438312 kB\nMemAvailable:    4893208 kB\n";
    try std.testing.expectEqual(@as(?u64, 4893208 * 1024), field(meminfo, "MemAvailable:"));
    try std.testing.expectEqual(@as(?u64, null), field(meminfo, "RssAnon:"));
    try std.testing.expectEqual(@as(?u64, 2548 * 1024), field("Name:\tx\nRssAnon:\t    2548 kB\n", "RssAnon:"));
    var buf: [256]u8 = undefined;
    const x: Figures = .{ .avail = 6 << 30, .device = 3 << 29, .allocs = 2711, .pinned = 1 << 28, .anon = 1 << 30 };
    try std.testing.expectEqualStrings("boot trace: bind: MemAvailable 6.00 GiB (drop 0.500), device 1.50 GiB in 2711 driver allocations, pinned 0.250 GiB, RssAnon 1.00 GiB", stepLine(&buf, "bind", x, 0.5));
}
