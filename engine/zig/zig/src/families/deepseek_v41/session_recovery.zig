//! Boot-only intersection of persisted prefix keys, before either rank assigns session entry IDs.
const std = @import("std");
const cuda = @import("cuda");
const Forward = @import("forward.zig").Forward;
const Digest = [16]u8;

fn less(_: void, a: Digest, b: Digest) bool {
    return std.mem.order(u8, &a, &b) == .lt;
}

fn contains(keys: []const Digest, key: Digest) bool {
    var lo: usize = 0;
    var hi = keys.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        switch (std.mem.order(u8, &key, &keys[mid])) {
            .eq => return true,
            .lt => hi = mid,
            .gt => lo = mid + 1,
        }
    }
    return false;
}

/// Every rank calls this once at boot, including ranks with no recoverable entries.
pub fn keepCommon(ctx: *anyopaque, keys: []const Digest, mask: []bool) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const world: usize = f.comm.world();
    const driver = f.runner.d;
    const stream = f.runner.stream;
    var count_dev = try cuda.DeviceBuffer.alloc(driver, 8 * (world + 1));
    defer count_dev.free();
    var count_host = try cuda.HostBuffer.alloc(driver, 8 * (world + 1));
    defer count_host.free();
    const counts = count_host.slice(i64);
    counts[0] = @intCast(keys.len);
    try count_dev.uploadAsync(0, std.mem.sliceAsBytes(counts[0..1]), stream.handle);
    try f.comm.allGather(count_dev.ptr, count_dev.ptr + 8, 1, .i64, stream.handle);
    try count_dev.downloadAsync(8, std.mem.sliceAsBytes(counts[1 .. 1 + world]), stream.handle);
    try stream.synchronize();
    var largest: usize = 0;
    for (counts[1 .. 1 + world]) |n| largest = @max(largest, @as(usize, @intCast(n)));
    if (largest == 0) return;

    const stride = largest * @sizeOf(Digest);
    var key_dev = try cuda.DeviceBuffer.alloc(driver, stride * (world + 1));
    defer key_dev.free();
    var key_host = try cuda.HostBuffer.alloc(driver, stride * (world + 1));
    defer key_host.free();
    const all = key_host.slice(Digest);
    @memset(key_host.slice(u8)[0..stride], 0);
    @memcpy(all[0..keys.len], keys);
    std.mem.sort(Digest, all[0..keys.len], {}, less);
    try key_dev.uploadAsync(0, key_host.slice(u8)[0..stride], stream.handle);
    try f.comm.allGather(key_dev.ptr, key_dev.ptr + stride, stride, .u8, stream.handle);
    try key_dev.downloadAsync(stride, key_host.slice(u8)[stride .. stride * (world + 1)], stream.handle);
    try stream.synchronize();
    for (keys, mask) |key, *keep| {
        keep.* = true;
        for (counts[1 .. 1 + world], 0..) |n, rank| {
            const peer = all[(rank + 1) * largest .. (rank + 1) * largest + @as(usize, @intCast(n))];
            if (!contains(peer, key)) {
                keep.* = false;
                break;
            }
        }
    }
}
