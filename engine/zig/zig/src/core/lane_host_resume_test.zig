//! A request whose client leaves mid-prompt keeps its finished prefix through LaneHost: the same prompt resumes there.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const lane_host = @import("lane_host.zig");
const LaneHost = lane_host.LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a prompt whose client leaves mid-prefill is kept: the same prompt resumes there, a different one does not" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    // 3 rows a round; a request resumes from the longest prefix the backend kept (the session store's find)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64, cached: u64, rec: usize, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        target: *lanes.fake.Fake,
        rounds: std.atomic.Value(u32) = .init(0),
        /// each admission's first piece start and piece count, in admission order (keys are reused addresses)
        recs: std.ArrayList(struct { first: ?u64 = null, count: u32 = 0 }) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            var cached: u64 = 0;
            for (p.target.saves.items) |v| if (v.len > cached and v.len < ask.prompt.len and std.mem.startsWith(u32, ask.prompt, v)) {
                cached = v.len;
            };
            p.recs.append(gpa, .{}) catch return .refuse;
            p.seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len, .done = cached, .cached = cached, .rec = p.recs.items.len - 1 }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    const r = &p.recs.items[x.rec];
                    if (r.first == null) r.first = x.done;
                    r.count += 1;
                    x.done = end;
                }
                if (x.done == x.n - 1) {
                    try finals.append(gpa, x.key);
                    x.decoding = true;
                }
            };
            _ = p.rounds.fetchAdd(1, .acq_rel);
        }
        fn leaving(ctx: *anyopaque, key: usize) ?u64 {
            for (self(ctx).seqs.items) |x| if (x.key == key) return if (!x.decoding and x.done > x.cached) x.done else null;
            return null;
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {}; // a round takes time: the cancel lands mid-prompt
        }
    };
    var pl: Plan = .{ .target = &target };
    defer pl.seqs.deinit(gpa);
    defer pl.recs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after, .leaving = Plan.leaving } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1 + i % 11);
    var other = prompt;
    other[5] = 12;
    var boxes: [3]Box = .{ .{}, .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    const e = host.engine();
    try host.start();
    defer host.stop();
    // 1: its client leaves after some rounds of its prompt
    const r1: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try e.submit(1, &r1, .{ .ctx = &boxes[0], .event = Box.event });
    while (pl.rounds.load(.acquire) < 20) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, boxes[0].wait());
    try std.testing.expectEqual(@as(usize, 1), target.saves.items.len);
    const kept = target.saves.items[0].len;
    try std.testing.expect(kept >= 60 and kept < 599 and kept % 3 == 0);
    try std.testing.expectEqualSlices(u32, prompt[0..kept], target.saves.items[0]);
    // 2: the same prompt resumes from what was kept: only the rest is prefilled
    const r2: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try e.submit(2, &r2, .{ .ctx = &boxes[1], .event = Box.event });
    try std.testing.expectEqual(Reason.length, boxes[1].wait());
    try std.testing.expectEqual(@as(?u64, kept), pl.recs.items[1].first);
    try std.testing.expectEqual(@as(u32, @intCast((599 - kept + 2) / 3)), pl.recs.items[1].count);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (boxes[1].tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    // 3: a prompt that differs early does not reuse it
    const r3: Request = .{ .prompt = &other, .max_tokens = 4 };
    try e.submit(3, &r3, .{ .ctx = &boxes[2], .event = Box.event });
    try std.testing.expectEqual(Reason.length, boxes[2].wait());
    try std.testing.expectEqual(@as(?u64, 0), pl.recs.items[2].first);
    try std.testing.expectEqual(@as(u32, 200), pl.recs.items[2].count);
}
