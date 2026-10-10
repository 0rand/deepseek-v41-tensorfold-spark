//! A stop's drain over HTTP: serve.run with a real SIGTERM, the Spark wire, a slow stub engine, the mini tokenizer.
const std = @import("std");
const api = @import("engine_api");
const serve_mod = @import("dsv41_serve");
const deepseek = @import("deepseek.zig");
const serve = @import("serve.zig");
const cli = @import("cli.zig");
const routes = @import("routes.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// Stops mid-reply: "@short" ends after 12 tokens, "@long" when halted or cancelled, "@hold" waits for `release`.
pub const Slow = struct {
    gpa: Allocator,
    io: std.Io,
    tok: *serve_mod.tokenizer.Tokenizer,
    eos: u32,
    halted: std.atomic.Value(bool) = .init(false),
    reason: []const u8 = "",
    active: std.atomic.Value(u32) = .init(0),
    halts: u32 = 0,
    closes: u32 = 0,
    /// halts before the first close (the halt must come first)
    halts_at_close: u32 = 0,
    cancelled: std.atomic.Value(u64) = .init(0),
    /// "@hold" replies stay silent until this is set
    release: std.atomic.Value(bool) = .init(false),

    fn engine(s: *Slow) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn self(ctx: *anyopaque) *Slow {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .name = "slow" };
    }

    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).cancelled.store(id, .release);
    }

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .running = self(ctx).active.load(.acquire) };
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s = self(ctx);
        _ = s.active.fetchAdd(1, .acq_rel);
        const t = std.Thread.spawn(.{}, run, .{ s, id, r, sink }) catch {
            _ = s.active.fetchSub(1, .acq_rel);
            return error.Busy;
        };
        t.detach();
    }

    fn run(s: *Slow, id: api.Id, r: *const api.Request, sink: api.Sink) void {
        defer _ = s.active.fetchSub(1, .acq_rel);
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const prompt = s.tok.decodeAlloc(a, r.prompt) catch "";
        const long = std.mem.indexOf(u8, prompt, "@long") != null;
        const hang = std.mem.indexOf(u8, prompt, "@hang") != null;
        const wait = std.mem.indexOf(u8, prompt, "@wait") != null;
        const hold = std.mem.indexOf(u8, prompt, "@hold") != null;
        const word = s.tok.encodeAlloc(a, if (long) " la" else " ok") catch &.{};
        // a long prefill: nothing for 800 ms ("@wait"), or nothing until stopped ("@hang")
        var quiet: u32 = 0;
        while (hang or (wait and quiet < 80) or (hold and !s.release.load(.acquire))) : (quiet += 1) {
            if (s.halted.load(.acquire)) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = s.reason } });
            if (s.cancelled.load(.acquire) == id) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            std.Io.sleep(s.io, .fromMilliseconds(10), .awake) catch {};
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var n: usize = 0;
        while (true) : (n += 1) {
            if (s.halted.load(.acquire)) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = s.reason } });
            if (s.cancelled.load(.acquire) == id) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            if (!long and n == 12) {
                sink.event(sink.ctx, id, &.{ .tokens = &.{s.eos} });
                return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
            }
            sink.event(sink.ctx, id, &.{ .tokens = word });
            std.Io.sleep(s.io, .fromMilliseconds(15), .awake) catch {};
        }
    }

    fn halt(ctx: *anyopaque, reason: []const u8) void {
        const s = self(ctx);
        s.reason = reason;
        s.halts += 1;
        s.halted.store(true, .release);
        // a round boundary: every reply has ended before halt returns
        while (s.active.load(.acquire) > 0) std.Io.sleep(s.io, .fromMilliseconds(1), .awake) catch {};
    }

    fn close(ctx: *anyopaque) void {
        const s = self(ctx);
        if (s.closes == 0) s.halts_at_close = s.halts;
        s.closes += 1;
    }
};

pub var listen_port: std.atomic.Value(u16) = .init(0);

fn onListen(port: u16) void {
    listen_port.store(port, .release);
}

pub const Reply = struct { status: u16, head: []const u8, body: []const u8 };

/// One request on its own connection (closed by us after it), the whole response read.
pub fn request(a: Allocator, io: std.Io, port: u16, method: []const u8, path: []const u8, body: []const u8) !Reply {
    const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
    const stream = try addr.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    const head = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, body.len, body });
    var wbuf: [4096]u8 = undefined;
    var w = std.Io.net.Stream.Writer.init(stream, io, &wbuf);
    try w.interface.writeAll(head);
    try w.interface.flush();
    var rbuf: [16384]u8 = undefined;
    var r = std.Io.net.Stream.Reader.init(stream, io, &rbuf);
    const text = try r.interface.allocRemaining(a, .unlimited);
    const split = std.mem.indexOf(u8, text, "\r\n\r\n") orelse return error.Head;
    var payload: []const u8 = text[split + 4 ..];
    if (std.mem.indexOf(u8, text[0..split], "chunked") != null) payload = try unchunk(a, payload);
    return .{ .status = try std.fmt.parseInt(u16, text[9..12], 10), .head = text[0..split], .body = payload };
}

fn unchunk(a: Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const eol = std.mem.indexOfPos(u8, s, i, "\r\n") orelse break;
        const n = try std.fmt.parseInt(usize, std.mem.trim(u8, s[i..eol], " "), 16);
        if (n == 0) break;
        try out.appendSlice(a, s[eol + 2 .. eol + 2 + n]);
        i = eol + 2 + n + 2;
    }
    return out.items;
}

pub const Background = struct {
    a: Allocator,
    io: std.Io,
    port: u16,
    body: []const u8,
    path: []const u8 = "/v1/chat/completions",
    reply: ?Reply = null,
    failed: bool = false,

    fn run(b: *Background) void {
        b.reply = request(b.a, b.io, b.port, "POST", b.path, b.body) catch {
            b.failed = true;
            return;
        };
    }
};

pub fn chatBody(a: Allocator, marker: []const u8, stream: bool) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"go {s}\"}}],\"stream\":{},\"max_tokens\":4000}}", .{ marker, stream });
}

pub const Served = struct {
    code: u8 = 255,
    fn run(x: *Served, gpa: Allocator, io: std.Io, setup: serve.Setup) void {
        x.code = serve.run(gpa, io, .{ .host = "127.0.0.1", .port = 0 }, setup);
    }
};

/// serve.run on its own thread with `drain_s`, the engine `slow`; the port once it listens.
pub fn start(gpa: Allocator, io: std.Io, env: *std.process.Environ.Map, slow: *Slow, ds: *deepseek.DeepSeek, x: *Served) !std.Thread {
    listen_port.store(0, .release);
    const t = try std.Thread.spawn(.{}, Served.run, .{ x, gpa, io, serve.Setup{
        .engine = slow.engine(),
        .text = ds.text(),
        .served = "deepseek-v41",
        .family = ds.family_(),
        .wire = .spark,
        .environ = env,
        .on_listen = onListen,
        .stop = .{ .ctx = slow, .halt = Slow.halt, .close = Slow.close },
    } });
    while (listen_port.load(.acquire) == 0) std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    return t;
}

pub fn waitActive(io: std.Io, slow: *Slow, n: u32) void {
    while (slow.active.load(.acquire) < n) std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
}

test "SIGTERM drains: 503 + Retry-After and /health draining while a reply finishes; at the deadline the rest end \"server restarting\" at a round boundary, then the engine closes" {
    const gpa = testing.allocator;
    const io = testing.io;
    const tok = try serve_mod.tokenizer.Tokenizer.parse(gpa, serve_mod.fixtures.mini_tokenizer);
    defer tok.deinit();
    const eos = tok.tokenId("<｜end▁of▁sentence｜>").?;
    const ds = try deepseek.DeepSeek.init(gpa, tok, .{ .eos = &.{eos} });
    defer ds.deinit(false);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("TENSORFOLD_NO_LIVE", "1");

    // 1. the reply in progress finishes within the drain: no halt, a whole reply; new work refused meanwhile
    {
        try env.put("TF_DSV41_DRAIN_S", "10");
        var slow: Slow = .{ .gpa = gpa, .io = io, .tok = tok, .eos = eos };
        var x: Served = .{};
        const t = try start(gpa, io, &env, &slow, ds, &x);
        const port = listen_port.load(.acquire);
        // held through the checks below: a reply that ended first would end the drain and close the listener
        var bg: Background = .{ .a = a, .io = io, .port = port, .body = try chatBody(a, "@hold", true) };
        const bt = try std.Thread.spawn(.{}, Background.run, .{&bg});
        errdefer slow.release.store(true, .release);
        waitActive(io, &slow, 1);
        try std.posix.raise(.TERM);
        // the drain has begun once a new request is refused
        var refused: Reply = undefined;
        while (true) {
            refused = try request(a, io, port, "POST", "/v1/chat/completions", try chatBody(a, "@short", false));
            if (refused.status == 503) break;
            try testing.expectEqual(@as(u16, 200), refused.status); // only before the signal lands
        }
        try testing.expect(std.mem.indexOf(u8, refused.head, "Retry-After: 30") != null);
        try testing.expect(std.mem.indexOf(u8, refused.body, "\"code\": \"server_restarting\"") != null);
        const anth = try request(a, io, port, "POST", "/v1/messages", "{\"model\":\"x\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        try testing.expectEqual(@as(u16, 503), anth.status);
        try testing.expect(std.mem.indexOf(u8, anth.body, "overloaded_error") != null);
        const health = try request(a, io, port, "GET", "/health", "");
        try testing.expectEqual(@as(u16, 503), health.status);
        try testing.expect(std.mem.indexOf(u8, health.head, "Retry-After: 30") != null);
        try testing.expect(std.mem.indexOf(u8, health.body, "\"status\": \"draining\"") != null);
        try testing.expect(std.mem.indexOf(u8, health.body, "\"ok\": false") != null);
        slow.release.store(true, .release);
        bt.join();
        t.join();
        try testing.expect(!bg.failed);
        try testing.expectEqual(@as(u16, 200), bg.reply.?.status);
        try testing.expect(std.mem.indexOf(u8, bg.reply.?.body, "\"finish_reason\": \"stop\"") != null);
        try testing.expect(std.mem.indexOf(u8, bg.reply.?.body, "restarting") == null);
        try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, bg.reply.?.body, "\r\n"), "data: [DONE]"));
        try testing.expectEqual(@as(u8, 0), x.code);
        try testing.expectEqual(@as(u32, 0), slow.halts);
        try testing.expectEqual(@as(u32, 1), slow.closes);
    }
    // 2. replies that outlast the drain end "server restarting"; the engine is halted, then closed once
    {
        try env.put("TF_DSV41_DRAIN_S", "0.3");
        var slow: Slow = .{ .gpa = gpa, .io = io, .tok = tok, .eos = eos };
        var x: Served = .{};
        const t = try start(gpa, io, &env, &slow, ds, &x);
        const port = listen_port.load(.acquire);
        var streamed: Background = .{ .a = a, .io = io, .port = port, .body = try chatBody(a, "@long", true) };
        var whole: Background = .{ .a = a, .io = io, .port = port, .body = try chatBody(a, "@long", false) };
        var msgs: Background = .{ .a = a, .io = io, .port = port, .path = "/v1/messages", .body = "{\"model\":\"x\",\"max_tokens\":4000,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"go @long\"}]}" };
        const st = try std.Thread.spawn(.{}, Background.run, .{&streamed});
        const wt = try std.Thread.spawn(.{}, Background.run, .{&whole});
        const mt = try std.Thread.spawn(.{}, Background.run, .{&msgs});
        waitActive(io, &slow, 3);
        try std.posix.raise(.TERM);
        st.join();
        wt.join();
        mt.join();
        t.join();
        try testing.expectEqual(@as(u8, 0), x.code);
        try testing.expectEqual(@as(u32, 1), slow.halts);
        try testing.expectEqual(@as(u32, 1), slow.closes);
        try testing.expectEqual(@as(u32, 1), slow.halts_at_close);
        try testing.expectEqualStrings(routes.restarting, slow.reason);
        const sb = streamed.reply.?.body;
        try testing.expectEqual(@as(u16, 200), streamed.reply.?.status);
        // the 503's own code and type, in Python's in-stream error event shape, then [DONE]
        try testing.expect(std.mem.indexOf(u8, sb, "data: {\"error\": {\"message\": \"" ++ routes.restarting ++ "\", \"type\": \"server_error\", \"code\": \"server_restarting\"}}\n\ndata: [DONE]") != null);
        try testing.expect(std.mem.indexOf(u8, sb, "RuntimeError") == null);
        try testing.expectEqual(@as(u16, 503), whole.reply.?.status);
        try testing.expect(std.mem.indexOf(u8, whole.reply.?.body, "\"code\": \"server_restarting\"") != null);
        try testing.expect(std.mem.indexOf(u8, msgs.reply.?.body, "event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"" ++ routes.restarting) != null);
    }
}

/// The race seam's state: the stop delivered between a request's count and its drain check.
const Gap = struct {
    var io: std.Io = undefined;
    var port: u16 = 0;
    var fired: std.atomic.Value(bool) = .init(false);
    var saw_draining: bool = false;
    var drain_waited: bool = false;
    var served: *Served = undefined;

    fn hook() void {
        if (fired.swap(true, .acq_rel)) return;
        std.posix.raise(.TERM) catch return;
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        // the drain has begun (it reads the count after it sets the flag): /health says so
        var tries: u32 = 0;
        while (tries < 500) : (tries += 1) {
            const h = request(arena.allocator(), io, port, "GET", "/health", "") catch break;
            if (h.status == 503 and std.mem.indexOf(u8, h.body, "draining") != null) {
                saw_draining = true;
                break;
            }
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
        // this request is counted: the drain may not finish while it is between its count and its check
        std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
        drain_waited = served.code == 255;
    }
};

test "a stop that lands between a request's count and its drain check: the request is refused (503) and the drain waited for it, never lost" {
    const gpa = testing.allocator;
    const io = testing.io;
    const tok = try serve_mod.tokenizer.Tokenizer.parse(gpa, serve_mod.fixtures.mini_tokenizer);
    defer tok.deinit();
    const eos = tok.tokenId("<｜end▁of▁sentence｜>").?;
    const ds = try deepseek.DeepSeek.init(gpa, tok, .{ .eos = &.{eos} });
    defer ds.deinit(false);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("TENSORFOLD_NO_LIVE", "1");
    try env.put("TF_DSV41_DRAIN_S", "10");
    var slow: Slow = .{ .gpa = gpa, .io = io, .tok = tok, .eos = eos };
    var x: Served = .{};
    const t = try start(gpa, io, &env, &slow, ds, &x);
    Gap.io = io;
    Gap.port = listen_port.load(.acquire);
    Gap.served = &x;
    Gap.fired.store(false, .release);
    routes.counted_hook = Gap.hook;
    defer routes.counted_hook = null;
    const r = try request(a, io, Gap.port, "POST", "/v1/chat/completions", try chatBody(a, "@short", false));
    t.join();
    try testing.expect(Gap.fired.load(.acquire));
    try testing.expect(Gap.saw_draining);
    try testing.expect(Gap.drain_waited);
    try testing.expectEqual(@as(u16, 503), r.status);
    try testing.expect(std.mem.indexOf(u8, r.body, "server_restarting") != null);
    try testing.expectEqual(@as(u32, 0), slow.closes -| 1); // closed once, after
    try testing.expectEqual(@as(u32, 0), slow.active.load(.acquire)); // nothing reached the engine
    try testing.expectEqual(@as(u8, 0), x.code);
}
