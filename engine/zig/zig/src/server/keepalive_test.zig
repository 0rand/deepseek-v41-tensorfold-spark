//! A silent stream keeps its client: keepalives on both routes before the first token; a failed one ends the request.
const std = @import("std");
const serve_mod = @import("dsv41_serve");
const deepseek = @import("deepseek.zig");
const spark = @import("spark.zig");
const d = @import("drain_test.zig");
const testing = std.testing;

/// A raw request whose response is read until the connection ends.
fn messagesBody(a: std.mem.Allocator, marker: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"x\",\"max_tokens\":50,\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"go {s}\"}}]}}", .{marker});
}

/// Sends `body` streamed and hangs up once the response head arrived and the request reached `slow`.
fn leave(a: std.mem.Allocator, io: std.Io, port: u16, body: []const u8, slow: *d.Slow) !void {
    const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
    const stream = try addr.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    const head = try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
    var wbuf: [4096]u8 = undefined;
    var w = std.Io.net.Stream.Writer.init(stream, io, &wbuf);
    try w.interface.writeAll(head);
    try w.interface.flush();
    var rbuf: [64]u8 = undefined;
    var r = std.Io.net.Stream.Reader.init(stream, io, &rbuf);
    _ = try r.interface.peek(1);
    // the head goes out before the engine has the request: a hang-up before that leaves nothing to cancel
    d.waitActive(io, slow, 1);
}

test "a silent engine's stream gets keepalives before its first token (chat: SSE comments, messages: ping events)" {
    try testing.expect((spark.Settings{}).keepalive_s > 0 and (spark.Settings{}).keepalive_s <= 15);
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
    try env.put("TF_DSV41_DRAIN_S", "5");
    try env.put("TF_DSV41_SSE_KEEPALIVE_S", "0.1");
    // only a failed write ends a request, so the keepalive's own write is what notices a client gone
    try env.put("GLM53_TF_DISCONNECT", "0");
    var slow: d.Slow = .{ .gpa = gpa, .io = io, .tok = tok, .eos = eos };
    var x: d.Served = .{};
    const t = try d.start(gpa, io, &env, &slow, ds, &x);
    const port = d.listen_port.load(.acquire);

    const chat = try d.request(a, io, port, "POST", "/v1/chat/completions", try d.chatBody(a, "@wait", true));
    try testing.expectEqual(@as(u16, 200), chat.status);
    const first_token = std.mem.indexOf(u8, chat.body, "\" ok\"") orelse {
        std.debug.print("{s}\n", .{chat.body[0..@min(chat.body.len, 1500)]});
        return error.NoToken;
    };
    try testing.expect(std.mem.count(u8, chat.body[0..first_token], ": keepalive\n\n") >= 3);
    try testing.expect(std.mem.indexOf(u8, chat.body, "data: [DONE]") != null);

    const msg = try d.request(a, io, port, "POST", "/v1/messages", try messagesBody(a, "@wait"));
    try testing.expectEqual(@as(u16, 200), msg.status);
    const start = std.mem.indexOf(u8, msg.body, "event: message_start") orelse return error.NoStart;
    const ping = std.mem.indexOf(u8, msg.body, "event: ping\ndata: {\"type\": \"ping\"}") orelse return error.NoPing;
    const delta = std.mem.indexOf(u8, msg.body, "event: content_block_delta") orelse return error.NoDelta;
    try testing.expect(start < ping and ping < delta);

    // a client that leaves a silent stream: the next keepalive fails to write and the engine's request is cancelled
    var idle: u32 = 0;
    while (slow.active.load(.acquire) > 0 and idle < 300) : (idle += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    try leave(a, io, port, try d.chatBody(a, "@hang", true), &slow);
    var waited: u32 = 0;
    while (slow.cancelled.load(.acquire) == 0 and waited < 1000) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    errdefer std.debug.print("not cancelled: {d} active, {d} waits\n", .{ slow.active.load(.acquire), waited });
    try testing.expect(slow.cancelled.load(.acquire) != 0);

    try std.posix.raise(.TERM);
    t.join();
    try testing.expectEqual(@as(u8, 0), x.code);
}
