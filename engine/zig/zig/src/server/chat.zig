//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const prompt_mod = @import("prompt.zig");
const log = @import("log.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const family_mod = @import("family.zig");
const spark = @import("spark.zig");
const chunk_plan = @import("chunk_plan.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };

/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// The request body (a family reads fields of its own from it, such as ``response_format``).
    body: Value = .null,
    /// ``parallel_tool_calls: false``: the reply keeps its first call (a family applies it while streaming).
    single_call: bool = false,
};

/// A streamed piece: content text (a string) or a delta object (reasoning or tool calls).
pub const Sink = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, delta: Value) error{Closed}!void,
};

pub const Reply = struct {
    content: []const u8,
    stop_sequence: ?[]const u8,
    reasoning: ?[]const u8,
    tool_calls_streamed: bool,
    finish_reason: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    runtime: Value,
    speculative: Value,
    /// A family's parsed calls (OpenAI tool-call objects); null: the route parses the content itself.
    calls: ?[]Value = null,
    /// The engine run as the Spark server reports it (``tensorfold``), the counted ids, image parts replaced.
    outcome: ?spark.Outcome = null,
    token_ids: []const u32 = &.{},
    images_omitted: u32 = 0,
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

/// Events the engine thread hands this request, read by the request's own thread.
const Mailbox = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    tokens: std.ArrayList(u32) = .empty,
    chunks: std.ArrayList(usize) = .empty, // where each round's tokens end
    cached: ?u32 = null,
    prefilled_ns: ?i96 = null,
    finished: bool = false,
    reason: api.Reason = .stop,
    stats: api.Stats = .{},
    widths: []u32 = &.{},
    raised: []bool = &.{},
    telemetry: []u8 = "",
    message: []u8 = "",

    fn onEvent(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
        const m: *Mailbox = @ptrCast(@alignCast(ctx));
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        switch (event.*) {
            .prefilled => |cached| {
                m.cached = cached;
                m.prefilled_ns = std.Io.Clock.awake.now(m.io).toNanoseconds();
            },
            .tokens => |t| {
                m.tokens.appendSlice(m.gpa, t) catch {};
                m.chunks.append(m.gpa, m.tokens.items.len) catch {};
            },
            .finished => |f| {
                m.finished = true;
                m.reason = f.reason;
                m.stats = f.stats;
                m.widths = m.gpa.dupe(u32, f.stats.prefill_widths) catch &.{};
                m.raised = m.gpa.dupe(bool, f.stats.prefill_raised) catch &.{};
                m.telemetry = m.gpa.dupe(u8, f.stats.telemetry_json) catch "";
                m.message = m.gpa.dupe(u8, f.message) catch "";
            },
        }
        m.cond.signal(m.io);
    }

    fn deinit(m: *Mailbox) void {
        m.tokens.deinit(m.gpa);
        m.chunks.deinit(m.gpa);
        m.gpa.free(m.widths);
        m.gpa.free(m.raised);
        m.gpa.free(m.telemetry);
        m.gpa.free(m.message);
    }
};

const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// Python's ``bool(fields.get(key))`` on the sampling fields.
fn flag(f: Value, key: []const u8) ?bool {
    const v = f.get(key) orelse return null;
    if (v == .null) return null;
    return v.truthy();
}

/// A request rendered and checked but not submitted; it holds the preparing count until generate or release.
pub const Prepared = struct {
    input: Input,
    request: api.Request,
    prompt_len: usize,
    received: i96,
    thinking: bool,
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    stops: fields_mod.Stops,
    preparing: bool,
    images_omitted: u32 = 0,
    /// the prompt's prepared images (TF_DSV41_IMAGES=native), held until the reply ends
    images: ?family_mod.Images = null,
};

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.
pub fn prepare(srv: *Server, cx: *Cx, input: Input, gone: anytype) Failure!Prepared {
    const a = cx.a;
    const io = srv.io;
    const received = nowNs(io);
    const f = input.fields;
    const stops_opt = try fields_mod.stopOptions(cx, f);
    var limit: i64 = @max(1, if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens);
    const priority = f.get("priority");
    const background = prompt_mod.isTitle(input.messages, input.tools.len > 0) or
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background"));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer release(srv, preparing);
    if (gone.check()) return error.Cancelled;
    var thinking = flag(f, "enable_thinking") orelse srv.config.enable_thinking;
    if (input.prompt != null) thinking = false;
    const asked: ?[]const u8 = if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null;
    const effort: ?[]const u8 = if (srv.family) |fam| asked orelse srv.config.reasoning_effort orelse fam.defaultEffort() else srv.effortFor(asked);
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    errdefer if (rendered.images) |im| im.release();
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0 and srv.config.wire == .spark) {
        // ``context_problem``: the prompt plus the reply asked for (1 without max_tokens) must fit the limit
        const reply_max: ?u64 = if (input.max_tokens) |m| (if (m > 0) @intCast(m) else null) else null;
        if (@as(u64, @intCast(n)) + (reply_max orelse 1) > @as(u64, @intCast(window))) return cx.fail(.context_length, "{s}", .{try spark.contextMessage(a, rendered.ids.len, reply_max, srv.info.context_window, input.prompt == null)});
        limit = @min(limit, window - n);
    } else if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.context_fitted) ", the most this server's memory budget fits" else "", n });
        if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        limit = @min(limit, room);
    }
    const system_len: usize = if (input.prompt != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .images = if (rendered.images) |im| im.held else null,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .shared_prefixes = shared.items,
        // a cut just before the conversation's own text: fresh sessions resume their whole harness
        .chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, rendered.ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, rendered.ids.len, srv.chunks.min_chunk),
        .tools_json = if (input.tools.len > 0) try json.stringify(a, .{ .array = @constCast(input.tools) }, .{ .ascii = false }) else "",
    };
    try srv.checkFeatures(cx, f, input.tools.len > 0, thinking, rendered.ids, input.tools, &request);
    if (thinking) {
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
        }
    }
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .effort = effort, .sampled = sampling != null, .drafts = drafts, .stops = stops_opt, .preparing = preparing, .images_omitted = rendered.images_omitted, .images = rendered.images };
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const thinking = prepared.thinking;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer release(srv, preparing);
    defer if (prepared.images) |im| im.release(); // after the engine's finished event (the loop below waits for it)
    if (request.background) {
        while (true) {
            const now = nowNs(io);
            const waiting = now < received + 150 * std.time.ns_per_ms or (srv.preparing.load(.acquire) > 0 and now < received + 2 * std.time.ns_per_s);
            if (!waiting) break;
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
    }
    if (gone.check()) return error.Cancelled;
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    // a family matches stop strings in the answer itself (``push``); the generic path, in the raw text on the engine's thread
    if (stops_opt.strings.len > 0 and srv.family == null) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    const submitted = nowNs(io);
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    // the Spark surface: the run in /health's bookkeeping and the request log, from here to its end
    const hid: ?u64 = if (srv.health) |h| h.begin(prepared.prompt_len) else null;
    const ticket = if (srv.reqlog) |l| l.begin(request.prompt) else null;
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| {
        if (hid) |x| srv.health.?.end(x, .{ .failed = .{ .value_error = true, .message = "busy" } });
        if (srv.reqlog) |l| l.end(ticket, input.body, .{ .chat = input.prompt == null, .finish = null, .completion_tokens = null, .outcome = null, .err = "EngineBusy", .thinking = thinking and input.prompt == null, .max_tokens_eff = effMax(srv, input) });
        return switch (e) {
            error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
            error.Closed => cx.fail(.other, "the scheduler is closed", .{}),
        };
    };
    release(srv, preparing); // a background request waits only while a foreground one prepares
    preparing = false;
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .reply_id = input.id, .sink = sink, .thinking = thinking or reply_text.isChannel(srv.markers), .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .hid = hid, .ticket = ticket, .submitted = submitted, .prompt_len = prepared.prompt_len };
    defer srv.noteRequest(prepared.prompt_len, gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    const result: Failure!Reply = blk: {
        if (srv.family) |fam| {
            gen.reader = fam.reader(a, thinking, input.tools, input.single_call) catch |e| break :blk e;
        } else if (sink != null and input.tools.len > 0) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        break :blk gen.finish(cx, prepared.prompt_len, received, submitted, thinking, effort, prepared.sampled, prepared.drafts, prepared.images_omitted);
    };
    if (result) |_| {} else |e| logEnded(input.id, e, cx.message, prepared.prompt_len, gen.collected.items.len, seconds(nowNs(io) - received));
    if (hid != null or ticket != null) sparkEnd(srv, &gen, input, thinking, result, cx.message);
    return result;
}

/// ``max_tokens or max_completion_tokens or the server's default``: the request log's ``max_tokens_eff``.
fn effMax(srv: *const Server, input: Input) i64 {
    if (input.max_tokens) |m| if (m > 0) return m;
    return srv.config.default_max_tokens;
}

/// The Spark surface's end of a run: /health's ``end`` (a cancelled run counts as finished, an engine failure is
/// fatal) and the request log's line.
fn sparkEnd(srv: *Server, g: *Generation, input: Input, thinking: bool, result: Failure!Reply, message: []const u8) void {
    var outcome: ?spark.Outcome = null;
    var failed: ?[]const u8 = null;
    if (result) |reply| outcome = reply.outcome else |e| switch (e) {
        error.Cancelled => {
            if (!g.engine_done) g.cancel();
            outcome = g.outcome(true);
        },
        error.OutOfMemory => failed = "MemoryError",
        else => failed = "RuntimeError",
    }
    if (failed != null and !g.engine_done) g.cancel();
    if (g.hid) |x| srv.health.?.end(x, if (outcome) |o| .{ .ok = o } else .{ .failed = .{ .value_error = false, .message = if (message.len > 0) message else failed.? } });
    const l = srv.reqlog orelse return;
    const chat_ = input.prompt == null;
    const reply: ?Reply = result catch null;
    l.end(g.ticket, input.body, .{
        .chat = chat_,
        .finish = if (reply) |r| r.finish_reason else null,
        .completion_tokens = if (reply) |r| r.completion_tokens else if (outcome) |o| o.completion else null,
        .outcome = outcome,
        .err = failed,
        .thinking = thinking and chat_,
        .max_tokens_eff = effMax(srv, input),
    });
}

/// The ``ended`` line of a submitted reply that ends without one: its client left, or it failed.
fn logEnded(id: []const u8, e: Failure, message: []const u8, prompt: usize, tokens: usize, after: f64) void {
    const why: ?[]const u8 = switch (e) {
        error.Cancelled => null,
        error.Refused => message, // once submitted, only the engine's own failure refuses
        else => @errorName(e),
    };
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, why, prompt, tokens, after)});
}

/// The reply to ``input``; ``sink`` hears the stream (null: not streamed). ``gone`` says the client left.
pub fn run(srv: *Server, cx: *Cx, input: Input, sink: ?Sink, gone: anytype) Failure!Reply {
    if (srv.config.compact_at == null) return generate(srv, cx, try prepare(srv, cx, input, gone), sink, gone);
    return @import("compact.zig").run(srv, cx, input, sink, gone);
}

/// Give back a foreground request's preparing count: at its submit, or when it is never generated.
pub fn release(srv: *Server, preparing: bool) void {
    if (preparing) _ = srv.preparing.fetchSub(1, .acq_rel);
}

/// The text a stream has sent: Python's ``streamed = visible``, grown by its delta while it only grows.
const Shown = struct {
    text: std.ArrayList(u8) = .empty,
    chars: usize = 0,
    at: ?[*]const u8 = null, // where the last shown text lay; the decode buffer only grows, so that prefix holds
    extends: bool = false,

    /// Python's ``now[len(shown):]``; ``stable``: ``now`` lies in the append-only decode buffer.
    fn after(s: *Shown, now: []const u8, stable: bool) []const u8 {
        const sent = s.text.items;
        s.extends = (stable and s.at == now.ptr and now.len >= sent.len) or std.mem.startsWith(u8, now, sent);
        if (!s.extends) return reply_text.afterChars(now, s.chars);
        if (stable) s.at = now.ptr; // the buffer moved but kept its bytes: the next check is free again
        return now[sent.len..];
    }

    fn set(s: *Shown, a: Allocator, now: []const u8, delta: []const u8) Allocator.Error!void {
        if (s.extends) {
            try s.text.appendSlice(a, delta);
            s.chars += reply_text.charCount(delta);
        } else {
            s.text = .empty;
            try s.text.appendSlice(a, now);
            s.chars = reply_text.charCount(now);
        }
        s.at = now.ptr;
    }
};

/// The token loop and the text it streams.
const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    sink: ?Sink,
    thinking: bool,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: Shown = .{},
    streaming_done: bool = false,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    /// when the stream last carried bytes (a keepalive goes out after a silence)
    wrote_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    // a family's reply: its reader, the answer so far, how much of it streamed, and a stop string's hit
    reader: ?family_mod.Reader = null,
    answer: std.ArrayList(u8) = .empty,
    answer_sent: usize = 0,
    deltas: std.ArrayList(Value) = .empty,
    stop_hit: bool = false,
    stop_at: usize = 0,
    // the Spark surface: the run's /health entry and request-log ticket
    hid: ?u64 = null,
    ticket: ?*spark.RequestLog.Ticket = null,
    submitted: i96 = 0,
    prompt_len: usize = 0,

    /// The run's stats as the Spark server reports them (the Python batch engine's ``job.stats``).
    fn outcome(g: *const Generation, cancelled: bool) spark.Outcome {
        const m = g.box;
        const s = m.stats;
        const end_ns = nowNs(g.srv.io);
        return .{
            .prompt = g.prompt_len,
            .cached = m.cached orelse 0,
            .ttft_s = if (g.first_ns) |f| @max(0, seconds(f - g.submitted)) else null,
            .cancelled = cancelled,
            .completion = m.tokens.items.len,
            .prefill_s = s.prefill_seconds,
            .decode_s = s.decode_seconds orelse if (m.prefilled_ns) |p| @max(0, seconds(end_ns - p)) else null,
            .rounds = s.rounds,
            .drafted = s.drafted,
            .accepted = s.accepted,
        };
    }

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// What closes a call the model's end token left open (``tool_parse.closeCall``); nothing when a stop string, the length or ``ignore_eos`` ended the reply instead.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        if (g.reader) |r| return g.commitFamily(r, chunk);
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const all = try g.visible.extend(g.a, g.srv.text, decodeText, fresh.items);
        const text = g.stops.visible(all, true);
        if (g.visible.pending()) return; // a character still split across tokens
        var answer = text;
        if (g.thinking) {
            const split = try reply_text.splitThinking(g.a, text, false, g.srv.markers);
            const piece = g.streamed_reasoning.after(split.reasoning, true);
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, split.reasoning, piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
            }
            answer = split.answer;
        }
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false) else answer;
        const vis = reply_text.streamingVisible(shown);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, .{ .string = delta });
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        if (g.srv.reqlog) |l| l.first(g.ticket);
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.wrote_ns = nowNs(g.srv.io);
    }

    /// Ends the request and waits for the engine; one it ended unfinished counts as a disconnect, as the Mac scheduler counts it.
    fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    fn loop(g: *Generation, gone: anytype) Failure!void {
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                g.chunk_index += 1;
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch null;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check()) {
                g.cancel();
                return error.Cancelled;
            }
            if (chunk == null and @hasDecl(@TypeOf(gone), "beat")) {
                // queueing and a long prefill send nothing: a client that times out a silent stream would leave
                const now = nowNs(g.srv.io);
                const since = g.wrote_ns orelse now;
                if (g.wrote_ns == null) g.wrote_ns = now;
                if (gone.beat(now - since) catch {
                    g.cancel();
                    return error.Cancelled;
                }) g.wrote_ns = now;
            }
            const tokens = chunk orelse continue;
            // /health hears the tokens once they are read (a stop string they hold has cancelled the run by then)
            defer if (g.hid) |x| g.srv.health.?.progress(x, tokens.len);
            if (g.reason == null) try g.commit(tokens);
        }
    }

    /// A family's round: tokens committed (an end-of-sentence id ends the reply and shows no text), read by the
    /// family's reader, and its deltas pushed as the Python app's ``push`` does.
    fn commitFamily(g: *Generation, r: family_mod.Reader, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        const start = g.collected.items.len;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        for (g.collected.items[start..]) |t| if (!g.isEos(t)) try r.push(&.{t});
        if (r.pending()) return; // a character still split across tokens
        g.deltas.clearRetainingCapacity();
        try r.feed(false, &g.deltas);
        try g.pushFamily(g.deltas.items, false);
    }

    /// Any end-of-sentence id, ``ignore_eos`` or not: never text.
    fn isEos(g: *const Generation, t: u32) bool {
        return std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// ``push``: reasoning and calls go out as they come; the answer is held where a stop string may begin, and cut
    /// where one does (which ends the reply).
    fn pushFamily(g: *Generation, deltas: []const Value, finished: bool) Failure!void {
        for (deltas) |d| switch (d) {
            .string => |t| try g.answer.appendSlice(g.a, t),
            else => if (g.sink) |sink| if (!g.streaming_done or finished) try g.emit(sink, d),
        };
        var full = g.answer.items;
        const stops = g.stops.strings;
        if (stops.len > 0 and !g.stop_hit) {
            var longest: usize = 0;
            for (stops) |st| longest = @max(longest, st.len);
            const from = (g.answer_sent + 1) -| longest;
            var cut: ?usize = null;
            for (stops) |st| if (std.mem.indexOfPos(u8, full, @min(from, full.len), st)) |at| {
                if (cut == null or at < cut.?) cut = at;
            };
            if (cut) |c| {
                g.answer.shrinkRetainingCapacity(c);
                full = g.answer.items;
                g.stop_hit = true;
                g.stop_at = g.collected.items.len;
                if (g.reason == null) {
                    g.reason = "stop";
                    g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
                }
            }
        }
        var upto = full.len;
        if (!finished and !g.stop_hit) {
            var hold: usize = 0;
            for (stops) |st| {
                var k = @min(st.len - 1, full.len);
                while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, full, st[0..k])) break;
                hold = @max(hold, k);
            }
            upto -= hold;
        }
        if (upto > g.answer_sent) {
            if (g.sink) |sink| try g.emit(sink, .{ .string = full[g.answer_sent..upto] });
            g.answer_sent = upto;
        }
    }

    /// A family's finished reply: the stream's last deltas, then the parts (``app._run``'s end).
    fn finishFamily(g: *Generation, r: family_mod.Reader) Failure!struct { content: []const u8, reasoning: ?[]const u8, calls: []Value, reason: []const u8, used: usize } {
        const streaming = g.sink != null;
        g.deltas.clearRetainingCapacity();
        try r.feed(true, &g.deltas);
        try g.pushFamily(g.deltas.items, true);
        const parsed = try r.parse();
        var content: []const u8 = undefined;
        var calls: []Value = undefined;
        if (streaming) {
            calls = try r.calls();
            content = g.answer.items[0..g.answer_sent];
        } else {
            calls = parsed.calls;
            content = parsed.content;
            for (g.stops.strings) |st| if (std.mem.indexOf(u8, content, st)) |at| {
                content = content[0..at];
                if (!g.stop_hit) {
                    g.stop_hit = true;
                    g.stop_at = g.collected.items.len;
                }
            };
        }
        const t = g.collected.items;
        const reason: []const u8 = if (calls.len > 0) "tool_calls" else if (g.stop_hit or (!g.ignore_eos and t.len > 0 and g.isEos(t[t.len - 1]))) "stop" else "length";
        if (g.srv.family) |fam| fam.remember(parsed.reasoning, content, calls);
        // a reply with tools on offer that ended without a call but holds tool markup: what the parser saw (once,
        // at most 300 bytes of the reply; the window is cut at UTF-8 boundaries and escaped by the log line)
        if (calls.len == 0 and g.tools.len > 0) if (r.markup() catch null) |m|
            std.log.warn("tool markup with no call: request {s}, finish {s}: {s}; reply bytes {d}..: {f}", .{ g.reply_id, reason, m.reason, m.at, std.zig.fmtString(m.window) });
        return .{ .content = content, .reasoning = if (parsed.reasoning.len > 0) parsed.reasoning else null, .calls = calls, .reason = reason, .used = if (g.stop_hit and g.stop_at > 0) g.stop_at else t.len };
    }

    fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, submitted: i96, thinking: bool, effort: ?[]const u8, exact: bool, drafts: bool, images_omitted: u32) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done) g.drain();
        const reason_generic: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .cancelled => "cancelled",
            .failed => {
                // a drain's deadline ended it: the 503's code, not a failure's
                if (g.srv.draining.load(.seq_cst) and std.mem.eql(u8, m.message, errors.restarting)) return cx.fail(.restarting, "{s}", .{errors.restarting});
                return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed");
            },
        };
        var reason = reason_generic;
        var raw: []const u8 = "";
        var content: []const u8 = undefined;
        var reasoning: ?[]const u8 = null;
        var calls: ?[]Value = null;
        var used = g.collected.items.len;
        if (g.reader) |r| {
            const f = try g.finishFamily(r);
            content = f.content;
            reasoning = f.reasoning;
            calls = f.calls;
            used = f.used;
            if (!(m.reason == .cancelled and g.reason == null)) reason = f.reason;
        } else {
            const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
            raw = try g.srv.text.decode(a, content_tokens);
            const visible = g.stops.visible(raw, false);
            // closed before the think split, so a call ending an unclosed think block is the answer
            const close = try g.closeCall(visible);
            const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
            content = text;
            if (g.thinking) {
                const split = try reply_text.splitThinking(a, text, true, g.srv.markers);
                const r = reply_text.pyStrip(split.reasoning);
                reasoning = if (r.len > 0) r else null;
                content = split.answer;
            } else {
                const h = reply_text.parseHarmony(text);
                content = h.content;
                reasoning = h.reasoning;
            }
            if (g.sink) |sink| {
                const thought = g.streamed_reasoning.text.items;
                if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                    try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
                const shown = if (g.calls != null) try reply_text.hideToolCalls(a, content, true) else content;
                const sent = g.streamed.text.items;
                if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) try g.emit(sink, .{ .string = shown[sent.len..] });
                if (g.calls) |*c| if (close.len > 0) {
                    // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                    var out: std.ArrayList(Value) = .empty;
                    try c.feed(if (g.thinking) content else text, &out);
                    for (out.items) |d| try g.emit(sink, d);
                };
            }
        }
        const finished_ns = nowNs(g.srv.io);
        const prefilled = m.prefilled_ns;
        const total = seconds(finished_ns - submitted);
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, total) });
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        if (images_omitted > 0) try runtime.put(a, "images_omitted", try json.intValue(a, images_omitted));
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        if (m.telemetry.len > 0) if ((try json.parse(a, m.telemetry)) == .ok) try spec.put(a, "proposer", (try json.parse(a, m.telemetry)).ok);
        const think_end: ?u32 = if (thinking) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = m.cached orelse 0,
            .completion_tokens = used,
            .reasoning_tokens = reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .calls = calls,
            .outcome = if (g.srv.config.wire == .spark) g.outcome(m.reason == .cancelled and g.reason == null) else null,
            .token_ids = g.collected.items[0..@min(used, g.collected.items.len)],
            .images_omitted = images_omitted,
        };
        if (std.mem.eql(u8, reason, "length") and thinking and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted });
        return reply;
    }
};

/// The request's stop strings as the engine checks them after each token: the newest tokens' text holds one.
const StopHook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    fn check(ctx: *anyopaque, emitted: []const u32) bool {
        const h: *StopHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const tail = emitted[emitted.len -| h.stops.tail()..];
        const text = h.srv.text.decode(arena.allocator(), tail) catch return false;
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};

fn decodeText(t: anytype, a: Allocator, tokens: []const u32) ![]u8 {
    return t.decode(a, tokens);
}

/// ``{key: text}``: a streamed delta (role, content or reasoning_content).
pub fn deltaOf(a: Allocator, key: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = text });
    return .{ .object = o };
}

/// The reply's token ids hashed: drafted and ``"draft": false`` replies must match.
pub fn tokenSha(tokens: []const u32) [12]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [16]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    var out: [12]u8 = undefined;
    const hex = std.fmt.bytesToHex(d[0..6].*, .lower);
    @memcpy(&out, &hex);
    return out;
}
