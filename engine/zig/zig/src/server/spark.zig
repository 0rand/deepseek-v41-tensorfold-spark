//! The Spark server's HTTP surface: what TensorFold's two-Spark servers (GLM-5.3 and DeepSeek-V4.1 prod,
//! ``glm5_next/spark/server.py`` with ``health.py`` and ``reqlog.py``) answer, for clients, dashboards and the ops
//! scripts written against them. ``Wire.spark`` selects it (``Config.wire``); ``Wire.tensorfold`` is upstream's.
//!
//! - **Errors**: ``{"error": {"message", "type", "code"?, "param"?}}``; an unknown path ``{"error": "not found"}``;
//!   the context limit in OpenAI's and vLLM's words, which clients parse to compact.
//! - **/health** (``Health``): ok / fatal / stalled / inflight and upstream's live totals (``drafted_total``,
//!   ``accepted_total``, ...); strict mode answers 503 after a fatal engine error and refuses new completions.
//! - **/metrics**: the ``tensorfold_*`` rows. **/v1/models**: ``max_model_len`` / ``context_length``, aliases.
//! - **Request log** (``RequestLog``): one JSON line a request, token counts, timings and hashes only.
//!
//! Knobs are the Python names: ``GLM53_TF_X`` wins, else ``TF_DSV41_X`` (``app.alias_knobs``): REQUEST_LOG (with
//! GLM53_TF_REQUEST_LOG_MB / _KEEP / _PROMPTS / _SALT), DISCONNECT (1), HEALTH (basic | strict), STALL_S (0), and
//! GLM53_TF_STALL_PREFILL_TPS (200).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// ``spark`` replies use the served ``model``, no ``completion_tokens_details``, the engine stats under ``tensorfold``.
pub const Wire = enum { tensorfold, spark };
/// /health's ``streams``: a batch engine's slots decoding, prefilling, and in all.
pub const Streams = struct { decoding: u32, prefilling: u32, max: u32 };
pub const Mode = enum { basic, strict };

/// The random hex digits after a reply id's ``chatcmpl-`` / ``cmpl-`` (``uuid4().hex[:24]``).
pub const id_digits = 24;

pub const LogSettings = struct {
    path: []const u8,
    max_bytes: u64 = 64 << 20,
    keep: u32 = 3,
    window: u32 = 64,
    salt: []const u8 = "",
};

pub const Settings = struct {
    disconnect: bool = true,
    health: Mode = .basic,
    stall_s: f64 = 0,
    stall_prefill_tps: f64 = 200,
    log: ?LogSettings = null,
    /// TF_DSV41_DRAIN_S / GLM53_TF_DRAIN_S: seconds requests in progress may finish after a stop signal (serve.zig).
    drain_s: f64 = 20,
    /// TF_DSV41_SSE_KEEPALIVE_S / GLM53_TF_SSE_KEEPALIVE_S: a silent stream's keepalive period (0: none)
    keepalive_s: f64 = 15,
    /// TF_DSV41_STALL_ROUND_S / GLM53_TF_STALL_ROUND_S: /health 503 when work is in flight and no round ended this long
    stall_round_s: f64 = 120,
    /// TF_DSV41_STALL_SLOW_S / GLM53_TF_STALL_SLOW_S: /health 503 when the rounds' pace has been slow this long
    stall_slow_s: f64 = 180,

    /// The knobs from ``env``; ``problem`` names a bad one.
    pub fn fromEnv(env: ?*const std.process.Environ.Map, problem: *[]const u8) error{Invalid}!Settings {
        var s: Settings = .{};
        const m = env orelse return s;
        if (knob(m, "DISCONNECT")) |v| {
            const t = trimmed(v, "1");
            if (!std.mem.eql(u8, t, "0") and !std.mem.eql(u8, t, "1")) return bad(problem, "GLM53_TF_DISCONNECT / TF_DSV41_DISCONNECT: expected 0 or 1");
            s.disconnect = std.mem.eql(u8, t, "1");
        }
        if (knob(m, "HEALTH")) |v| {
            const t = trimmed(v, "basic");
            s.health = if (std.ascii.eqlIgnoreCase(t, "basic")) .basic else if (std.ascii.eqlIgnoreCase(t, "strict")) .strict else return bad(problem, "GLM53_TF_HEALTH / TF_DSV41_HEALTH: expected one of basic, strict");
        }
        if (knob(m, "STALL_S")) |v| s.stall_s = try nonNegative(v, 0, problem, "GLM53_TF_STALL_S: must be a number >= 0");
        if (knob(m, "STALL_ROUND_S")) |v| s.stall_round_s = try nonNegative(v, 120, problem, "TF_DSV41_STALL_ROUND_S / GLM53_TF_STALL_ROUND_S: seconds, a number >= 0");
        if (knob(m, "STALL_SLOW_S")) |v| s.stall_slow_s = try nonNegative(v, 180, problem, "TF_DSV41_STALL_SLOW_S / GLM53_TF_STALL_SLOW_S: seconds, a number >= 0");
        if (knob(m, "SSE_KEEPALIVE_S")) |v| s.keepalive_s = try nonNegative(v, 15, problem, "TF_DSV41_SSE_KEEPALIVE_S / GLM53_TF_SSE_KEEPALIVE_S: seconds, a number >= 0");
        if (knob(m, "DRAIN_S")) |v| s.drain_s = try nonNegative(v, 20, problem, "TF_DSV41_DRAIN_S / GLM53_TF_DRAIN_S: seconds, a number >= 0");
        if (m.get("GLM53_TF_STALL_PREFILL_TPS")) |v| s.stall_prefill_tps = try nonNegative(v, 200, problem, "GLM53_TF_STALL_PREFILL_TPS: must be a number >= 0");
        if (knob(m, "REQUEST_LOG")) |v| {
            const path = std.mem.trim(u8, v, " \t");
            if (path.len > 0 and !std.mem.eql(u8, path, "0")) {
                var l: LogSettings = .{ .path = path };
                if (m.get("GLM53_TF_REQUEST_LOG_MB")) |x| {
                    const mb = std.fmt.parseFloat(f64, trimmed(x, "64")) catch return bad(problem, "GLM53_TF_REQUEST_LOG_MB: a number > 0");
                    if (!(mb > 0)) return bad(problem, "GLM53_TF_REQUEST_LOG_MB must be > 0");
                    l.max_bytes = @intFromFloat(mb * (1 << 20));
                }
                if (m.get("GLM53_TF_REQUEST_LOG_KEEP")) |x| l.keep = std.fmt.parseInt(u32, trimmed(x, "3"), 10) catch return bad(problem, "GLM53_TF_REQUEST_LOG_KEEP: an integer >= 0");
                if (m.get("GLM53_TF_REQUEST_LOG_PROMPTS")) |x| l.window = std.fmt.parseInt(u32, trimmed(x, "64"), 10) catch return bad(problem, "GLM53_TF_REQUEST_LOG_PROMPTS: an integer >= 0");
                if (m.get("GLM53_TF_REQUEST_LOG_SALT")) |x| l.salt = x;
                s.log = l;
            }
        }
        return s;
    }

    fn knob(m: *const std.process.Environ.Map, comptime name: []const u8) ?[]const u8 {
        return m.get("GLM53_TF_" ++ name) orelse m.get("TF_DSV41_" ++ name);
    }

    fn trimmed(v: []const u8, default: []const u8) []const u8 {
        const t = std.mem.trim(u8, v, " \t");
        return if (t.len == 0) default else t;
    }

    fn bad(problem: *[]const u8, message: []const u8) error{Invalid} {
        problem.* = message;
        return error.Invalid;
    }

    fn nonNegative(v: []const u8, default: f64, problem: *[]const u8, message: []const u8) error{Invalid}!f64 {
        const t = std.mem.trim(u8, v, " \t");
        if (t.len == 0) return default;
        const x = std.fmt.parseFloat(f64, t) catch return bad(problem, message);
        if (!(x >= 0)) return bad(problem, message);
        return x;
    }
};

/// Python's ``round(x, n)`` (correctly rounded through the decimal text).
pub fn round(x: f64, comptime n: u8) f64 {
    var buf: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{n}) ++ "}", .{x}) catch return x;
    return std.fmt.parseFloat(f64, text) catch x;
}

/// A JSON number as Python's ``dict`` held it: an int when integral and ``integral``, else a float.
fn number(a: Allocator, x: f64, integral: bool) Allocator.Error!Value {
    if (integral and x == @trunc(x) and @abs(x) < 9.0e15) return json.intValue(a, @as(i64, @intFromFloat(x)));
    return .{ .float = x };
}

/// What one request's engine run reports (the Python batch engine's ``job.stats``, in its key order).
pub const Outcome = struct {
    prompt: usize,
    cached: u32 = 0,
    ttft_s: ?f64 = null,
    cancelled: bool = false,
    /// tokens the engine delivered (an EOS and tokens past a stop string included)
    completion: usize = 0,
    prefill_s: ?f64 = null,
    decode_s: ?f64 = null,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,

    /// The reply's ``tensorfold`` object.
    pub fn value(o: Outcome, a: Allocator) Allocator.Error!*json.Object {
        const s = try json.newObject(a);
        try s.put(a, "prompt", try json.intValue(a, o.prompt));
        try s.put(a, "cached", try json.intValue(a, o.cached));
        if (o.ttft_s) |t| try s.put(a, "ttft_s", .{ .float = round(t, 4) });
        try s.put(a, "finish", .{ .string = if (o.cancelled) "cancelled" else "done" });
        try s.put(a, "completion", try json.intValue(a, o.completion));
        if (o.prefill_s) |p| try s.put(a, "prefill_s", .{ .float = round(p, 4) });
        if (o.decode_s) |d| try s.put(a, "decode_s", .{ .float = round(d, 4) });
        try s.put(a, "rounds", try json.intValue(a, o.rounds));
        try s.put(a, "drafted", try json.intValue(a, o.drafted));
        try s.put(a, "accepted", try json.intValue(a, o.accepted));
        return s;
    }
};

// -- /health and /metrics -------------------------------------------------------------------------------------------

/// ``health.Health``: requests in flight, the first fatal engine error, counters and upstream's live totals.
pub const Health = struct {
    gpa: Allocator,
    io: std.Io,
    mode: Mode,
    stall_s: f64,
    prefill_tps: f64,
    stall_round_s: f64,
    stall_slow_s: f64,
    started: i96,
    mutex: std.Io.Mutex = .init,
    fatal: ?[]u8 = null,
    inflight: std.AutoArrayHashMapUnmanaged(u64, Entry) = .empty,
    next_id: u64 = 0,
    last_done: ?i96 = null,
    c: Counters = .{},
    t: Totals = .{},

    const Entry = struct { prompt: f64, start: i96, last: i96, tokens: f64 = 0 };
    const Counters = struct { requests: f64 = 0, errors: f64 = 0, rejected: f64 = 0, prompt_tokens: f64 = 0, completion_tokens: f64 = 0, cached_tokens: f64 = 0, decode_rounds: f64 = 0, decode_seconds: f64 = 0, prefill_seconds: f64 = 0 };
    const Totals = struct { requests_total: f64 = 0, prompt_tokens_total: f64 = 0, completion_tokens_total: f64 = 0, prefill_seconds_total: f64 = 0, decode_seconds_total: f64 = 0, cached_tokens_total: f64 = 0, rounds_total: f64 = 0, drafted_total: f64 = 0, accepted_total: f64 = 0 };

    /// How a run ended: its stats, or an error (``value_error``: the request's own fault, not the engine's).
    pub const End = union(enum) { ok: Outcome, failed: struct { value_error: bool, message: []const u8 } };

    pub fn init(gpa: Allocator, io: std.Io, s: Settings) Health {
        return .{ .gpa = gpa, .io = io, .mode = s.health, .stall_s = s.stall_s, .prefill_tps = s.stall_prefill_tps, .stall_round_s = s.stall_round_s, .stall_slow_s = s.stall_slow_s, .started = now(io) };
    }

    pub fn deinit(h: *Health) void {
        if (h.fatal) |f| h.gpa.free(f);
        h.inflight.deinit(h.gpa);
    }

    fn now(io: std.Io) i96 {
        return std.Io.Clock.awake.now(io).toNanoseconds();
    }

    fn secs(ns: i96) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e9;
    }

    pub fn begin(h: *Health, prompt: usize) u64 {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const id = h.next_id;
        h.next_id += 1;
        const t = now(h.io);
        h.inflight.put(h.gpa, id, .{ .prompt = @floatFromInt(prompt), .start = t, .last = t }) catch {};
        return id;
    }

    pub fn progress(h: *Health, id: u64, n: usize) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const e = h.inflight.getPtr(id) orelse return;
        e.last = now(h.io);
        e.tokens += @floatFromInt(n);
    }

    pub fn end(h: *Health, id: u64, how: End) void {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const entry = h.inflight.fetchSwapRemove(id);
        h.last_done = now(h.io);
        const t = &h.t;
        t.requests_total += 1;
        if (entry) |e| {
            t.prompt_tokens_total += e.value.prompt;
            t.completion_tokens_total += e.value.tokens;
        }
        h.c.requests += 1;
        switch (how) {
            .failed => |f| {
                h.c.errors += 1;
                if (!f.value_error and h.fatal == null) h.fatal = h.gpa.dupe(u8, f.message[0..@min(f.message.len, 500)]) catch null;
            },
            .ok => |o| {
                const cached: f64 = @floatFromInt(o.cached);
                t.prefill_seconds_total += o.prefill_s orelse 0;
                t.decode_seconds_total += o.decode_s orelse 0;
                t.cached_tokens_total += cached;
                t.rounds_total += @floatFromInt(o.rounds);
                t.drafted_total += @floatFromInt(o.drafted);
                t.accepted_total += @floatFromInt(o.accepted);
                h.c.prompt_tokens += if (entry) |e| e.value.prompt else 0;
                h.c.completion_tokens += @floatFromInt(o.completion);
                h.c.cached_tokens += cached;
                h.c.decode_rounds += @floatFromInt(o.rounds);
                h.c.decode_seconds += o.decode_s orelse 0;
                h.c.prefill_seconds += o.prefill_s orelse 0;
            },
        }
    }

    /// Why a new completion must not start (strict mode after a fatal error), or null.
    pub fn reject(h: *Health, a: Allocator) ?[]const u8 {
        if (h.mode != .strict) return null;
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const f = h.fatal orelse return null;
        h.c.rejected += 1;
        return std.fmt.allocPrint(a, "the engine failed earlier ({s}); restart both ranks", .{f}) catch "the engine failed earlier; restart both ranks";
    }

    const Stalled = struct { prompt: f64, quiet_s: f64, allowed_s: f64 };

    fn stalled(h: *Health, a: Allocator, t: i96) Allocator.Error![]Stalled {
        var out: std.ArrayList(Stalled) = .empty;
        if (h.stall_s <= 0) return out.items;
        for (h.inflight.values()) |e| {
            var allow = h.stall_s;
            if (e.tokens == 0 and h.prefill_tps > 0) allow += e.prompt / h.prefill_tps;
            const quiet = secs(t - e.last);
            if (quiet > allow) try out.append(a, .{ .prompt = e.prompt, .quiet_s = round(quiet, 1), .allowed_s = round(allow, 1) });
        }
        return out.items;
    }

    /// Why the engine is stuck (no round with work in flight, or a slow pace held too long), or null.
    pub fn stuck(h: *const Health, a: Allocator, st: api.Status, inflight: usize) Allocator.Error!?[]const u8 {
        const p = st.pace orelse return null;
        const busy = inflight > 0 or st.running > 0 or st.waiting > 0;
        if (busy and h.stall_round_s > 0) if (p.last_round_age_s) |age| if (age > h.stall_round_s)
            return try std.fmt.allocPrint(a, "no round for {d:.0} s with work in flight", .{age});
        if (h.stall_slow_s > 0 and p.slow_s > h.stall_slow_s)
            return try std.fmt.allocPrint(a, "rounds slower than their baseline for {d:.0} s (decode {d:.3} s vs {d:.3}; prefill {d:.0} vs {d:.0} rows/s)", .{ p.slow_s, p.decode_s, p.decode_base_s, p.prefill_rows_s, p.prefill_base_rows_s });
        return null;
    }

    /// ``GET /health``: the status code and body. ``streams``: the engine's slots (null: not a batch engine).
    pub fn status(h: *Health, a: Allocator, streams: ?Streams, context: u32) Allocator.Error!struct { code: u16, body: Value } {
        const o = try json.newObject(a);
        h.mutex.lockUncancelable(h.io);
        const t = now(h.io);
        const st = try h.stalled(a, t);
        const ok = h.fatal == null and st.len == 0;
        var oldest: f64 = 0;
        var running_tokens: f64 = 0;
        for (h.inflight.values()) |e| {
            oldest = @max(oldest, secs(t - e.start));
            running_tokens += e.tokens;
        }
        const running = h.inflight.count();
        try o.put(a, "ok", .{ .bool = ok });
        try o.put(a, "mode", .{ .string = @tagName(h.mode) });
        try o.put(a, "uptime_s", .{ .float = round(secs(t - h.started), 1) });
        try o.put(a, "inflight", try json.intValue(a, running));
        try o.put(a, "oldest_s", .{ .float = round(oldest, 1) });
        try o.put(a, "idle_s", if (h.last_done != null and running == 0) .{ .float = round(secs(t - h.last_done.?), 1) } else .null);
        try o.put(a, "requests", try number(a, h.c.requests, true));
        try o.put(a, "errors", try number(a, h.c.errors, true));
        if (h.fatal) |f| try o.put(a, "fatal", .{ .string = try a.dupe(u8, f) });
        if (st.len > 0) {
            const list = try a.alloc(Value, st.len);
            for (st, list) |s, *slot| {
                const x = try json.newObject(a);
                try x.put(a, "prompt", .{ .float = s.prompt });
                try x.put(a, "quiet_s", .{ .float = s.quiet_s });
                try x.put(a, "allowed_s", .{ .float = s.allowed_s });
                slot.* = .{ .object = x };
            }
            try o.put(a, "stalled", .{ .array = list });
        }
        try o.put(a, "backend", .{ .string = "tensorfold" });
        try o.put(a, "busy", .{ .bool = running > 0 });
        try o.put(a, "requests_running", try json.intValue(a, running));
        const tt = h.t;
        h.mutex.unlock(h.io);
        inline for (@typeInfo(Totals).@"struct".field_names) |name| {
            var x: f64 = @field(tt, name);
            if (comptime std.mem.eql(u8, name, "completion_tokens_total")) x += running_tokens;
            try o.put(a, name, try number(a, if (x == @trunc(x)) x else round(x, 6), true));
        }
        if (streams) |s| {
            const x = try json.newObject(a);
            try x.put(a, "decoding", try json.intValue(a, s.decoding));
            try x.put(a, "prefilling", try json.intValue(a, s.prefilling));
            try x.put(a, "max", try json.intValue(a, s.max));
            try o.put(a, "streams", .{ .object = x });
        }
        if (context > 0) try o.put(a, "context_length", try json.intValue(a, context));
        return .{ .code = if (h.mode == .strict and !ok) 503 else 200, .body = .{ .object = o } };
    }

    /// ``GET /metrics``: Prometheus text, one ``model`` label.
    pub fn metrics(h: *Health, w: *std.Io.Writer, served: []const u8) std.Io.Writer.Error!void {
        h.mutex.lockUncancelable(h.io);
        const t = now(h.io);
        const c = h.c;
        const inflight = h.inflight.count();
        var stalled_n: usize = 0;
        if (h.stall_s > 0) {
            var scratch: [4096]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&scratch);
            stalled_n = if (h.stalled(fba.allocator(), t)) |st| st.len else |_| 0;
        }
        const fatal: f64 = if (h.fatal != null) 1 else 0;
        const uptime = secs(t - h.started);
        h.mutex.unlock(h.io);
        const Row = struct { []const u8, []const u8, []const u8, f64 };
        const rows = [_]Row{
            .{ "tensorfold_requests_total", "counter", "completions finished (errors included)", c.requests },
            .{ "tensorfold_request_errors_total", "counter", "completions that raised", c.errors },
            .{ "tensorfold_requests_rejected_total", "counter", "completions refused after a fatal error", c.rejected },
            .{ "tensorfold_prompt_tokens_total", "counter", "prompt tokens of finished completions", c.prompt_tokens },
            .{ "tensorfold_cached_tokens_total", "counter", "prompt tokens resumed instead of prefilled", c.cached_tokens },
            .{ "tensorfold_completion_tokens_total", "counter", "generated tokens", c.completion_tokens },
            .{ "tensorfold_decode_rounds_total", "counter", "decode rounds (tokens / rounds: draft acceptance)", c.decode_rounds },
            .{ "tensorfold_decode_seconds_total", "counter", "seconds decoding", c.decode_seconds },
            .{ "tensorfold_prefill_seconds_total", "counter", "seconds prefilling", c.prefill_seconds },
            .{ "tensorfold_requests_inflight", "gauge", "completions running now", @floatFromInt(inflight) },
            .{ "tensorfold_requests_stalled", "gauge", "running completions past GLM53_TF_STALL_S", @floatFromInt(stalled_n) },
            .{ "tensorfold_engine_fatal", "gauge", "1 once the engine raised (restart both ranks)", fatal },
            .{ "tensorfold_uptime_seconds", "gauge", "seconds since the server started", uptime },
        };
        for (rows) |r| {
            try w.print("# HELP {s} {s}\n# TYPE {s} {s}\n{s}{{model=\"", .{ r[0], r[2], r[0], r[1], r[0] });
            for (served) |ch| switch (ch) {
                '\\' => try w.writeAll("\\\\"),
                '"' => try w.writeAll("\\\""),
                else => try w.writeByte(ch),
            };
            if (r[3] == @trunc(r[3]) and @abs(r[3]) < 9.0e15) try w.print("\"}} {d}\n", .{@as(i64, @intFromFloat(r[3]))}) else try w.print("\"}} {d:.6}\n", .{r[3]});
        }
    }
};

// -- replies --------------------------------------------------------------------------------------------------------

/// ``context_problem``'s message: OpenAI's and vLLM's words, then how to raise the limit.
pub fn contextMessage(a: Allocator, prompt: usize, asked: ?u64, limit: u32, chat: bool) Allocator.Error![]const u8 {
    const where = if (chat) "messages" else "prompt";
    const need = prompt + (asked orelse 1);
    const what = if (asked) |n|
        try std.fmt.allocPrint(a, "However, you requested {d} tokens ({d} in the {s}, {d} in the completion). Please reduce the length of the {s} or completion.", .{ need, prompt, where, n, where })
    else
        try std.fmt.allocPrint(a, "However, your {s} resulted in {d} tokens. Please reduce the length of the {s}.", .{ where, prompt, where });
    return std.fmt.allocPrint(a, "This model's maximum context length is {d} tokens. {s} (This TensorFold server was started with --context {d} (CONTEXT); restart both ranks with CONTEXT={d} or more to serve it.)", .{ limit, what, limit, need });
}

/// ``error_body``: ``{"error": {"message", "type", "code"?, "param"?}}``.
pub fn errorBody(a: Allocator, message: []const u8, kind: []const u8, code: ?[]const u8, param: ?[]const u8) Allocator.Error!Value {
    const e = try json.newObject(a);
    try e.put(a, "message", .{ .string = message });
    try e.put(a, "type", .{ .string = kind });
    if (code) |c| try e.put(a, "code", .{ .string = c });
    if (param) |p| try e.put(a, "param", .{ .string = p });
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

/// ``GET /v1/models``: the served name, then each alias, with the context limit as vLLM names it.
pub fn models(a: Allocator, ids: []const []const u8, served: []const u8, created: i64, limit: u32) Allocator.Error!Value {
    const data = try a.alloc(Value, ids.len);
    for (ids, data) |id, *slot| {
        const m = try json.newObject(a);
        try m.put(a, "id", .{ .string = id });
        try m.put(a, "object", .{ .string = "model" });
        try m.put(a, "owned_by", .{ .string = "tensorfold" });
        if (limit > 0) {
            try m.put(a, "created", try json.intValue(a, created));
            try m.put(a, "root", .{ .string = served });
            try m.put(a, "max_model_len", try json.intValue(a, limit));
            try m.put(a, "context_length", try json.intValue(a, limit));
        }
        slot.* = .{ .object = m };
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    return .{ .object = o };
}

/// ``parse_numbers``: the sampling and length fields read strictly (no booleans; ``20.0`` and ``"20"`` are integers,
/// ``20.5`` is not); the first bad one's message.
pub fn numbersProblem(a: Allocator, body: Value) Allocator.Error!?[]const u8 {
    const names = [_]struct { []const u8, bool }{ .{ "temperature", false }, .{ "top_p", false }, .{ "top_k", true }, .{ "seed", true }, .{ "max_tokens", true }, .{ "max_completion_tokens", true } };
    for (names) |n| {
        const v = body.field(n[0]) orelse continue;
        if (!numberOk(v, n[1])) return try std.fmt.allocPrint(a, "{s} must be {s} or null", .{ n[0], if (n[1]) "an integer" else "a finite number" });
    }
    return null;
}

fn numberOk(v: Value, integer: bool) bool {
    switch (v) {
        .int => return true,
        .float => |f| return if (integer) f == @trunc(f) and std.math.isFinite(f) else std.math.isFinite(f),
        .string => |s| {
            const t = std.mem.trim(u8, s, " \t\r\n\x0b\x0c");
            if (integer) {
                const digits = std.mem.trimStart(u8, t, "+-");
                if (digits.len == 0 or digits.len + 1 < t.len) return false;
                for (digits, 0..) |ch, i| if (!(std.ascii.isDigit(ch) or (ch == '_' and i > 0 and i + 1 < digits.len and digits[i - 1] != '_'))) return false;
                return true;
            }
            const f = std.fmt.parseFloat(f64, t) catch return false;
            return std.math.isFinite(f);
        },
        else => return false,
    }
}

/// ``parse_stop``: a string, a list of strings, or null.
pub fn stopProblem(body: Value) ?[]const u8 {
    const v = body.get("stop") orelse return null;
    switch (v) {
        .null, .string => return null,
        .array => |items| {
            for (items) |x| if (x != .string) return "stop must be a string or a list of strings";
            return null;
        },
        else => return "stop must be a string or a list of strings",
    }
}

/// ``token_ids_problem``: a ``/v1/completions`` token-ID prompt that is not one flat list of in-vocabulary ints.
pub fn tokenIdsProblem(a: Allocator, prompt: []const Value, vocab: u32) Allocator.Error!?[]const u8 {
    if (prompt.len == 0) return "prompt must not be empty";
    var lists = true;
    var strings = true;
    for (prompt) |p| {
        lists = lists and p == .array;
        strings = strings and p == .string;
    }
    if (lists or strings) return "one prompt a request: send a string or one list of token ids";
    for (prompt) |p| if (p != .int) return "a token-ID prompt must be a list of integers";
    for (prompt) |p| {
        const n = p.int64() orelse -1;
        if (n < 0 or n >= vocab) return try std.fmt.allocPrint(a, "token id {s} is outside the vocabulary (0 to {d})", .{ p.int, @as(i64, vocab) - 1 });
    }
    return null;
}

// -- the request log ------------------------------------------------------------------------------------------------

/// ``reqlog.RequestLog``: rank 0's request log, one JSON line a request (no text: counts, timings, hashes of ids).
pub const RequestLog = struct {
    gpa: Allocator,
    io: std.Io,
    s: LogSettings,
    /// the chat template's role tokens the tokenizer knows (``<|user|>``, ``<|assistant|>``, ``<|observation|>``)
    roles: Roles,
    mutex: std.Io.Mutex = .init,
    n: u64 = 0,
    live: std.AutoArrayHashMapUnmanaged(u64, *Ticket) = .empty,
    /// the last ``window`` prompts, oldest first
    kept: std.ArrayList(Kept) = .empty,
    fd: ?std.posix.fd_t = null,
    written: u64 = 0,
    warned: bool = false,

    pub const hash_tokens = 4096;
    pub const Roles = struct { user: ?u32 = null, assistant: ?u32 = null, observation: ?u32 = null };
    const Kept = struct { n: u64, ids: []u32, conv: [16]u8 };

    pub const Ticket = struct {
        n: u64,
        start: f64,
        ids: []u32,
        first: ?f64 = null,
        conv: ?[16]u8 = null,
    };

    pub fn init(gpa: Allocator, io: std.Io, s: LogSettings, roles: Roles) RequestLog {
        return .{ .gpa = gpa, .io = io, .s = s, .roles = roles };
    }

    pub fn deinit(l: *RequestLog) void {
        for (l.kept.items) |k| l.gpa.free(k.ids);
        l.kept.deinit(l.gpa);
        for (l.live.values()) |t| l.free(t);
        l.live.deinit(l.gpa);
        if (l.fd) |fd| _ = std.posix.system.close(fd);
    }

    fn free(l: *RequestLog, t: *Ticket) void {
        l.gpa.free(t.ids);
        l.gpa.destroy(t);
    }

    fn unix(l: *const RequestLog) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.real.now(l.io).toNanoseconds())) / 1e9;
    }

    /// A request's prompt ids, before it runs.
    pub fn begin(l: *RequestLog, ids: []const u32) ?*Ticket {
        const t = l.gpa.create(Ticket) catch return null;
        t.* = .{ .n = 0, .start = l.unix(), .ids = l.gpa.dupe(u32, ids) catch {
            l.gpa.destroy(t);
            return null;
        } };
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.n += 1;
        t.n = l.n;
        l.live.put(l.gpa, t.n, t) catch {};
        return t;
    }

    /// The first streamed delta.
    pub fn first(l: *const RequestLog, t: ?*Ticket) void {
        const x = t orelse return;
        if (x.first == null) x.first = l.unix();
    }

    /// blake2b-64 of the ids as little-endian int32, keyed with the salt when set; 16 hex digits.
    pub fn hash(l: *const RequestLog, ids: []const u32) [16]u8 {
        const B = std.crypto.hash.blake2.Blake2b(64);
        var h = B.init(.{ .key = if (l.s.salt.len > 0) l.s.salt[0..@min(l.s.salt.len, 64)] else null });
        if (@import("builtin").cpu.arch.endian() == .little) h.update(std.mem.sliceAsBytes(ids)) else for (ids) |x| {
            var le: [4]u8 = undefined;
            std.mem.writeInt(u32, &le, x, .little);
            h.update(&le);
        }
        var d: [8]u8 = undefined;
        h.final(&d);
        return std.fmt.bytesToHex(d, .lower);
    }

    fn firstOf(ids: []const u32, targets: []const ?u32) ?usize {
        var best: ?usize = null;
        for (targets) |t| if (t) |x| if (std.mem.indexOfScalar(u32, ids[0..(best orelse ids.len)], x)) |i| {
            best = i;
        };
        return best;
    }

    fn conv(l: *const RequestLog, ids: []const u32) [16]u8 {
        if (firstOf(ids, &.{l.roles.assistant})) |at| return l.hash(ids[0 .. at + 1]);
        return l.hash(ids[0..@min(ids.len, hash_tokens)]);
    }

    /// How a request ended, for its line.
    pub const Result = struct {
        chat: bool,
        /// null: no result (it raised)
        finish: ?[]const u8,
        completion_tokens: ?usize,
        outcome: ?Outcome,
        /// the exception class it raised
        err: ?[]const u8 = null,
        thinking: ?bool,
        max_tokens_eff: ?i64,
    };

    /// Writes the request's line (never fails a request) and keeps its prompt for the prefix fields.
    pub fn end(l: *RequestLog, t: ?*Ticket, body: Value, r: Result) void {
        const x = t orelse return;
        var arena = std.heap.ArenaAllocator.init(l.gpa);
        defer arena.deinit();
        if (l.record(arena.allocator(), x, body, r)) |line| l.write(line) else |_| l.warn("the record could not be built");
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        _ = l.live.swapRemove(x.n);
        if (l.s.window == 0) return l.free(x);
        if (l.kept.items.len >= l.s.window) l.gpa.free(l.kept.orderedRemove(0).ids);
        l.kept.append(l.gpa, .{ .n = x.n, .ids = x.ids, .conv = x.conv orelse l.conv(x.ids) }) catch {
            return l.free(x);
        };
        l.gpa.destroy(x);
    }

    fn warn(l: *RequestLog, what: []const u8) void {
        if (l.warned) return;
        l.warned = true;
        std.debug.print("[tensorfold] request log (GLM53_TF_REQUEST_LOG={s}): {s}; further errors not shown\n", .{ l.s.path, what });
    }

    const Prefix = struct { lcp: usize = 0, same: usize = 0, other: usize = 0 };

    /// The longest common prefix with every earlier prompt still kept or in flight, by conversation.
    fn analyse(l: *RequestLog, t: *Ticket) Prefix {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        if (t.conv == null) t.conv = l.conv(t.ids);
        var p: Prefix = .{};
        const fold = struct {
            fn one(q: *Prefix, mine: []const u32, conv_: [16]u8, other: []const u32, oconv: [16]u8) void {
                if (other.len == 0 or mine.len == 0 or other[0] != mine[0]) return;
                const d = std.mem.indexOfDiff(u32, mine, other) orelse mine.len;
                q.lcp = @max(q.lcp, d);
                if (std.mem.eql(u8, &conv_, &oconv)) q.same = @max(q.same, d) else q.other = @max(q.other, d);
            }
        }.one;
        for (l.kept.items) |k| if (k.n < t.n) fold(&p, t.ids, t.conv.?, k.ids, k.conv);
        for (l.live.values()) |o| if (o.n < t.n) {
            if (o.conv == null) o.conv = l.conv(o.ids);
            fold(&p, t.ids, t.conv.?, o.ids, o.conv.?);
        };
        return p;
    }

    fn num(a: Allocator, v: ?f64, comptime digits: u8) Allocator.Error!Value {
        _ = a;
        return if (v) |x| .{ .float = round(x, digits) } else .null;
    }

    fn record(l: *RequestLog, a: Allocator, t: *Ticket, body: Value, r: Result) ![]const u8 {
        const o = try json.newObject(a);
        const st = r.outcome;
        const prompt = t.ids.len;
        const cached: usize = if (st) |s| s.cached else 0;
        const prefill_s: ?f64 = if (st) |s| s.prefill_s else null;
        const decode_s: ?f64 = if (st) |s| s.decode_s else null;
        const done: ?usize = r.completion_tokens;
        const finish: Value = if (r.err != null) .{ .string = "error" } else if (st != null and st.?.cancelled) .{ .string = "cancelled" } else if (r.finish) |f| .{ .string = f } else .null;
        const kw = body.get("chat_template_kwargs") orelse Value.null;
        const asked = (body.field("max_tokens") orelse body.field("max_completion_tokens"));
        const now_s = l.unix();
        try o.put(a, "ts", .{ .float = round(now_s, 3) });
        try o.put(a, "start", .{ .float = round(t.start, 3) });
        try o.put(a, "n", try json.intValue(a, t.n));
        try o.put(a, "kind", .{ .string = if (r.chat) "chat" else "completion" });
        try o.put(a, "prompt", try json.intValue(a, prompt));
        try o.put(a, "cached", try json.intValue(a, cached));
        try o.put(a, "cache_src", .{ .string = if (cached > 0) "slot" else "none" });
        try o.put(a, "prefill_s", try num(a, prefill_s, 4));
        try o.put(a, "prefill_tps", if (prefill_s) |p| (if (p != 0) Value{ .float = round(@as(f64, @floatFromInt(prompt - cached)) / round(p, 4), 1) } else .null) else .null);
        try o.put(a, "queue_s", .null);
        try o.put(a, "first_s", if (t.first) |f| .{ .float = round(f - t.start, 3) } else .null);
        try o.put(a, "decode_tokens", if (done) |d| try json.intValue(a, d) else .null);
        try o.put(a, "decode_s", try num(a, decode_s, 4));
        try o.put(a, "decode_tps", if (decode_s != null and decode_s.? != 0 and done != null and done.? != 0) .{ .float = round(@as(f64, @floatFromInt(done.?)) / round(decode_s.?, 4), 2) } else .null);
        try o.put(a, "tokens_per_round", .null);
        try o.put(a, "rounds", if (st) |s| try json.intValue(a, s.rounds) else .null);
        for ([_][]const u8{ "pieces", "slot", "kv_pages", "kv_free", "marks", "multi" }) |k| try o.put(a, k, .null);
        try o.put(a, "thinking", if (r.thinking) |b| .{ .bool = b } else .null);
        try o.put(a, "effort", if (r.thinking orelse false) (kw.get("reasoning_effort") orelse .null) else .null);
        try o.put(a, "effort_asked", body.get("reasoning_effort") orelse .null);
        try o.put(a, "max_tokens", if (asked != null and asked.?.truthy()) (if (asked.?.int64()) |n| try json.intValue(a, n) else asked.?) else .null);
        try o.put(a, "max_tokens_eff", if (r.max_tokens_eff) |n| try json.intValue(a, n) else .null);
        try o.put(a, "finish", finish);
        try o.put(a, "error", if (r.err) |e| .{ .string = e } else .null);
        try o.put(a, "tools", try json.intValue(a, if (body.field("tools")) |x| (if (x == .array) x.array.len else 0) else 0));
        try o.put(a, "messages", if (r.chat) try json.intValue(a, if (body.field("messages")) |x| (if (x == .array) x.array.len else 0) else 0) else .null);
        try o.put(a, "policy", .null);
        try o.put(a, "fast_prefill", .null);
        const p = l.analyse(t);
        const ids = t.ids;
        const sys_len = firstOf(ids, &.{ l.roles.user, l.roles.assistant, l.roles.observation });
        const head = ids[0..@min(ids.len, hash_tokens)];
        try o.put(a, "head_hash", .{ .string = try a.dupe(u8, &l.hash(head)) });
        try o.put(a, "head_len", try json.intValue(a, head.len));
        try o.put(a, "sys_len", if (sys_len) |s| try json.intValue(a, s) else .null);
        try o.put(a, "sys_hash", if (sys_len != null and sys_len.? > 0) .{ .string = try a.dupe(u8, &l.hash(ids[0..sys_len.?])) } else .null);
        try o.put(a, "conv", .{ .string = try a.dupe(u8, &t.conv.?) });
        try o.put(a, "lcp", try json.intValue(a, p.lcp));
        try o.put(a, "lcp_same", try json.intValue(a, p.same));
        try o.put(a, "lcp_other", try json.intValue(a, p.other));
        const text = try json.stringify(a, .{ .object = o }, .{ .compact = true });
        return std.mem.concat(a, u8, &.{ text, "\n" });
    }

    fn write(l: *RequestLog, line: []const u8) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.writeLocked(line) catch |e| l.warn(@errorName(e));
    }

    fn open(l: *RequestLog) !void {
        if (std.fs.path.dirname(l.s.path)) |d| std.Io.Dir.cwd().createDirPath(l.io, d) catch {};
        const pathz = try l.gpa.dupeSentinel(u8, l.s.path, 0);
        defer l.gpa.free(pathz);
        const fd = try std.posix.openatZ(std.posix.AT.FDCWD, pathz, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .CLOEXEC = true }, 0o644);
        l.fd = fd;
        const end_at = std.posix.system.lseek(fd, 0, std.posix.SEEK.END);
        l.written = if (end_at < 0) 0 else @intCast(end_at);
    }

    fn writeLocked(l: *RequestLog, line: []const u8) !void {
        if (l.fd == null) try l.open();
        if (l.written > 0 and l.written + line.len > l.s.max_bytes) try l.rotate();
        var off: usize = 0;
        while (off < line.len) {
            const rc = std.posix.system.write(l.fd.?, line.ptr + off, line.len - off);
            if (std.posix.errno(rc) != .SUCCESS) return error.WriteFailed;
            off += @intCast(rc);
        }
        l.written += line.len;
    }

    /// ``<file>`` -> ``<file>.1`` (``.1`` -> ``.2`` ..., ``keep`` old files).
    fn rotate(l: *RequestLog) !void {
        _ = std.posix.system.close(l.fd.?);
        l.fd = null;
        const dir = std.Io.Dir.cwd();
        var buf_a: [std.fs.max_path_bytes]u8 = undefined;
        var buf_b: [std.fs.max_path_bytes]u8 = undefined;
        if (l.s.keep == 0) {
            dir.deleteFile(l.io, l.s.path) catch {};
        } else {
            var i: u32 = l.s.keep - 1;
            while (i > 0) : (i -= 1) {
                const src = try std.fmt.bufPrint(&buf_a, "{s}.{d}", .{ l.s.path, i });
                const dst = try std.fmt.bufPrint(&buf_b, "{s}.{d}", .{ l.s.path, i + 1 });
                dir.rename(src, dir, dst, l.io) catch {};
            }
            dir.rename(l.s.path, dir, try std.fmt.bufPrint(&buf_b, "{s}.1", .{l.s.path}), l.io) catch {};
        }
        try l.open();
        l.written = 0;
    }
};

test "contextMessage is the Python server's" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("This model's maximum context length is 512 tokens. However, you requested 585 tokens (73 in the messages, 512 in the completion). Please reduce the length of the messages or completion. (This TensorFold server was started with --context 512 (CONTEXT); restart both ranks with CONTEXT=585 or more to serve it.)", try contextMessage(arena.allocator(), 73, 512, 512, true));
    try std.testing.expectEqualStrings("This model's maximum context length is 10 tokens. However, your prompt resulted in 12 tokens. Please reduce the length of the prompt. (This TensorFold server was started with --context 10 (CONTEXT); restart both ranks with CONTEXT=13 or more to serve it.)", try contextMessage(arena.allocator(), 12, null, 10, false));
}

test "the request log's hash is Python's blake2b(digest_size=8) of int32 ids" {
    // hashlib.blake2b(array("i", [0, 1, 2]).tobytes(), digest_size=8).hexdigest() and with key=b"salt"
    var l = RequestLog.init(std.testing.allocator, std.testing.io, .{ .path = "/dev/null" }, .{});
    defer l.deinit();
    try std.testing.expectEqualStrings("99a5826197698e6d", &l.hash(&.{ 0, 1, 2 }));
    l.s.salt = "salt";
    try std.testing.expectEqualStrings("c90ad2de02f9487a", &l.hash(&.{ 0, 1, 2 }));
}

test "health: stuck when work is in flight and no round ended for the limit, or the pace has been slow too long" {
    const a = std.testing.allocator;
    var h = Health.init(a, std.testing.io, .{});
    defer h.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = arena.allocator();
    // healthy: rounds a moment ago, at their baseline
    var st: api.Status = .{ .running = 1, .pace = .{ .last_round_age_s = 0.2, .decode_s = 0.07, .decode_base_s = 0.07 } };
    try std.testing.expect((try h.stuck(x, st, 1)) == null);
    // no round for longer than the limit with work in flight; idle (nothing in flight) is not stuck
    st.pace.?.last_round_age_s = 121;
    try std.testing.expect(std.mem.startsWith(u8, (try h.stuck(x, st, 1)).?, "no round for 121 s"));
    st.running = 0;
    try std.testing.expect((try h.stuck(x, st, 0)) == null);
    // a measured stall: 380 ms rounds against 73 ms, slow for over three minutes
    st.pace = .{ .last_round_age_s = 0.4, .decode_s = 0.38, .decode_base_s = 0.073, .slow_s = 181 };
    try std.testing.expect(std.mem.startsWith(u8, (try h.stuck(x, st, 1)).?, "rounds slower than their baseline for 181 s"));
    // an engine that keeps no pace is never called stuck
    try std.testing.expect((try h.stuck(x, .{ .running = 3 }, 3)) == null);
}
