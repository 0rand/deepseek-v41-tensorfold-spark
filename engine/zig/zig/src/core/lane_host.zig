//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const pace_mod = @import("pace.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

/// A family's round planner (DeepSeek-V4.1: prod's batcher, families/deepseek_v41/kv/sched.zig): which queued request
/// starts each round, and which prompt rows run between the decode rounds; the backend runs prompts in pieces
/// (lanes.Backend begin / piece / finish). A round: the admissions decided in queue order, then what must run before
/// them (`admitted`: the slots released last round), the admitted streams begun, this round's pieces, a decode round of
/// the streams decoding, the prompts whose rows are in (finals: their first token), then `after`. Keys are the host's
/// handles for its requests.
pub const Rounds = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const Verdict = enum { admit, wait, skip, refuse };
    pub const Piece = struct { key: usize, start: u64, end: u64, save: bool, run: bool = true };
    pub const Ask = struct { key: usize, prompt: []const u32, max_tokens: u32, submitted_s: f64 };

    pub const VTable = struct {
        /// An admission pass starts (a lane is free and requests wait).
        begin_admit: *const fn (ctx: *anyopaque) void,
        /// The next queued request: start it, keep it and those behind it queued this round (wait), look past it
        /// (skip), or refuse it (`words`: why).
        admit: *const fn (ctx: *anyopaque, ask: Ask, words: *[]const u8) Verdict,
        /// The pass is decided: what runs before the admitted streams begin.
        admitted: *const fn (ctx: *anyopaque) anyerror!void,
        /// Around an admitted request's stream begin (`damaged`: its saved prefix could not be restored).
        arm: *const fn (ctx: *anyopaque, key: usize) void,
        began: *const fn (ctx: *anyopaque, key: usize, damaged: bool) void,
        /// This round's pieces, in order, and the requests whose prompts are in.
        plan: *const fn (ctx: *anyopaque, pieces: *std.ArrayList(Piece), finals: *std.ArrayList(usize)) anyerror!void,
        /// A request left: finished, cancelled, failed or refused, started or not.
        left: *const fn (ctx: *anyopaque, key: usize) void,
        /// After the round: the pieces' seconds and the decode round's (finals included).
        after: *const fn (ctx: *anyopaque, piece_s: f64, decode_s: f64) void,
        /// A request cancelled mid-prompt: the row its finished prefix is kept at (a save-only piece), or null.
        leaving: ?*const fn (ctx: *anyopaque, key: usize) ?u64 = null,
    };
};

/// Makes and frees a request's copy proposer (`prompt` and `eos` outlive it).
pub const Proposers = struct {
    ctx: *anyopaque,
    make: *const fn (ctx: *anyopaque, gpa: Allocator, prompt: []const u32, eos: []const u32) anyerror!lanes.proposer.Proposer,
    free: *const fn (ctx: *anyopaque, p: lanes.proposer.Proposer) void,
};

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    /// A family's own copy proposer for each request (null: lanes.SuffixLookup at `min_match`)
    proposers: ?Proposers = null,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    /// the rounds' pace against their baseline, read by /health (under the mutex)
    pace: pace_mod.Pace = .{},
    /// `halt`'s reason: what is left finishes failed with it (null: a stop cancels what is left)
    halt_reason: ?[]const u8 = null,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    /// Runs first on the engine thread: a backend whose API state is per thread (a CUDA context) binds it there.
    on_thread: ?ThreadInit = null,
    /// a family's round planner (prompts in pieces between decode rounds); null: whole prompts at admission
    rounds: ?Rounds = null,
    pieces: std.ArrayList(Rounds.Piece) = .empty,
    finals: std.ArrayList(usize) = .empty,

    pub const ThreadInit = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        /// the family's proposer (LaneHost.proposers), in place of `proposer`
        custom: ?lanes.proposer.Proposer = null,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        submitted: i96 = 0,
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| _ = store.keep(job.request.prompt, at, s, job.request.chunks);
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            if (job.stream.reuse_failed) store.resumed(e, job.request.prompt, false) else if (job.stream.cached == e.at) store.resumed(e, job.request.prompt, true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread (after `halt`: the thread is gone already).
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
        h.pieces.deinit(h.gpa);
        h.finals.deinit(h.gpa);
    }

    /// A drain's deadline: the round in flight completes on every rank, so the ranks stop at the same boundary.
    pub fn halt(h: *LaneHost, reason: []const u8) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.halt_reason = reason;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn } };
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink, .submitted = std.Io.Clock.awake.now(h.io).toNanoseconds() };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
            .oldest_queued_s = if (h.queued.items.len > 0) seconds(now - oldestQueued(h)) else 0,
            .pace = paceStatus(&h.pace, now),
        };
    }

    fn oldestQueued(h: *const LaneHost) i96 {
        var at = h.queued.items[0].submitted;
        for (h.queued.items) |job| at = @min(at, job.submitted);
        return at;
    }

    fn paceStatus(p: *const pace_mod.Pace, now: i96) api.PaceStatus {
        return .{
            .last_round_age_s = if (p.last_end) |e| seconds(now - e) else null,
            .last_round_s = @as(f64, @floatFromInt(p.last_ns)) / 1e9,
            .decode_s = if (p.decode.ready()) p.decode.quick else 0,
            .decode_base_s = if (p.decode.ready()) p.decode.base else 0,
            .prefill_rows_s = if (p.prefill.ready()) p.prefill.quick else 0,
            .prefill_base_rows_s = if (p.prefill.ready()) p.prefill.base else 0,
            .slow_s = p.slowFor(now),
        };
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, "");
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        if (h.rounds) |rd| rd.vtable.left(rd.ctx, @intFromPtr(job));
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            s.deinit(h.gpa);
            h.freeProposer(job);
        }
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                if (job.started and !job.stream.finished) h.keepLeaving(job);
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    /// A request whose client left mid-prompt keeps its finished prefix as a session entry: the same prompt resumes.
    fn keepLeaving(h: *LaneHost, job: *Job) void {
        const rd = h.rounds orelse return;
        const ask = rd.vtable.leaving orelse return;
        const at = ask(rd.ctx, @intFromPtr(job)) orelse return;
        h.core.piece(&job.stream, at, at, true) catch |e| std.log.warn("lane host: a leaving prompt's {d} rows were not kept ({t})", .{ at, e });
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        const job = h.queued.orderedRemove(0);
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        if (!h.openStream(job)) return true;
        const began = job.began;
        if (h.loneFits(job)) return h.runLone(job, began);
        h.core.addStream(&job.stream) catch |e| return if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    /// An admitted job's stream (its kept prefix looked up, its drafter): false when it failed (the job is finished).
    fn freeProposer(h: *LaneHost, job: *Job) void {
        if (job.custom) |c| h.proposers.?.free(h.proposers.?.ctx, c) else job.proposer.deinit();
        job.custom = null;
    }

    fn openStream(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (h.cache) |store| if (store.lookup(h.gpa, r.prompt, r.history_len, r.shared_prefixes, r.chunks)) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
        } else |_| {};
        if (h.proposers) |ps| {
            job.custom = ps.make(ps.ctx, h.gpa, r.prompt, r.eos) catch return !h.drop(job, "the drafter could not start");
        } else job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return !h.drop(job, "the drafter could not start");
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.custom orelse job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .chunks = r.chunks,
            .reuse = reuse,
        }) catch {
            h.freeProposer(job);
            return !h.drop(job, "out of memory");
        };
        job.started = true;
        job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        return true;
    }

    fn seconds(ns: i96) f64 {
        return @as(f64, @floatFromInt(ns)) / 1e9;
    }

    /// One round under the family's planner; false when it did nothing (nothing to admit, prefill or decode).
    fn round(h: *LaneHost, rd: Rounds) bool {
        var did = false;
        // the admission pass, decided whole before any of it runs
        var admits: std.ArrayList(*Job) = .empty;
        defer admits.deinit(h.gpa);
        var refused: std.ArrayList(struct { job: *Job, why: []const u8 }) = .empty;
        defer refused.deinit(h.gpa);
        h.lock();
        if (h.queued.items.len > 0 and h.admitted.items.len < h.info_.lanes) {
            rd.vtable.begin_admit(rd.ctx);
            var free = h.info_.lanes - h.admitted.items.len;
            var i: usize = 0;
            while (i < h.queued.items.len and free > 0) {
                const job = h.queued.items[i];
                var why: []const u8 = "";
                const v = rd.vtable.admit(rd.ctx, .{ .key = @intFromPtr(job), .prompt = job.request.prompt, .max_tokens = job.request.max_tokens, .submitted_s = seconds(job.submitted) }, &why);
                switch (v) {
                    .admit => {
                        if (h.admitted.append(h.gpa, job)) |_| {} else |_| break;
                        if (admits.append(h.gpa, job)) |_| {} else |_| {
                            _ = h.admitted.pop();
                            break;
                        }
                        _ = h.queued.orderedRemove(i);
                        free -= 1;
                    },
                    .wait => break,
                    .skip => i += 1,
                    .refuse => {
                        if (refused.append(h.gpa, .{ .job = job, .why = why })) |_| {} else |_| break;
                        _ = h.queued.orderedRemove(i);
                    },
                }
            }
        }
        h.unlock();
        for (refused.items) |x| h.finish(x.job, .failed, x.why);
        rd.vtable.admitted(rd.ctx) catch |e| {
            h.failAll(h.words(e));
            return true;
        };
        for (admits.items) |job| {
            did = true;
            if (!h.openStream(job)) continue;
            rd.vtable.arm(rd.ctx, @intFromPtr(job));
            const b = h.core.beginStream(&job.stream) catch |e| {
                _ = h.drop(job, h.words(e));
                continue;
            };
            rd.vtable.began(rd.ctx, @intFromPtr(job), b.damaged);
        }
        // this round's pieces, a decode round of the streams decoding, then the prompts whose rows are in
        rd.vtable.plan(rd.ctx, &h.pieces, &h.finals) catch |e| {
            h.failAll(h.words(e));
            return true;
        };
        const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (h.core.cfg.piece_runs and h.core.backend.vtable.pieces != null) {
            // one call for the round's pieces (the backend runs several slots' rows in one forward); a failure fails each
            var list: [16]lanes.backend.Backend.Piece = undefined;
            var jobs: [16]*Job = undefined;
            var n: usize = 0;
            for (h.pieces.items) |piece| {
                if (!piece.run) continue;
                did = true;
                const job: *Job = @ptrFromInt(piece.key);
                if (job.stream.finished) continue;
                if (n == list.len) {
                    _ = h.drop(job, "too many prompt pieces in one round");
                    continue;
                }
                list[n] = .{ .stream = &job.stream, .start = piece.start, .end = piece.end, .save = piece.save };
                jobs[n] = job;
                n += 1;
            }
            if (n > 0) h.core.pieces(list[0..n]) catch |e| {
                for (jobs[0..n]) |job| h.dropPiece(job, h.words(e));
            };
        } else for (h.pieces.items) |piece| {
            if (!piece.run) continue;
            did = true;
            const job: *Job = @ptrFromInt(piece.key);
            if (job.stream.finished) continue;
            h.core.piece(&job.stream, piece.start, piece.end, piece.save) catch |e| {
                h.dropPiece(job, h.words(e));
            };
        }
        const t1 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        // a joined prompt's last token decodes in this round's window (Config.join_tail; Python's batcher: `finals`
        // before `_decode`), so its stream opens first; otherwise its own window runs after the round
        const join = h.core.cfg.join_tail;
        if (join and h.finals.items.len > 0) {
            did = true;
            h.runFinals();
        }
        var step_ns: i96 = 0;
        if (h.core.activeCount() > 0) {
            did = true;
            const s0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                return true;
            };
            step_ns = std.Io.Clock.awake.now(h.io).toNanoseconds() - s0;
        }
        if (!join and h.finals.items.len > 0) {
            did = true;
            h.runFinals();
        }
        const t2 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var i: usize = 0;
        while (i < h.admitted.items.len) {
            const job = h.admitted.items[i];
            if (job.started and h.deliver(job)) {
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
            } else i += 1;
        }
        rd.vtable.after(rd.ctx, seconds(t1 - t0), seconds(t2 - t1));
        if (did) {
            var rows: u64 = 0;
            for (h.pieces.items) |piece| if (piece.run) {
                rows += piece.end - piece.start;
            };
            h.lock();
            h.pace.round(t2, rows, seconds(t1 - t0), seconds(step_ns));
            h.unlock();
        }
        return did;
    }

    /// A job whose piece failed: dropped, and out of this round's finals (the planner may have listed its prompt as
    /// finished in the same round; `drop` frees the job, so runFinals must not see its key).
    fn dropPiece(h: *LaneHost, job: *Job, message: []const u8) void {
        const key = @intFromPtr(job);
        var i: usize = 0;
        while (i < h.finals.items.len) {
            if (h.finals.items[i] == key) _ = h.finals.orderedRemove(i) else i += 1;
        }
        _ = h.drop(job, message);
    }

    /// The prompts whose rows are in: each stream's last row (Engine.finishStream), then it decodes with the rounds.
    fn runFinals(h: *LaneHost) void {
        if (h.core.cfg.replay_runs) {
            // the finals' shared work first (the DSV4.1 target: their decoder replays in one forward); a failure fails each
            var ss: [16]*lanes.Stream = undefined;
            var jobs: [16]*Job = undefined;
            var n: usize = 0;
            for (h.finals.items) |key| {
                const job: *Job = @ptrFromInt(key);
                if (job.stream.finished or n == ss.len) continue;
                ss[n] = &job.stream;
                jobs[n] = job;
                n += 1;
            }
            h.core.prepareFinals(ss[0..n]) catch |e| {
                for (jobs[0..n]) |job| _ = h.drop(job, h.words(e));
                return;
            };
        }
        if (h.core.cfg.join_tail and h.core.cfg.join_drafts) {
            // the round's finals together: the joined prompts' first drafts in one pass (Engine.finishStreams)
            var ss: [16]*lanes.Stream = undefined;
            var jobs: [16]*Job = undefined;
            var errs: [16]?anyerror = undefined;
            var at: usize = 0;
            while (at < h.finals.items.len) {
                var n: usize = 0;
                while (at < h.finals.items.len and n < ss.len) : (at += 1) {
                    const job: *Job = @ptrFromInt(h.finals.items[at]);
                    if (job.stream.finished) continue;
                    ss[n] = &job.stream;
                    jobs[n] = job;
                    n += 1;
                }
                const joined = h.core.finishStreams(ss[0..n], errs[0..n]) catch |e| blk: {
                    for (errs[0..n]) |*x| x.* = e;
                    break :blk 0;
                };
                // TF_DSV41_JOIN_DRAFTS: a round's line when its finished prompts joined (their first drafts in one pass)
                if (joined > 0) std.log.info("join drafts: {d} of {d} finished prompts drafted in one pass", .{ joined, n });
                for (jobs[0..n], errs[0..n]) |job, x| {
                    if (x) |e| {
                        if (e == error.Cancelled) {
                            h.core.discard(&job.stream);
                            h.remove(job);
                            h.finish(job, .cancelled, "");
                        } else _ = h.drop(job, h.words(e));
                        continue;
                    }
                    h.prefilled(job, job.began);
                }
            }
            return;
        }
        for (h.finals.items) |key| {
            const job: *Job = @ptrFromInt(key);
            if (job.stream.finished) continue;
            h.core.finishStream(&job.stream) catch |e| {
                if (e == error.Cancelled) {
                    h.core.discard(&job.stream);
                    h.remove(job);
                    h.finish(job, .cancelled, "");
                } else _ = h.drop(job, h.words(e));
                continue;
            };
            h.prefilled(job, job.began);
        }
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        if (h.cache) |store| store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        h.lock();
        defer h.unlock();
        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
    }

    fn run(h: *LaneHost) void {
        if (h.on_thread) |t| t.run(t.ctx) catch |e| std.log.err("lane host: the engine thread's init failed ({t})", .{e});
        if (h.rounds) |rd| return h.runRounds(rd);
        while (true) {
            h.takeCancels();
            while (h.admitOne()) {}
            h.noteLive();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            const s0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            const s1 = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.lock();
            h.pace.round(s1, 0, 0, seconds(s1 - s0));
            h.unlock();
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// The engine thread under a round planner: rounds while requests wait or run; an idle round (requests held back
    /// by the planner, nothing running) waits for an arrival, a cancel or a short time before asking again.
    fn runRounds(h: *LaneHost, rd: Rounds) void {
        while (true) {
            h.takeCancels();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            const work = h.admitted.items.len > 0 or (h.queued.items.len > 0 and h.admitted.items.len < h.info_.lanes);
            if (!work) {
                if (h.cancels.items.len == 0) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            const did = h.round(rd);
            h.noteLive();
            if (!did) {
                h.lock();
                if (h.cancels.items.len == 0) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(20), .clock = .awake } }) catch {};
                h.unlock();
            }
        }
    }

    /// `halt`: between rounds, every request queued or admitted finishes failed with `reason` (its stream discarded).
    fn haltAll(h: *LaneHost, reason: []const u8) void {
        h.lock();
        const queued = h.gpa.dupe(*Job, h.queued.items) catch &.{};
        h.queued.clearRetainingCapacity();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        if (queued.len + jobs.len > 0) std.log.warn("lane host: stopping at a round boundary: {d} running and {d} queued request(s) end ({s})", .{ jobs.len, queued.len, reason });
        for (jobs) |job| {
            if (job.started and !job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, reason);
        }
        for (queued) |job| h.finish(job, .failed, reason);
        h.gpa.free(queued);
        h.gpa.free(jobs);
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        std.log.err("lane host: a round failed ({s}): every admitted stream fails", .{message});
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        if (h.halt_reason) |reason| return h.haltAll(reason);
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
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
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_host_resume_test.zig");
    _ = @import("pace.zig");
}
