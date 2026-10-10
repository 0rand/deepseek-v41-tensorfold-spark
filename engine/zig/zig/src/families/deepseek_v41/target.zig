//! M3's GPU target: draft/iface.zig's `Target` over forward.zig, so the lanes backend (draft/lanes.zig) drives the real
//! model as it drives the CPU twin. This first form is one slot, chain windows (a draft chain under the pending token),
//! greedy choices and the prefill as consecutive decode windows; trees (`parents`) run as chain windows on the slot
//! (draft/branches.zig: the main chain, then a winning sibling's path in place of it); several slots are refused by
//! name until state.zig (per-slot pools) lands. Sampling (`lanes.Sampling`, T > 0, keyed at `draws`) goes through
//! `sampler` (sampling_gpu.zig: Python's candidates, nucleus statistics and keyed choice); without one it is refused.
//! M5 (`sessions`, with the paged pool): a request is admitted by the pool pages it may write (`admit`: its reply budget;
//! prod's batch.py `_admit`), then its prompt resumes the longest saved entry that strictly prefixes it (RAM or NVMe)
//! and prefills only the rest; a released slot is saved as an entry (its pages shared, its bounded state kept), the
//! residency budget parking or dropping the least recently used chats. With several slots (`rows`) the same runs on
//! the activated slot while the other slots' windows wait, as prod's round runs its admissions before its windows.
//! Prompts in pieces (`begin` / `piece` / `finish`, the served shape only: own prefill, the verify tail; TF_DSV41_
//! PREFILL_PIECES, serve_engine.zig's rounds over kv/sched.zig): the request admitted as the round planner decided
//! (`admission`: its pages, the entry it resumes, the round's spills), each piece of rows run between other slots'
//! decode rounds (its snapshot right after it at the replay point; full mode: its taps to the drafter, decode.py
//! `prefill`), then the decoder replay and the prompt's last token's window. A released slot's pages wait for the
//! next round's admissions to be decided (`defer_release`, `settleReleases`: prod's `_finish` at the next round's
//! start, after its plan).

const std = @import("std");
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const sg = @import("sessions_gpu.zig");
const batch = @import("batch.zig");
const samp = @import("sampling_gpu.zig");
const branches = @import("draft/branches.zig");
const pk = @import("prod_knobs.zig");
const ced = @import("ced.zig");
const pf = @import("forward_prefill.zig");

pub const Error = error{ TreeWindow, Sampling, Slots, NotAChain, WindowRows };

pub const GpuTarget = struct {
    f: *fwd.Forward,
    /// rows a prefill window takes (the decode path's buckets; the forward plans buffers for them)
    prefill_rows: u32 = 16,
    /// a slot that already holds the prompt (a gate resuming the reference's state): prefill returns this choice
    /// without running the prompt again
    resume_first: ?u32 = null,
    /// the most rows a window may take: the forward's buffers are planned for its buckets (run.zig has no bounds)
    max_rows: u32 = 16,
    /// the prompt through the forward's own prefill (2,048-row segments, Python's chunks; forward.planPrefill sized
    /// its buffers) instead of `prefill_rows`-row decode windows
    own_prefill: bool = false,
    /// with `own_prefill`: where the prompt's last token runs (prod_knobs.PromptTail; model.zig reads
    /// TF_DSV41_PROMPT_TAIL). `.verify` is Python's server: the reply's first decode window, left pending as a chain
    /// window is. `.prefill`: the last prefill segment's last row, as Python's Forward.prompt (the gates' references)
    prompt_tail: pk.PromptTail = .prefill,
    /// M5: the session store over the paged pool (null: every prompt from an empty slot, nothing kept)
    sessions: ?*sg.Sessions = null,
    /// several live slots (TF_DSV41_SLOTS > 1, batch.zig): windows in row mode over every slot's segment, the prompt on
    /// the activated slot (greedy chains; trees and sampling refused there by name)
    rows: ?*batch.Batch = null,
    /// the keyed sampler (model.zig makes it on every rank): sampled choices; null refuses them (error.Sampling)
    sampler: ?*samp.GpuSampler = null,
    /// draft trees (`parents`): resolved as chain windows on the slot (draft/branches.zig), the last one kept by path
    tree: branches.Resolver = .{},
    /// the DSpark pass the lanes drive (model.zig): with `.verify`, the prompt's prefilled tail's taps go to it before
    /// the reply's first verify window overwrites them (Python: the decoder replay's `replayed`, the last piece's keep);
    /// `ingest_window`: the drafter's context rows (DSpark's shape window)
    pass: ?iface.Pass = null,
    ingest_window: u32 = 0,
    /// each slot's next request's reply budget (`admit`, before its prefill): its pool admission's price
    max_new: [16]?u64 = @splat(null),
    /// the next `begin`'s admission as the round planner decided it (kv/sched.zig); null: priced at `begin`
    admission: ?Planned = null,
    /// releases wait for `settleReleases` (the rounds: after the next round's admissions are decided)
    defer_release: bool = false,
    /// TF_DSV41_ENGRAM_WARM=1 (`warm`): Python's commit and draft-end Engram warms (every rank alike)
    engram_warm: bool = false,
    releasing: [16]bool = @splat(false),
    /// TF_DSV41_PIECE_RUNS: rounds whose pieces ran as one multi-segment run
    piece_runs: u64 = 0,
    /// TF_DSV41_REPLAY_RUNS: slots whose decoder replay a batched run did (their `finish` / `tail` skips it), and the
    /// runs so far
    replayed: [16]bool = @splat(false),
    replay_runs: u64 = 0,

    /// A request's admission (kv/sched.zig Admission + the round's spills, with its first admission).
    pub const Planned = struct { need: u32, hit: ?@import("kv").sessions.store.Hit, spills: []const u32 = &.{} };

    pub fn target(t: *GpuTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .admit = admit, .begin = begin, .piece = piece, .finish = finish, .tail = tail, .pieces = pieces, .replays = replays, .warm = warm };

    fn admit(p: *anyopaque, slot: u32, max_new: u64) void {
        const t = self_(p);
        if (slot < t.max_new.len) t.max_new[slot] = max_new;
    }

    fn self_(p: *anyopaque) *GpuTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self_(p);
        if (t.rows) |b| try b.beginPrefill(slot) else if (slot != 0) return error.Slots;
        _ = t.takeReplayed(slot);
        const keyed = try t.keyedOf(sampling);
        t.tree.clear();
        if (t.resume_first) |first| if (t.f.slot.pos == ids.len and t.f.slot.pending == null) return first;
        // the request's pool pages (its reply budget from `admit`), then a saved entry that prefixes the prompt: the
        // slot continues at its position, the rest is prefilled
        const max_new: ?u64 = if (slot < t.max_new.len) t.max_new[slot] else null;
        if (slot < t.max_new.len) t.max_new[slot] = null;
        // a failed prompt (a damaged NVMe blob, a pool that ran out) leaves the slot empty on every rank, its pages and
        // reservation back (prod: the request fails, its slot finishes), not holding them for the next request
        errdefer if (t.sessions != null) t.abandon(slot);
        const at: usize = if (t.sessions) |ss| @intCast(try ss.resume_(ids, max_new)) else 0;
        const rest = ids[at..];
        if (t.own_prefill and t.prompt_tail == .verify) {
            // Python's server (batch.py `_pieces` / `finals`): the rows before the last token are prefilled, and the
            // last token is the pending row of the reply's first verify window: this target's own decode window
            // (one slot or row mode), its choice greedy or keyed at `draw`, left pending for the next window to keep
            // (a chain window never kept keeps every row; release keeps it before a session save)
            const sp = pk.promptSplit(rest.len, .verify);
            t.f.prefill_state.taps_rows = 0; // a one-token prompt prefills nothing: no stale taps handed over
            // sessions: the prompt's snapshot at its replay point (batch.py `save_at`, rounds.py `_save(slot,
            // "prompt")`): its rows prefilled, an entry (CED: with the stash), then the rest
            const save_at: ?u64 = if (t.sessions != null) ced.savePoint(ids.len, at) else null;
            if (save_at) |s_at| {
                const cut: usize = @intCast(s_at);
                try t.f.prefill(ids[at..cut]);
                // full mode: the piece's last rows to the drafter first (decode.py `prefill`), so the snapshot's
                // drafter rings hold them (drafter.py `snapshot`)
                if (t.f.prefill_state.mode == .full) try t.handTaps(slot);
                _ = try t.sessions.?.savePrompt();
                try t.f.prefill(ids[cut .. at + sp.prefill]);
            } else try t.f.prefill(rest[0..sp.prefill]);
            // the decoder replay over the prompt's tail (rounds.py `finish_prompt`; full mode: nothing)
            try t.f.finishPrompt();
            try t.handTaps(slot);
            var one: [1]u32 = undefined;
            var out = [_][]u32{&one};
            try window(p, &.{.{ .slot = slot, .start = ids.len - sp.window, .tokens = ids[ids.len - sp.window ..], .parents = null, .draws = &.{draw}, .sampling = sampling }}, &out);
            return one[0];
        }
        if (t.own_prefill) {
            // the keyed choice after the prompt: the last segment's last row
            try t.f.prefill(rest);
            const seg: usize = @import("forward_prefill.zig").segmentRows(t.f);
            const last = if (rest.len % seg == 0) seg else rest.len % seg;
            if (keyed) |s| return t.pickOne(@intCast(last - 1), s, draw);
            const picks = try t.f.gpa.alloc(u32, last);
            defer t.f.gpa.free(picks);
            try t.f.greedy(picks);
            return picks[last - 1];
        }
        var at_ = at;
        var choice: [1]u32 = .{0};
        while (at_ < ids.len) {
            const n = @min(@min(t.prefill_rows, t.max_rows), ids.len - at_);
            try t.f.window(ids[at_ .. at_ + n]);
            if (at_ + n == ids.len) {
                // the keyed choice after the prompt's last row: the last row's greedy pick, or its sampled one
                if (keyed) |s| choice[0] = try t.pickOne(@intCast(n - 1), s, draw) else {
                    var rows: [64]u32 = undefined;
                    try t.f.greedy(rows[0..n]);
                    choice[0] = rows[n - 1];
                }
            }
            try t.f.keep(@intCast(n - 1));
            at_ += n;
        }
        return choice[0];
    }

    /// The pieces path's slot for an operation: activated (row mode), or the one slot.
    fn onSlot(t: *GpuTarget, slot: u32) !void {
        if (t.rows) |b| try b.beginPrefill(slot) else if (slot != 0) return error.Slots;
    }

    fn begin(p: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const t = self_(p);
        if (!(t.own_prefill and t.prompt_tail == .verify)) return error.NoPieces;
        _ = t.takeReplayed(slot); // a batched replay of a request that never finished is not this one's
        const planned = t.admission;
        t.admission = null;
        try t.onSlot(slot);
        t.tree.clear();
        const max_new: ?u64 = if (slot < t.max_new.len) t.max_new[slot] else null;
        if (slot < t.max_new.len) t.max_new[slot] = null;
        errdefer if (t.sessions != null) t.abandon(slot);
        t.f.prefill_state.taps_rows = 0;
        const ss = t.sessions orelse return .{};
        if (planned) |a| {
            const r = try ss.resumePlanned(ids.len, a.need, a.hit, a.spills);
            return .{ .at = r.at, .damaged = r.damaged };
        }
        const before = ss.stats.damaged;
        const at = try ss.resume_(ids, max_new);
        return .{ .at = at, .damaged = ss.stats.damaged != before };
    }

    fn piece(p: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void {
        const t = self_(p);
        try t.onSlot(slot);
        if (start == end and save) {
            // a leaving prompt's finished rows kept as they stand (the lane host: its client left mid-prompt)
            if (t.f.slot.pos != start or start < ced.session_min or end + 1 > ids.len) return error.NotAChain;
            if (t.sessions) |ss| _ = try ss.savePrompt();
            return;
        }
        if (t.f.slot.pos != start or start >= end or end + 1 > ids.len) return error.NotAChain;
        // the next round's piece of this prompt read ahead behind this one (Python's rounds._prefetch(ahead=
        // prefetch.next_pieces): the rows after it, as many as it holds, up to the prompt's last token)
        const last = ids.len - 1;
        if (end < last) try t.f.prefillAhead(ids[end..@min(last, end + (end - start))]);
        try t.f.prefill(ids[start..end]);
        // rounds.py `_save`: a history under session_min is not snapshotted (a damaged entry's restart cuts there)
        if (save and end >= ced.session_min) if (t.sessions) |ss| {
            _ = try ss.savePrompt();
        };
        // full mode: the drafter reads each piece's last rows (decode.py `prefill`'s ingest of a piece's tail)
        if (t.f.prefill_state.mode == .full) try t.handTaps(slot);
    }

    fn finish(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self_(p);
        try t.onSlot(slot);
        if (ids.len == 0 or t.f.slot.pos != ids.len - 1) return error.NotAChain;
        if (t.f.prefill_state.mode == .replay and !t.takeReplayed(slot)) {
            // the decoder replay over the prompt's tail (rounds.py `finish_prompt`), its taps the drafter's context
            t.f.prefill_state.taps_rows = 0;
            try t.f.finishPrompt();
            try t.handTaps(slot);
        }
        var one: [1]u32 = undefined;
        var out = [_][]u32{&one};
        try window(p, &.{.{ .slot = slot, .start = ids.len - 1, .tokens = ids[ids.len - 1 ..], .parents = null, .draws = &.{draw}, .sampling = sampling }}, &out);
        return one[0];
    }

    /// A round's pieces of several slots (TF_DSV41_PIECE_RUNS, lanes' Config.piece_runs): CED replay's encoder passes
    /// of all of them in one run (Forward.prefillMulti: Python's prefill over prefill_runs), then each piece's prompt
    /// snapshot, as `piece` a piece; the run's refusals (a short or image segment, more rows than a prefill segment, split
    /// KV) and full mode (its drafter taps a piece) run a piece at a time.
    fn pieces(p: *anyopaque, list: []const iface.Target.Piece) anyerror!void {
        const t = self_(p);
        if (list.len > 1 and list.len <= 16 and t.f.prefill_state.mode == .replay and t.f.pf_switch != null) {
            var segs: [16]pf.MultiSeg = undefined;
            for (list, segs[0..list.len]) |q, *seg| {
                if (q.start >= q.end or q.end + 1 > q.ids.len) return error.NotAChain;
                seg.* = .{ .slot = q.slot, .start = q.start, .ids = q.ids[q.start..q.end] };
            }
            if (try pf.runnable(t.f, segs[0..list.len])) {
                try t.onSlot(list[0].slot); // the slots' pending windows kept (every rank), as each `piece` does
                try t.f.prefillMulti(segs[0..list.len]);
                t.piece_runs += 1;
                if (t.piece_runs == 1 or t.piece_runs % 256 == 0) std.log.scoped(.dsv41).info("piece runs: {d} (this one {d} segments in one run)", .{ t.piece_runs, list.len });
                // rounds.py `_save` after the round's prefill, a piece at a time
                for (list) |q| if (q.save and q.end >= ced.session_min) if (t.sessions) |ss| {
                    try t.onSlot(q.slot);
                    _ = try ss.savePrompt();
                };
                return;
            }
        }
        for (list) |q| try piece(p, q.slot, q.ids, q.start, q.end, q.save);
    }

    /// Iface.Target.warm: row mode warms on every rank (Batch.warmEngram, op 56); one slot, this rank's reads of the
    /// one slot (the followers have R3's row-0 read at op_ds_propose). Off unless TF_DSV41_ENGRAM_WARM=1.
    fn warm(p: *anyopaque, asks: []const iface.Warm) void {
        const t = self_(p);
        if (!t.engram_warm or asks.len == 0) return;
        if (t.rows) |b| {
            b.warmEngram(asks) catch |e| std.log.scoped(.dsv41).warn("engram warm: {t} (the windows read their own rows)", .{e});
            return;
        }
        for (asks) |a| if (a.slot == 0) t.f.prefetchEngram(a.start, a.ids);
    }

    /// TF_DSV41_ENGRAM_WARM: 0 (default) | 1.
    pub fn engramWarmEnv() !bool {
        const v = std.mem.trim(u8, std.mem.span(std.c.getenv("TF_DSV41_ENGRAM_WARM") orelse return false), " ");
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
        if (std.mem.eql(u8, v, "1")) return true;
        return error.BadEngramWarm;
    }

    /// `finish` up to the last token (TF_DSV41_TAIL_JOIN): the decoder replay and its taps; the slot stays at the last
    /// token's position with no window pending, which the next round's window (one slot or row mode) starts at.
    fn tail(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 {
        _ = draw;
        const t = self_(p);
        try t.onSlot(slot);
        if (ids.len == 0 or t.f.slot.pos != ids.len - 1) return error.NotAChain;
        _ = try t.keyedOf(sampling); // refused here as `finish` would, not in the round
        if (t.f.prefill_state.mode == .replay and !t.takeReplayed(slot)) {
            t.f.prefill_state.taps_rows = 0;
            try t.f.finishPrompt();
            try t.handTaps(slot);
        }
        return null;
    }

    /// The releases held since the last round (`defer_release`) run now, slot by slot.
    pub fn settleReleases(t: *GpuTarget) void {
        for (&t.releasing, 0..) |*x, s| if (x.*) {
            x.* = false;
            t.releaseNow(@intCast(s));
        };
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.targetWindow(segments, choices, t.sampler);
        if (segments.len != 1) return error.Slots;
        const s = segments[0];
        if (s.slot != 0) return error.Slots;
        if (s.parents != null) return t.tree.window(.{ .ptr = t, .vtable = &chains }, s, choices[0]);
        t.tree.clear();
        const keyed = try t.keyedOf(s.sampling);
        // a chain window never kept keeps every row (iface.Target.window): lanes' undrafted step (queue) sends the
        // next row without a keep
        if (t.f.slot.pending) |w| {
            if (s.start != w.start + w.n) return error.NotAChain;
            try t.f.keep(w.n - 1);
        }
        if (s.start != t.f.slot.pos) return error.NotAChain;
        if (s.tokens.len > t.max_rows) return error.WindowRows;
        try t.f.window(s.tokens);
        if (keyed) |smp| return t.sampler.?.choose(0, @intCast(s.tokens.len), smp, s.draws, choices[0][0..s.tokens.len]);
        try t.f.greedy(choices[0][0..s.tokens.len]);
    }

    /// The slot's pending window (a chain window never kept), if any.
    pub fn hasPending(t: *const GpuTarget, slot: u32) bool {
        if (t.rows) |b| return slot < b.ss.n and b.pending(slot) != null;
        return slot == 0 and t.f.slot.pending != null;
    }

    /// The slot's pending window dropped on every rank, nothing committed (TF_DSV41_CALIB=measure, calib_gpu.zig:
    /// Python times `fw.window` without a commit). Row mode: op_rows_drop; one slot: op 20 (Forward.drop).
    pub fn dropWindow(t: *GpuTarget, slot: u32) !void {
        if (t.rows) |b| return b.drop(slot);
        if (slot != 0) return error.Slots;
        t.tree.clear();
        if (t.f.slot.pending != null) try t.f.drop();
    }

    /// The prompt's prefilled tail's taps into the DSpark pass (ced.handoff): its last `ingest_window` rows, one ingest.
    fn handTaps(t: *GpuTarget, slot: u32) !void {
        const st = &t.f.prefill_state;
        return t.handTapsAt(slot, st.taps_at, st.taps_rows, 0);
    }

    /// The drafter's context from "w.taps": positions [at, at + rows) at rows [base, base + rows) (a batched replay's
    /// slot at its rows of the run), the last `ingest_window` of them (ced.handoff).
    fn handTapsAt(t: *GpuTarget, slot: u32, at: u64, n: u32, base: u32) !void {
        const ps = t.pass orelse return;
        const h = ced.handoff(at, n, t.ingest_window) orelse return;
        var rows: [256]u32 = undefined;
        if (h.n > rows.len) return error.WindowRows;
        for (rows[0..h.n], 0..) |*r, i| r.* = base + h.row0 + @as(u32, @intCast(i));
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        const tp: iface.Taps = .{ .address = t.f.runner.addressOf("w.taps") orelse return error.Unbound, .rows = base + n, .stride = k * t.f.cfg.hidden };
        try ps.ingest(slot, h.start, tp, rows[0..h.n]);
    }

    /// A batched run replayed `slot`'s tail (TF_DSV41_REPLAY_RUNS): true once, and the flag cleared.
    fn takeReplayed(t: *GpuTarget, slot: u32) bool {
        if (slot >= t.replayed.len or !t.replayed[slot]) return false;
        t.replayed[slot] = false;
        return true;
    }

    /// TF_DSV41_REPLAY_RUNS: the round's finished prompts' CED decoder replays in one run (Forward.replayMulti), before
    /// their `finish` / `tail` (which then skip theirs), each slot's taps to the drafter as `finish` hands them, in the
    /// finals' order. A tail of 32 rows or fewer (a short prompt) and the run's other refusals replay alone there.
    fn replays(p: *anyopaque, list: []const iface.Target.Final) anyerror!void {
        const t = self_(p);
        if (list.len < 2 or t.f.prefill_state.mode != .replay or t.f.pf_switch == null) return;
        var segs: [16]pf.MultiSeg = undefined;
        var n: usize = 0;
        for (list) |q| {
            if (q.ids.len < 2 or n == segs.len) continue;
            const end: u64 = q.ids.len - 1;
            const tl = ced.tailOf(end);
            if (tl.m <= 32) continue;
            segs[n] = .{ .slot = q.slot, .start = tl.r0, .ids = q.ids[tl.r0..end] };
            n += 1;
        }
        if (n < 2 or !try pf.replayRunnable(t.f, segs[0..n])) return;
        try t.onSlot(segs[0].slot); // the slots' pending windows kept (every rank), as each `finish` does
        var slots: [16]u32 = undefined;
        for (segs[0..n], slots[0..n]) |seg, *x| x.* = seg.slot;
        var out: [16]pf.Replayed = undefined;
        try t.f.replayMulti(slots[0..n], &out);
        for (out[0..n]) |o| {
            t.replayed[o.slot] = true;
            try t.handTapsAt(o.slot, o.at, o.rows, o.row0);
        }
        t.replay_runs += 1;
        if (t.replay_runs == 1 or t.replay_runs % 256 == 0) std.log.scoped(.dsv41).info("replay runs: {d} (this one {d} slots' tails in one run)", .{ t.replay_runs, n });
    }

    /// T > 0: the sampling the sampler keys (null: greedy); refused without a sampler.
    fn keyedOf(t: *const GpuTarget, sampling: ?lanes.Sampling) !?lanes.Sampling {
        if (!samp.sampled(sampling)) return null;
        if (t.sampler == null) return error.Sampling;
        return sampling;
    }

    /// The keyed choice of "w.logits" row `row`, drawn at `draw`.
    fn pickOne(t: *GpuTarget, row: u32, s: lanes.Sampling, draw: u64) !u32 {
        var one: [1]u32 = undefined;
        try t.sampler.?.choose(row, 1, s, &.{draw}, &one);
        return one[0];
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.keepPath(slot, path);
        if (slot != 0) return error.Slots;
        if (t.tree.live) return t.f.keep(try t.tree.keep(path));
        // a chain keeps rows 0..k: the forward commits row 0 + k accepted drafts
        for (path, 0..) |row, i| if (row != i) return error.NotAChain;
        if (path.len == 0) return error.NotAChain;
        try t.f.keep(@intCast(path.len - 1));
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const t = self_(p);
        if (t.rows) |b| if (b.tapsOf(slot)) |x| return .{ .address = x.address, .rows = x.rows, .stride = @intCast(t.f.cfg.dspark_targets.items().len * t.f.cfg.hidden) };
        const n: u32 = if (t.f.slot.pending) |w| w.n else @intCast(t.f.ids.len);
        const D: u32 = t.f.cfg.hidden;
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        return .{ .address = t.f.runner.addressOf("w.taps") orelse 0, .rows = n, .stride = k * D, .map = t.tree.map() };
    }

    /// The tree resolver's chains: a chain window through `window` (its rules and refusals), dropped on every rank.
    const chains: branches.Chains.VTable = .{ .run = chainRun, .drop = chainDrop };

    fn chainRun(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, sampling: ?lanes.Sampling, out: []u32) anyerror!void {
        var one = [_][]u32{out};
        return window(p, &.{.{ .slot = slot, .start = start, .tokens = tokens, .parents = null, .draws = draws, .sampling = sampling }}, &one);
    }

    fn chainDrop(p: *anyopaque, slot: u32) anyerror!void {
        if (slot != 0) return error.Slots;
        return self_(p).f.drop();
    }

    /// The slot emptied on every rank without a session save (a prompt that failed part way).
    fn abandon(t: *GpuTarget, slot: u32) void {
        t.tree.clear();
        const r = if (t.rows) |b| b.release(slot) else t.f.release();
        r catch |e| std.log.err("dsv41 target: emptying slot {d} after a failed prompt failed ({t})", .{ slot, e });
    }

    fn release(p: *anyopaque, slot: u32) void {
        const t = self_(p);
        if (t.defer_release and slot < t.releasing.len) {
            t.releasing[slot] = true;
            return;
        }
        t.releaseNow(slot);
    }

    fn releaseNow(t: *GpuTarget, slot: u32) void {
        if (t.rows) |b| {
            // outside the served shape the turn is kept for the next one, as the one-slot path below (prod's exact
            // prefill tag in full mode, rounds.py `_finish`); the served shape saved its prompt at its replay point
            if (t.sessions) |ss| if (!(t.own_prefill and t.prompt_tail == .verify)) {
                b.saveTurn(slot, ss) catch |e| {
                    std.log.err("sessions: save of slot {d} failed ({t}); the slot is released", .{ slot, e });
                    b.release(slot) catch |e2| std.log.err("dsv41 target: releasing slot {d} failed ({t})", .{ slot, e2 });
                };
                return;
            };
            return b.release(slot) catch |e| std.log.err("dsv41 target: releasing slot {d} failed ({t})", .{ slot, e });
        }
        if (slot != 0) return;
        t.tree.clear();
        if (t.sessions) |ss| {
            // the served shape keeps no turn snapshots (rounds.py `_finish`: only the exact prefill tag in full mode
            // does; Zig's prefill is the fast path, and CED replays a turn's decoder rows differently): the prompt
            // was saved at its replay point
            if (t.own_prefill and t.prompt_tail == .verify) {
                t.f.release() catch |e| std.log.err("dsv41 target: releasing the slot on the other ranks failed ({t})", .{e});
                return;
            }
            // the turn kept for the next one (a pending window is the reply's last token: committed first)
            if (t.f.slot.pending) |w| t.f.keep(w.n - 1) catch |e| std.log.err("sessions: keep before save failed ({t})", .{e});
            _ = ss.save() catch |e| {
                std.log.err("sessions: save failed ({t}); the slot is released", .{e});
                t.f.release() catch |e2| std.log.err("dsv41 target: releasing the slot on the other ranks failed ({t})", .{e2});
            };
            return;
        }
        t.f.release() catch |e| std.log.err("dsv41 target: releasing the slot on the other ranks failed ({t})", .{e});
    }
};
