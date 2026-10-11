//! Batched decode windows over several slots (Python rowgraphs.py / forward._run in row mode, G8): one forward over
//! every slot's window, its rows padded to a bucket (rowtab.zig), the CSA2 steps in row mode (rowmode.zig) over the
//! stacked state (slots.zig). Eager and graphed run the same program: the key's (bucket rows, context bucket), emitted
//! with its end at the context bucket's end (the indexer scores the bucket's keys; a row's keys past its own end score
//! -inf and never reach its attention), so graphs on == off and batched == alone, row for row.
//!
//! A window: every slot with a pending window kept whole first (a chain window never kept keeps every row), each
//! slot's pages mapped, the row table and Engram rows staged (one copy each, pinned halves), the program run (or its
//! graph replayed). Each slot's window stays pending until `keep(slot, accepted)`: the ratio-2 carry from its accepted
//! row of the window's compressor projection into the slot's carry row, its Engram tail, its position. The leader sends
//! every operation on the plan link first (op_rows, op_rows_keep; slots.zig op_select), so the followers run the same
//! launches and collectives in the same order.
//!
//! Knobs: TF_DSV41_SLOTS (slots.zig), TF_DSV41_ROWS_CAP (16: rows a forward; block.emit's decode path is gated up to
//! 16 rows, mhc_cuda's MAXR, so a larger mix runs as several forwards, rows being independent), TF_DSV41_GRAPHS and
//! graphs.Settings' (the graph cache keyed by (rows, context bucket), both ranks agreeing).

const std = @import("std");
const ph = @import("phases.zig");
const cuda = @import("cuda");
const kv = @import("kv");
const calls = @import("calls.zig");
const block = @import("block.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const egate = @import("engram_gate.zig");
const graphs = @import("graphs.zig");
const round_graph = @import("round_graph.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const slots_mod = @import("slots.zig");
const iface = @import("draft/iface.zig");
const buffers = @import("buffers.zig");
const samp = @import("sampling_gpu.zig");
const sessions_gpu = @import("sessions_gpu.zig");

pub const Error = error{ BadWindow, NotAChain, RowsCap, BadPlan, TreeWindow, Sampling, Slots };

/// With taps (drafts): each slot's taps rows of a round's earlier forwards, [S, seg_max, taps width] bf16 (an external
/// persistent role: dspark_slots.zig's ingest reads a stashed slot's rows here).
pub const taps_stash_role = "s.rows.taps";

/// TF_DSV41_ROWS_CAP: rows one forward takes (a bucket; default 16).
pub fn capFromEnv() !u32 {
    const v = std.c.getenv("TF_DSV41_ROWS_CAP") orelse return 16;
    const n = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (rowtab.bucketRows(n, 64) != n) return error.RowsCap;
    return n;
}

/// A slot's rows in the last row window; `stashed`: its ratio-2 projection rows were copied aside (a later forward of
/// the same round overwrote the window's), so keep reads them there, and its taps are gone.
const Pend = struct { a: u32, start: u64, n: u32, ids: [rowtab.seg_max]u32 = undefined, stashed: bool = false };

pub const Batch = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    ss: *slots_mod.SlotSet,
    cap: u32,
    settings: graphs.Settings,
    tab: Table,
    /// each ratio-2 kv source's projection rows of pending slots, [S, seg_max, 2D] bf16 (a round split over forwards)
    stash: std.ArrayList(Stash) = .empty,
    /// with taps: the pending windows' taps rows of a round split over forwards (taps_stash_role), null without
    taps_stash: ?cuda.DeviceBuffer = null,
    stages: std.ArrayList(graphs.EngramStage) = .empty,
    /// each key's row program (emitted once, the forward's arena)
    programs: std.AutoHashMapUnmanaged(graphs.Key, []const calls.Call) = .empty,
    /// each key's program's last "w.taps" call (run.lastTaps, once a key: a scan of every argument of ~1,300 calls at
    /// 24 rows, which ran before every window's launch)
    taps_at: std.AutoHashMapUnmanaged(graphs.Key, ?usize) = .empty,
    g: ?Graphs = null,
    pend: []?Pend,
    /// the last window's rows (real, padded)
    n: u32 = 0,
    R: u32 = 0,
    current: []const calls.Call = &.{},
    /// TF_DSV41_DRAFT_OVERLAP: the last window recorded the runner's taps mark (its program writes taps), so the
    /// drafter's ingests of its rows may wait on that mark instead of the window's end
    marked: bool = false,
    /// the calls of `current` the graph body issues (TF_DSV41_ROUND_GRAPH's `split`: the head's, then the tail's)
    span: [2]usize = .{ 0, 0 },
    /// TF_DSV41_ROUND_GRAPH (round_graph.zig)
    rg: round_graph.Settings = .{},
    stats: struct {
        windows: u64 = 0,
        rows: u64 = 0,
        padded: u64 = 0,
        split: u64 = 0,
        /// first uses of a key: its program emitted (`builds`, host ms) and its graph captured (`captures`, the
        /// window's host ms including the capture's own run); `eager`: windows the cache ran without a graph
        builds: u64 = 0,
        build_ns: u64 = 0,
        captures: u64 = 0,
        capture_ns: u64 = 0,
        eager: u64 = 0,
        /// a window's host time on the round's critical path, by stretch: `prep` (checks, the plan link's send, pages,
        /// the program), `stage` (the row table, Engram's arm or rows), `launch` (the graph's launch or the eager
        /// window); `keep_ns`: the keeps (carries)
        prep_ns: u64 = 0,
        stage_ns: u64 = 0,
        launch_ns: u64 = 0,
        keep_ns: u64 = 0,
    } = .{},
    /// TF_DSV41_WIN_PROF=1: host ns a row window spends in each step of runRows (and in keeps), every rank, logged at
    /// deinit; null when off (no clock reads)
    prof: ?*WinProf = null,
    /// the session store (model.zig): each slot's kept rows are its committed ids
    sess: ?*sessions_gpu.Sessions = null,

    /// After Runner.bind and SlotSet.bindBases (the stacked roles have addresses): the table, the Engram stages.
    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, ss: *slots_mod.SlotSet, cap: u32) !*Batch {
        const b = try gpa.create(Batch);
        errdefer gpa.destroy(b);
        const r = f.runner;
        // graphs as the one-slot forward's default (M4: on unless TF_DSV41_GRAPHS=0; the gates' drivers: off)
        const s = try graphs.Settings.fromEnvOr(f.graphs_default);
        b.* = .{ .gpa = gpa, .f = f, .ss = ss, .cap = cap, .settings = s, .tab = try Table.init(r.d, cap), .pend = try gpa.alloc(?Pend, ss.n), .rg = try round_graph.current() };
        if (WinProf.fromEnv()) b.prof = try gpa.create(WinProf);
        if (b.prof) |p| p.* = .{};
        @memset(b.pend, null);
        {
            var la = std.heap.ArenaAllocator.init(gpa);
            defer la.deinit();
            const w = 2 * @as(usize, f.cfg.head_dim);
            for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
                try b.stash.append(gpa, .{ .layer = L, .buf = try cuda.DeviceBuffer.alloc(r.d, 2 * w * rowtab.seg_max * @as(usize, ss.n)) });
            };
        }
        try r.external(rowmode.table_role, b.tab.dev.ptr);
        if (f.opts.taps) {
            b.taps_stash = try cuda.DeviceBuffer.alloc(r.d, tapsBytes(f) * rowtab.seg_max * @as(usize, ss.n));
            try r.external(taps_stash_role, b.taps_stash.?.ptr);
        }
        // every buffer a row window bakes in, at its largest: the exchanges' send, Engram's gathered rows
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        const most: usize = @max(cap, f.planned_rows);
        _ = try f.sendBuffer(r, 2 * most * D);
        if (f.engram) |h| {
            var la = std.heap.ArenaAllocator.init(gpa);
            defer la.deinit();
            const cols = engramCols(f, h);
            for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) {
                try b.stages.append(gpa, try graphs.EngramStage.init(r.d, L, cap, cols));
            };
            const bytes = W * 2 * most * cols;
            if (f.gathered == null or f.gathered.?.len < bytes) {
                if (f.gathered) |*x| x.free();
                f.gathered = try cuda.DeviceBuffer.alloc(r.d, bytes);
            }
        }
        if (s.on) {
            b.g = @as(Graphs, undefined); // non-null, payload set by init (a bare undefined leaves the optional's tag undefined: ReleaseSafe's .? panics)
            try b.g.?.init(f, s, b.rg.split);
            std.log.scoped(.dsv41).info("row graphs: on, buckets up to {d} rows, context buckets of {d}{s}", .{ cap, s.bucket, if (b.rg.split) " (split after the head's layers)" else "" });
        }
        return b;
    }

    pub fn deinit(b: *Batch) void {
        if (b.prof) |p| {
            p.log(b.f.comm.rank());
            b.gpa.destroy(p);
        }
        if (b.g) |*g| g.deinit();
        for (b.stash.items) |*x| x.buf.free();
        b.stash.deinit(b.gpa);
        if (b.taps_stash) |*x| x.free();
        for (b.stages.items) |*st| st.deinit();
        b.stages.deinit(b.gpa);
        b.programs.deinit(b.gpa);
        b.taps_at.deinit(b.gpa);
        b.tab.deinit();
        b.gpa.free(b.pend);
        b.gpa.destroy(b);
    }

    /// The row programs' roles into the buffer plan (before Runner.bind): the stacked state sized for every slot.
    pub fn plan(f: *fwd.Forward, p: *buffers.Plan, nslots: u32, cap: u32) !void {
        const k = f.kv orelse return error.RowsNeedPool;
        // the row table is Batch's own buffer: Runner.bind leaves an external role alone (its address set in init)
        try f.runner.external(rowmode.table_role, 0);
        const a = f.arena.allocator();
        for (rowtab.buckets) |R| {
            if (R > cap) break;
            const cs = try block.emit(a, f.cfg, f.widths, rowOptions(f), try f.backbone(a), R, f.opts.limit - R, true);
            try p.add(try rowmode.transform(a, cs, R, .{ .slots = nslots, .rmax = cap, .pts = k.slot.max_pages }));
        }
    }

    pub fn pending(b: *const Batch, slot: u32) ?Pend {
        return b.pend[slot];
    }

    /// The slot's rows of the last window's taps (draft/iface.zig Taps); a stashed window's from the taps stash.
    pub fn tapsOf(b: *const Batch, slot: u32) ?struct { address: u64, rows: u32 } {
        const p = b.pend[slot] orelse return null;
        const row = tapsBytes(b.f);
        if (p.stashed) {
            const x = b.taps_stash orelse return null;
            return .{ .address = x.ptr + row * rowtab.seg_max * slot, .rows = p.n };
        }
        const base = b.f.runner.addressOf("w.taps") orelse return null;
        return .{ .address = base + row * p.a, .rows = p.n };
    }

    // -------------------------------------------------------------------------------------------------------------
    // the window

    /// Every pending window kept whole (a chain window never kept keeps every row): before a one-slot operation or
    /// a new round overwrites the window's rows. The leader's keeps go to the followers as op_rows_keep.
    pub fn settle(b: *Batch) !void {
        for (b.pend, 0..) |p, s| if (p) |x| try b.keep(@intCast(s), x.n - 1);
    }

    /// The leader's forward over `mix` (each slot at its position; `ids` every window's ids in order). `more`: a later
    /// forward of the same round (the earlier forwards' windows stay pending, their rows stashed); else every pending
    /// window is kept whole first. Logits of the real rows stay in "w.logits" rows [0, n) (Forward.greedy); each slot's
    /// window is pending until keep.
    pub fn window(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        if (!more) try b.settle();
        return b.runRows(mix, ids, more);
    }

    fn runRows(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        const f = b.f;
        var tp = ph.nowNs();
        var pt = WinProf.start(b.prof);
        defer pt.end();
        try rowtab.check(mix, b.ss.n, @intCast(f.opts.limit));
        const n = rowtab.totalRows(mix);
        if (ids.len != n) return error.BadWindow;
        const R = rowtab.bucketRows(n, b.cap) orelse return error.RowsCap;
        try b.ss.deactivate();
        for (mix) |m| if (m.start != b.ss.states[m.slot].pos or b.pend[m.slot] != null) return error.NotAChain;
        if (b.ss.leads()) try b.send(mix, ids, more);
        if (more) try b.stashPending();
        pt.mark(.send);
        // every slot's pages for its positions, the changed table rows up before the launches
        const k = f.kv.?;
        for (mix) |m| try k.slots[m.slot].ensure(m.start + m.rows);
        try k.dev.syncTables();
        pt.mark(.tables);
        const s = b.settings;
        const key: graphs.Key = .{ .rows = R, .ctx = graphs.bucketOf(rowtab.lastPos(mix), s.bucket, s.grow) };
        const cs = try b.program(key, s);
        pt.mark(.program);
        const r = f.runner;
        b.lap(&tp, &b.stats.prep_ns);
        try b.tab.stage(r.stream, mix, ids, R, b.ss.n);
        pt.mark(.rowtab);
        try b.stageEngram(mix, ids, R);
        b.lap(&tp, &b.stats.stage_ns);
        defer b.lap(&tp, &b.stats.launch_ns);
        pt.mark(.engram);
        var a: u32 = 0;
        for (mix) |m| {
            b.pend[m.slot] = .{ .a = a, .start = m.start, .n = m.rows };
            @memcpy(b.pend[m.slot].?.ids[0..m.rows], ids[a..][0..m.rows]);
            b.ss.states[m.slot].pending = .{ .start = m.start, .n = m.rows };
            a += m.rows;
        }
        b.n = n;
        b.R = R;
        b.current = cs;
        // the forward's glue reads a pending window (its rows); the row path's own steps are handled here
        f.slot = .{ .pending = .{ .start = 0, .n = R } };
        defer f.slot = .{};
        r.glue = .{ .ctx = b, .run = glueFn };
        b.stats.windows += 1;
        b.stats.rows += n;
        b.stats.padded += R - n;
        r.rows = R; // l2pf: the window's rows (with its padding, the graph's key) past 16 skip the prefetch sites
        // the taps mark (dspark_slots.zig's overlapped ingests): armed for this window alone, eager or captured alike,
        // so a key's graph holds the mark's node exactly when its program writes taps
        b.marked = false;
        if (r.taps_mark) |*m| {
            m.armed = true;
            b.marked = (b.taps_at.get(key) orelse null) != null;
        }
        defer if (r.taps_mark) |*m| {
            m.armed = false;
        };
        pt.mark(.bookkeep);
        if (b.g) |*g| {
            const fp = b.fingerprint();
            pt.mark(.fingerprint);
            // TF_DSV41_ROUND_GRAPH=split: the head's graph, then the tail's, launched while the GPU runs the head
            const at = if (b.rg.split) round_graph.splitAt(cs, b.rg.head) else null;
            const parts: [2][2]usize = if (at) |i| .{ .{ 0, i }, .{ i, cs.len } } else .{ .{ 0, cs.len }, .{ cs.len, cs.len } };
            for (parts, 0..) |sp, pi| {
                if (sp[0] == sp[1]) continue;
                b.span = sp;
                const pk: graphs.Key = .{ .rows = key.rows, .ctx = key.ctx, .part = if (at == null) 0 else @intCast(pi + 1) };
                const t0 = ph.nowNs();
                const out = try g.cache.run(pk, r.stream, fp, .{ .ctx = b, .run = body });
                pt.mark(if (pi == 0) .launch_head else .launch_tail);
                switch (out) {
                    .captured => {
                        const ns = ph.nowNs() - t0;
                        b.stats.captures += 1;
                        b.stats.capture_ns += ns;
                        if (graphLog()) std.log.scoped(.dsv41).info("row graphs: capture {d} ({d} rows, context bucket {d}, part {d}) in {d:.1} ms", .{ g.cache.stats.captured, pk.rows, pk.ctx, pk.part, @as(f64, @floatFromInt(ns)) / 1e6 });
                    },
                    .eager => b.stats.eager += 1,
                    .replayed => {},
                }
                if (outcomeLog(out, g)) std.log.scoped(.dsv41).info("row graphs: capture {d} ({d} rows, context bucket {d}, part {d}), {d} replayed", .{ g.cache.stats.captured, pk.rows, pk.ctx, pk.part, g.cache.stats.replayed });
            }
            if (at != null) b.stats.split += 1;
            return f.gatedEnd();
        }
        try r.timedWindow(cs);
        try f.gatedEnd();
    }

    fn body(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const b: *Batch = @ptrCast(@alignCast(ctx));
        _ = stream; // the runner's stream: the one the cache captures
        try b.f.runner.window(b.current[b.span[0]..b.span[1]]);
    }

    /// The key's program: block.emit at R rows ending at the context bucket's end, in row mode.
    fn program(b: *Batch, key: graphs.Key, s: graphs.Settings) ![]const calls.Call {
        if (b.programs.get(key)) |cs| return cs;
        const t0 = ph.nowNs();
        defer {
            b.stats.builds += 1;
            b.stats.build_ns += ph.nowNs() - t0;
        }
        const f = b.f;
        const a = f.arena.allocator();
        const start = graphs.emitStart(key.rows, key.ctx, s, @intCast(f.opts.limit));
        const one = try block.emit(a, f.cfg, f.widths, rowOptions(f), try f.backbone(a), key.rows, start, true);
        const cs = try rowmode.transform(a, one, key.rows, .{ .slots = b.ss.n, .rmax = b.cap, .pts = f.kv.?.slot.max_pages });
        try b.taps_at.put(b.gpa, key, run.lastTaps(cs));
        try b.programs.put(b.gpa, key, cs);
        return cs;
    }

    /// Commits slot's pending window: its first row + `accepted` drafts (Forward.keep, by row).
    pub fn keep(b: *Batch, slot: u32, accepted: u32) !void {
        const p = b.pend[slot] orelse return error.BadWindow;
        var tk = ph.nowNs();
        defer b.lap(&tk, &b.stats.keep_ns);
        if (accepted >= p.n) return error.BadWindow;
        const f = b.f;
        var pt = WinProf.start(b.prof);
        defer pt.endAs(.keep);
        if (b.ss.leads()) try f.link.?.send(&.{ slots_mod.op_rows_keep, slot, accepted });
        const r = f.runner;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(b.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const dst = b.ss.base(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            if (p.stashed) {
                const x = b.stashOf(L) orelse return error.Unbound;
                try r.kernels.others(r.stream).carry(x.buf.ptr + 2 * w * rowtab.seg_max * slot, w, 0, accepted, w, dst.base + slot * dst.bytes);
            } else {
                const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
                try r.kernels.others(r.stream).carry(src, w, 0, p.a + accepted, w, dst.base + slot * dst.bytes);
            }
        };
        const st = b.ss.state(slot);
        if (b.sess) |x| try x.commitAt(slot, p.ids[0 .. accepted + 1]);
        if (f.engram) |h| tailAfter(st, h.max_ngram - 1, p.ids[0 .. accepted + 1]);
        st.pos = p.start + accepted + 1;
        st.pending = null;
        b.pend[slot] = null;
    }

    /// 56 (op 55's neighbour, the CUDA port range 56-59): [op, n, (slot, start, rows, ids...) an ask] the
    /// Engram warm (TF_DSV41_ENGRAM_WARM), every rank's own columns
    pub const op_engram_warm: i64 = 56;
    pub const max_warm = 16;
    pub const max_warm_ids = 16;

    /// TF_DSV41_ENGRAM_WARM (target.warm): Python's commit (decode.prefetch_pending: the next window's pending row)
    /// and draft-end (decode.draft's gate.warm: the pending row and every drafted one, before depth cuts them) Engram
    /// reads, as gate warm jobs on every rank: the round's own reads then find the rows cached or in flight. Each
    /// ask's lookback is its slot's committed tail plus, for a draft before its window's keep, the rows of the window
    /// in flight that `start` implies are kept (engram_host.lookbackAt, R3's rule). Reads only: a window's rows are
    /// its own reads' bytes whatever is warmed, and a wrong guess is a wasted read.
    pub fn warmEngram(b: *Batch, asks: []const iface.Warm) !void {
        const f = b.f;
        if (f.gate == null or f.engram == null) return;
        if (b.ss.leads()) {
            var msg: [2 + max_warm * (3 + max_warm_ids)]i64 = undefined;
            var at: usize = 2;
            var n: usize = 0;
            for (asks) |a| {
                if (n == max_warm or a.ids.len == 0 or a.slot >= b.ss.n) continue;
                const m = @min(a.ids.len, max_warm_ids);
                msg[at..][0..3].* = .{ a.slot, @intCast(a.start), @intCast(m) };
                for (a.ids[0..m], msg[at + 3 ..][0..m]) |t, *y| y.* = t;
                at += 3 + m;
                n += 1;
            }
            if (n == 0) return;
            msg[0] = op_engram_warm;
            msg[1] = @intCast(n);
            try f.link.?.send(msg[0..at]);
        }
        b.warmLocal(asks);
    }

    /// op 56's asks into `asks` (their ids in `ids`); the count.
    fn parseWarm(msg: []const i64, asks: *[max_warm]iface.Warm, ids: *[max_warm * max_warm_ids]u32) !usize {
        if (msg.len < 2) return error.BadPlan;
        const n: usize = @intCast(msg[1]);
        if (n > max_warm) return error.BadPlan;
        var at: usize = 2;
        for (0..n) |i| {
            if (at + 3 > msg.len) return error.BadPlan;
            const m: usize = @intCast(msg[at + 2]);
            if (m == 0 or m > max_warm_ids or at + 3 + m > msg.len) return error.BadPlan;
            const out = ids[i * max_warm_ids ..][0..m];
            for (msg[at + 3 ..][0..m], out) |x, *y| y.* = @intCast(x);
            asks[i] = .{ .slot = @intCast(msg[at]), .start = @intCast(msg[at + 1]), .ids = out };
            at += 3 + m;
        }
        if (at != msg.len) return error.BadPlan;
        return n;
    }

    /// This rank's warm job for `asks` (the leader's or op 56's), the same items on every rank.
    fn warmLocal(b: *Batch, asks: []const iface.Warm) void {
        const f = b.f;
        const g = f.gate orelse return;
        const h = f.engram orelse return;
        var items: [max_warm]egate.Item = undefined;
        var bufs: [max_warm][2 * eh.max_ngram + 64]u32 = undefined;
        var n: usize = 0;
        for (asks) |a| {
            if (n == max_warm or a.ids.len == 0 or a.slot >= b.ss.n) continue;
            const items_n = @min(a.ids.len, max_warm_ids);
            const lb = warmLookback(b.ss.state(a.slot).*, b.pend[a.slot], a.start, h.max_ngram - 1, &bufs[n]) orelse continue;
            items[n] = .{ .ids = a.ids[0..items_n], .tail = lb };
            n += 1;
        }
        if (n > 0) g.warm(items[0..n]);
    }

    /// The slot's pending window dropped on every rank without a row committed (op_rows_drop; TF_DSV41_CALIB=measure's
    /// timed windows, calib_gpu.zig): the slot stays at its position, the next window rewrites the dropped rows'
    /// positions before any read (as a rejected draft's).
    pub fn drop(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.pend[slot] == null) return;
        if (b.ss.leads()) try b.f.link.?.send(&.{ slots_mod.op_rows_drop, slot });
        b.dropPending(slot);
    }

    fn dropPending(b: *Batch, slot: u32) void {
        b.pend[slot] = null;
        b.ss.state(slot).pending = null;
    }

    fn stashOf(b: *Batch, L: u32) ?*Stash {
        for (b.stash.items) |*x| if (x.layer == L) return x;
        return null;
    }

    /// The pending windows' ratio-2 projection rows copied aside (every rank: op_rows' `more` flag), before a later
    /// forward of the round overwrites them.
    fn stashPending(b: *Batch) !void {
        const r = b.f.runner;
        var nb: [64]u8 = undefined;
        const w = 2 * @as(usize, b.f.cfg.head_dim);
        for (b.pend, 0..) |*pp, s| if (pp.*) |*p| if (!p.stashed) {
            for (b.stash.items) |*x| {
                const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{x.layer})) orelse return error.Unbound;
                try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.buf.ptr + 2 * w * rowtab.seg_max * s, src + 2 * w * p.a, 2 * w * p.n, r.stream.handle), "cuMemcpyDtoDAsync");
            }
            if (b.taps_stash) |x| {
                const t = tapsBytes(b.f);
                const src = r.addressOf("w.taps") orelse return error.Unbound;
                try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.ptr + t * rowtab.seg_max * s, src + t * p.a, t * p.n, r.stream.handle), "cuMemcpyDtoDAsync");
            }
            p.stashed = true;
        };
    }

    /// A slot leaves (every rank, the leader's order): its pending window dropped, then Forward.release on it.
    pub fn release(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.ss.leads()) try b.f.link.?.send(&.{ slots_mod.op_rows_drop, slot });
        b.dropPending(slot);
        try b.ss.activate(slot);
        try b.f.release();
    }

    /// A slot leaves as a session turn (target.zig, outside the served shape): its pending window kept whole (a chain
    /// window never kept keeps every row), the slot activated, the entry saved and the slot emptied (every rank:
    /// op_rows_keep, op_select, op_sess_save).
    pub fn saveTurn(b: *Batch, slot: u32, ss: *sessions_gpu.Sessions) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.pend[slot]) |p| try b.keep(slot, p.n - 1);
        try b.ss.activate(slot);
        _ = try ss.save();
    }

    // -------------------------------------------------------------------------------------------------------------
    // draft/iface.zig's Target over row windows (target.zig's hooks)

    /// Target.window: the segments as one round (several forwards when their rows pass the cap), each row's choice
    /// into `choices`: greedy, or (T > 0, with `sampler`) the slot's keyed choices over its own rows. A forward's
    /// segments sit back to back in "w.logits" rows [0, n), in segment order: segment i's rows are [a_i, a_i + len_i),
    /// one GpuSampler.choose (one op_sample) each, after the forward's greedy (only when one of its segments is greedy).
    pub fn targetWindow(b: *Batch, segments: []const iface.Segment, choices: [][]u32, sampler: ?*samp.GpuSampler) !void {
        if (segments.len == 0 or segments.len > 16) return error.Slots;
        var mix: [16]rowtab.Seg = undefined;
        for (segments, mix[0..segments.len]) |sg, *m| {
            if (sg.parents != null) return error.TreeWindow;
            if (samp.sampled(sg.sampling) and sampler == null) return error.Sampling;
            if (sg.slot >= b.ss.n) return error.Slots;
            // a pending window must end where the segment starts (a chain never kept keeps every row)
            if (b.pend[sg.slot]) |p| if (sg.start != p.start + p.n) return error.NotAChain;
            m.* = .{ .slot = sg.slot, .start = sg.start, .rows = @intCast(sg.tokens.len) };
        }
        var ranges: [16][2]usize = undefined;
        const nf = try rowtab.pack(mix[0..segments.len], b.cap, &ranges);
        var ids: [64]u32 = undefined;
        var picks: [64]u32 = undefined;
        for (ranges[0..nf], 0..) |rg, j| {
            var n: usize = 0;
            for (segments[rg[0]..rg[1]]) |sg| {
                @memcpy(ids[n..][0..sg.tokens.len], sg.tokens);
                n += sg.tokens.len;
            }
            try b.window(mix[rg[0]..rg[1]], ids[0..n], j > 0);
            var greedy_rows = false;
            for (segments[rg[0]..rg[1]]) |sg| greedy_rows = greedy_rows or !samp.sampled(sg.sampling);
            if (greedy_rows) try b.f.greedy(picks[0..n]);
            // TF_DSV41_SAMP_BATCH: the forward's keyed segments in one sampler call (when two or more), the rest (whole
            // rows) one call a segment as before
            var many: [16]samp.Seg = undefined;
            var nm: usize = 0;
            var batched: [16]bool = @splat(false);
            if (sampler) |x| if (x.batch) {
                var a: usize = 0;
                for (segments[rg[0]..rg[1]], choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
                    if (samp.sampled(sg.sampling) and x.batchable(sg.sampling.?)) {
                        many[nm] = .{ .row0 = @intCast(a), .n = @intCast(sg.tokens.len), .s = sg.sampling.?, .positions = sg.draws, .out = ch[0..sg.tokens.len] };
                        batched[i] = true;
                        nm += 1;
                    }
                    a += sg.tokens.len;
                }
                if (nm >= 2) try x.chooseSegments(many[0..nm]) else batched = @splat(false);
            };
            var a: usize = 0;
            for (segments[rg[0]..rg[1]], choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
                const rows = ch[0..sg.tokens.len];
                if (batched[i]) {} else if (samp.sampled(sg.sampling))
                    try sampler.?.choose(@intCast(a), @intCast(rows.len), sg.sampling.?, sg.draws, rows)
                else
                    @memcpy(rows, picks[a..][0..rows.len]);
                a += rows.len;
            }
        }
    }

    /// Target.keep: a chain's rows 0..k.
    pub fn keepPath(b: *Batch, slot: u32, path: []const u32) !void {
        if (slot >= b.ss.n or path.len == 0) return error.NotAChain;
        for (path, 0..) |row, i| if (row != i) return error.NotAChain;
        try b.f.engramCheck(); // a gated window whose rows failed: its picks are not committed
        return b.keep(slot, @intCast(path.len - 1));
    }

    /// Target.prefill's start: pending windows kept, the slot active for the one-slot prefill.
    pub fn beginPrefill(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        try b.settle();
        try b.ss.activate(slot);
    }

    fn send(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        var msg: [2 + 3 * 16 + 64]i64 = undefined;
        if (mix.len > 16 or ids.len > 64) return error.BadWindow;
        msg[0] = slots_mod.op_rows;
        msg[1] = @as(i64, @intCast(mix.len)) | (@as(i64, @intFromBool(more)) << 8);
        for (mix, 0..) |m, i| {
            msg[2 + 3 * i] = m.slot;
            msg[3 + 3 * i] = @intCast(m.start);
            msg[4 + 3 * i] = m.rows;
        }
        const at = 2 + 3 * mix.len;
        for (ids, msg[at .. at + ids.len]) |t, *y| y.* = t;
        try b.f.link.?.send(msg[0 .. at + ids.len]);
    }

    /// A follower: the leader's op_rows / op_rows_keep / op_select (Forward.follow's extension).
    pub fn follow(b: *Batch, msg: []const i64) !void {
        switch (msg[0]) {
            slots_mod.op_select => try b.ss.activate(@intCast(msg[1])),
            slots_mod.op_rows_keep => try b.keep(@intCast(msg[1]), @intCast(msg[2])),
            slots_mod.op_rows_drop => b.dropPending(@intCast(msg[1])),
            op_engram_warm => {
                var asks: [max_warm]iface.Warm = undefined;
                var ids: [max_warm * max_warm_ids]u32 = undefined;
                const n = try parseWarm(msg, &asks, &ids);
                b.warmLocal(asks[0..n]);
            },
            slots_mod.op_rows => {
                const k: usize = @intCast(msg[1] & 0xff);
                const more = (msg[1] >> 8) & 1 == 1;
                if (k == 0 or k > 16 or msg.len < 2 + 3 * k) return error.BadPlan;
                var mix: [16]rowtab.Seg = undefined;
                for (mix[0..k], 0..) |*m, i| m.* = .{ .slot = @intCast(msg[2 + 3 * i]), .start = @intCast(msg[3 + 3 * i]), .rows = @intCast(msg[4 + 3 * i]) };
                var ids: [64]u32 = undefined;
                const rest = msg[2 + 3 * k ..];
                if (rest.len > ids.len) return error.BadPlan;
                for (rest, ids[0..rest.len]) |t, *y| y.* = @intCast(t);
                try b.runRows(mix[0..k], ids[0..rest.len], more);
            },
            else => return error.BadPlan,
        }
    }

    pub fn ext(b: *Batch) fwd.Forward.Ext {
        return .{ .ptr = b, .run = followFn };
    }

    fn followFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const b: *Batch = @ptrCast(@alignCast(ptr));
        return b.follow(msg);
    }

    // -------------------------------------------------------------------------------------------------------------
    // inputs and glue

    /// Each Engram layer's rows of the window: a segment's ids with its slot's tail through the forward's source
    /// (Host.windowRows: R3's table under TF_DSV41_ENGRAM_DIR, else M2a's index rows, as Forward.fillEngramRows);
    /// padding rows the last real row's (Python's in-graph mirror gather, the same bytes).
    fn stageEngram(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, R: u32) !void {
        const f = b.f;
        f.gated = false;
        const h = f.engram orelse return;
        if (b.stages.items.len == 0) return;
        // the table opens on the forward's first Engram window (a prompt always comes first); never index rows instead
        if (std.c.getenv("TF_DSV41_ENGRAM_DIR") != null and f.engram_rows == null) return error.EngramNotOpen;
        const src: ?eh.Source = try f.engramSource(h); // the gate's locked table while it lives
        const n = rowtab.totalRows(mix);
        const dim: usize = f.cfg.engram_head_dim;
        // TF_DSV41_ENGRAM_GATE: the round to the gate's worker (the bucket's padding its last row, as below); the
        // window's engram_rows steps launch the gate kernel
        {
            var items: [64]egate.Item = undefined;
            var a: usize = 0;
            for (mix, 0..) |m, i| {
                const st = &b.ss.states[m.slot];
                items[i] = .{ .ids = ids[a .. a + m.rows], .tail = st.tail[0..st.tail_len] };
                a += m.rows;
            }
            f.gated = mix.len <= items.len and try f.armEngram(items[0..mix.len], R);
            if (f.gated) return;
        }
        // every slot's rows on every Engram layer in flight first (Python's round job: one batched read), so the
        // per-slot, per-layer waits below cost one read latency, not one a slot and layer
        if (src != null) {
            var a: usize = 0;
            for (mix) |m| {
                const st = &b.ss.states[m.slot];
                f.issueEngram(ids[a .. a + m.rows], st.tail[0..st.tail_len]);
                a += m.rows;
            }
        }
        for (b.stages.items) |*stg| {
            const cols = stg.cols;
            const out = try stg.begin(R);
            var a: usize = 0;
            for (mix) |m| {
                const st = &b.ss.states[m.slot];
                try h.windowRows(b.gpa, src, ids[a .. a + m.rows], st.tail[0..st.tail_len], stg.layer, f.comm.rank(), f.comm.world(), dim, out[a * cols .. (a + m.rows) * cols]);
                a += m.rows;
            }
            for (n..R) |t| @memcpy(out[t * cols ..][0..cols], out[(n - 1) * cols ..][0..cols]);
            try stg.commit(f.runner.stream, R);
        }
    }

    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const b: *Batch = @ptrCast(@alignCast(ctx));
        const f = b.f;
        const step = c.name["glue.".len..];
        const k = r.kernels.others(r.stream);
        const s = r.stream.handle;
        const R: usize = b.R;
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        if (std.mem.eql(u8, step, "positions")) {
            // positions come from the table; the SWA lower bounds are zeros (Context.lo)
            const lo = try r.tensorAddr(c.args[2].arg.t);
            return r.d.check(r.d.api.cuMemsetD32Async(lo, 0, R, s), "cuMemsetD32Async");
        }
        if (std.mem.eql(u8, step, "embed")) {
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const V: usize = f.cfg.vocab / W;
            const rows = try f.sendBuffer(r, 2 * R * D);
            try k.embedSend(b.tab.dev.ptr + rowtab.colOffset(.ids, b.cap), R, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try f.comm.allGather(rows, recv, R * D, .bf16, s);
            return k.embedSum(recv, W, R, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "engram_rows")) {
            const L: u32 = @intCast(c.args[1].arg.i);
            for (b.stages.items) |*st| if (st.layer == L) {
                if (f.gated) try f.gateCopy(r, L, st.dev.ptr, R);
                try f.comm.allGather(st.dev.ptr, f.gathered.?.ptr, R * st.cols, .bf16, s);
                return k.gatherCols(f.gathered.?.ptr, W, R, st.cols, try r.tensorAddr(c.args[0].arg.t));
            };
            return error.NotEngram;
        }
        if (std.mem.eql(u8, step, "kx_dense")) return b.exchange(r, c);
        if (std.mem.eql(u8, step, "trace")) return error.TracedRows;
        return fwd.Forward.glueFn(f, r, c);
    }

    /// Split KV's dense exchange in row mode: each row's selection through its slot's row of the stacked split table.
    fn exchange(b: *Batch, r: *run.Runner, c: *const calls.Call) !void {
        const kx = b.f.kv orelse return error.GlueNotBuilt;
        const x = c.args;
        if (x.len != 7) return error.BadGlue;
        const sel = x[0].arg.t;
        const da: kv.split.DenseArgs = .{
            .sel = try r.tensorAddr(sel), .rows = @intCast(sel.shape[0]), .k = @intCast(sel.shape[1]),
            .table = try r.tensorAddr(x[2].arg.t), .pts = kx.slot.max_pages,
            .rslot = b.tab.dev.ptr + rowtab.rslot32Offset(b.cap), .psh = @intCast(x[6].arg.i),
            .base = try r.tensorAddr(x[1].arg.t), .row_bytes = block.Pool.row_bytes, .world = kx.world,
            .send = try r.tensorAddr(x[3].arg.t), .tok = try r.tensorAddr(x[5].arg.t),
        };
        try kx.x.window(kv.split.PackArgs.of(da, kx.rank, kx.lens.?.ptr), try r.tensorAddr(x[4].arg.t), r.stream.handle);
        kx.stats.dense += 1;
    }

    /// The addresses a row graph bakes in beyond its program's roles: any change drops every graph.
    /// Adds the time since `*t` to `*acc` and restarts `*t`.
    fn lap(_: *const Batch, t: *u64, acc: *u64) void {
        const now = ph.nowNs();
        acc.* += now - t.*;
        t.* = now;
    }

    fn fingerprint(b: *const Batch) u64 {
        const f = b.f;
        var h = std.hash.Wyhash.init(0x5107);
        const r = f.runner;
        for (r.owned.items) |x| {
            h.update(std.mem.asBytes(&x.ptr));
            h.update(std.mem.asBytes(&x.len));
        }
        for ([_]?cuda.DeviceBuffer{ r.scratch, f.send, f.gathered, b.tab.dev }) |x| if (x) |y| h.update(std.mem.asBytes(&y.ptr));
        for (b.stages.items) |st| h.update(std.mem.asBytes(&st.dev.ptr));
        for (b.ss.roles.values()) |st| h.update(std.mem.asBytes(&st.base));
        if (f.kv) |k| for (k.dev.tensors) |t| h.update(std.mem.asBytes(&t.ptr));
        // the taps mark's event: a graph captured without it (or with another) records no mark, so it is dropped
        if (r.taps_mark) |m| h.update(std.mem.asBytes(&m.ev.handle));
        return h.final();
    }
};

/// Bytes of one taps row (bf16 [3 x hidden]).
/// The forward's options for a row-mode program (block.Options.rows: RowWin's selections).
fn rowOptions(f: *const fwd.Forward) block.Options {
    var o = f.opts;
    o.rows = true;
    return o;
}

fn tapsBytes(f: *const fwd.Forward) usize {
    return 2 * f.cfg.dspark_targets.items().len * @as(usize, f.cfg.hidden);
}

/// This rank's columns of an Engram row (its heads' share).
fn engramCols(f: *const fwd.Forward, h: *const eh.Host) usize {
    return h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep_n:].
fn tailAfter(st: *fwd.Slot, keep_n: usize, ids: []const u32) void {
    st.tail_len = eh.tailAfter(&st.tail, st.tail_len, keep_n, ids);
}

/// The row table on the device ([NCOL, rmax] int64, then the int32 rslot copy) and its two pinned staging halves; a
/// half is rewritten only after its previous copy finished (its event), so no step waits on the window in flight.
pub const Table = struct {
    rmax: u32,
    dev: cuda.DeviceBuffer,
    host: [2]cuda.HostBuffer,
    copied: [2]cuda.Event,
    half: u1 = 0,

    pub fn init(d: *const cuda.Driver, rmax: u32) !Table {
        const n = rowtab.tableBytes(rmax);
        var t: Table = .{ .rmax = rmax, .dev = try cuda.DeviceBuffer.alloc(d, n), .host = undefined, .copied = undefined };
        try t.dev.fill8(0, null);
        for (&t.host, &t.copied) |*h, *e| {
            h.* = try cuda.HostBuffer.alloc(d, n);
            @memset(h.bytes, 0);
            e.* = try cuda.Event.init(d, false);
            try e.record(.{ .d = d, .handle = null });
        }
        return t;
    }

    pub fn deinit(t: *Table) void {
        t.dev.free();
        for (&t.host, &t.copied) |*h, *e| {
            h.free();
            e.deinit();
        }
    }

    /// The mix's table (rowtab.plan) copied on `stream` ahead of the window.
    pub fn stage(t: *Table, stream: cuda.Stream, mix: []const rowtab.Seg, ids: []const u32, R: u32, nslots: u32) !void {
        const h = t.half;
        try t.copied[h].synchronize();
        const bytes = t.host[h].bytes;
        const cols: []i64 = @alignCast(std.mem.bytesAsSlice(i64, bytes[0 .. 8 * rowtab.ncol * @as(usize, t.rmax)]));
        const r32: []i32 = @alignCast(std.mem.bytesAsSlice(i32, bytes[rowtab.rslot32Offset(t.rmax)..][0 .. 4 * @as(usize, t.rmax)]));
        try rowtab.plan(mix, ids, R, t.rmax, nslots, cols, r32);
        try t.dev.uploadAsync(0, bytes, stream.handle);
        try t.copied[h].record(stream);
        t.half = h +% 1;
    }
};

const Stash = struct { layer: u32, buf: cuda.DeviceBuffer };

/// TF_DSV41_GRAPH_LOG=1: every row-graph capture logged with its key and host ms (else the powers of two).
fn graphLog() bool {
    const v = std.c.getenv("TF_DSV41_GRAPH_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn outcomeLog(out: graphs.gc.Outcome, g: *const Graphs) bool {
    return out == .captured and (g.cache.stats.captured & (g.cache.stats.captured - 1)) == 0;
}

/// Row graphs' state: the cache over the runner's stream, keyed (bucket rows, context bucket), every rank agreeing.
const Graphs = struct {
    engine: graphs.gc.CudaEngine,
    agreement: graphs.Agreement,
    cache: graphs.Cache,

    /// `split`: a window's two graphs (head, tail) held in one cache of twice the cap (a head is ~1/43 of a window)
    fn init(g: *Graphs, f: *fwd.Forward, s: graphs.Settings, split: bool) !void {
        const r = f.runner;
        g.* = .{ .engine = .{ .d = r.d }, .agreement = try graphs.Agreement.init(r.d, f.comm, r.stream), .cache = undefined };
        g.cache = graphs.Cache.init(f.gpa, g.engine.engine(), .{ .on = true, .max = if (split) 2 * s.max else s.max, .hold = s.floor_hold });
        g.cache.agree = g.agreement.agree();
        g.cache.room = graphs.room;
        g.cache.meter = graphs.meter;
        g.cache.name = "row graphs";
        graphs.rule = s.room;
    }

    fn deinit(g: *Graphs) void {
        const cs = g.cache.stats;
        std.log.scoped(.dsv41).info("row graphs: {d} captured, {d} replayed, {d} eager, {d} keys held, {d} evicted, {d} dropped{s}", .{ cs.captured, cs.replayed, cs.eager, g.cache.count(), cs.evicted, cs.dropped, if (g.cache.failed) " (off after a failed capture)" else "" });
        g.cache.deinit();
        g.agreement.deinit();
    }
};


/// The lookback a warm of a window at `start` reads after: the slot's committed tail, plus the rows of its pending
/// window (starting at the slot's position) that `start` implies are kept; null when `start` is not reachable.
pub fn warmLookback(st: anytype, pend: anytype, start: u64, keep_n: usize, buf: []u32) ?[]const u32 {
    const pending: ?[]const u32 = if (pend) |p| (if (p.start == st.pos) p.ids[0..p.n] else null) else null;
    return eh.lookbackAt(st.tail[0..st.tail_len], st.pos, pending, start, keep_n, buf);
}

/// TF_DSV41_WIN_PROF=1 (Batch.prof): host time of a row window's steps (runRows: the plan send, page tables, the
/// program, the row table, Engram staging / the gate's arm, bookkeeping, the fingerprint, the head and tail graph
/// launches, the rest) and of keeps, summed over the run; logged by every rank at deinit. Host clock reads only.
pub const WinProf = struct {
    pub const Step = enum { send, tables, program, rowtab, engram, bookkeep, fingerprint, launch_head, launch_tail, rest, keep };
    const nsteps = std.enums.values(Step).len;
    ns: [nsteps]u64 = @splat(0),
    n: [nsteps]u64 = @splat(0),
    windows: u64 = 0,

    pub fn fromEnv() bool {
        const v = std.c.getenv("TF_DSV41_WIN_PROF") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// A timer from now (no clock read when profiling is off).
    pub fn start(p: ?*WinProf) Timer {
        return .{ .p = p, .last = if (p != null) ph.nowNs() else 0 };
    }

    pub const Timer = struct {
        p: ?*WinProf,
        last: u64,

        /// The time since the last mark (or the start) booked to `s`.
        pub fn mark(t: *Timer, s: Step) void {
            const p = t.p orelse return;
            const x = ph.nowNs();
            p.ns[@intFromEnum(s)] += x - t.last;
            p.n[@intFromEnum(s)] += 1;
            t.last = x;
        }

        pub fn end(t: *Timer) void {
            const p = t.p orelse return;
            t.mark(.rest);
            p.windows += 1;
        }

        pub fn endAs(t: *Timer, s: Step) void {
            if (t.p == null) return;
            t.mark(s);
        }
    };

    pub fn log(p: *const WinProf, rank: u32) void {
        var buf: [1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (std.enums.values(Step)) |st| {
            const i = @intFromEnum(st);
            if (p.n[i] > 0) w.print(" {t} {d:.1}", .{ st, @as(f64, @floatFromInt(p.ns[i])) / 1e3 / @as(f64, @floatFromInt(p.n[i])) }) catch {};
        }
        std.log.scoped(.dsv41).info("window host steps (rank {d}, {d} windows, us a call):{s}", .{ rank, p.windows, w.buffered() });
    }
};

test "TF_DSV41_WIN_PROF: marks book each step's time once, the rest at the end, keeps on their own" {
    var p: WinProf = .{};
    var t = WinProf.start(&p);
    t.mark(.send);
    t.mark(.engram);
    t.end();
    var k = WinProf.start(&p);
    k.endAs(.keep);
    try std.testing.expectEqual(@as(u64, 1), p.windows);
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(WinProf.Step.send)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(WinProf.Step.engram)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(WinProf.Step.rest)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(WinProf.Step.keep)]);
    try std.testing.expectEqual(@as(u64, 0), p.n[@intFromEnum(WinProf.Step.tables)]);
    var off = WinProf.start(null);
    off.mark(.send);
    off.end();
    try std.testing.expectEqual(@as(u64, 0), off.last);
}
