//! Prod's round planner (Python 8474f31 batch.py, the batcher's `_plan`) for the served engine's lane host: which queued
//! request starts this round and which prompt rows run between the decode rounds. Rank 0's decisions only; the host
//! (core/lane_host.zig with a `Rounds`, serve_engine.zig) runs them through the lanes, and the followers replay the
//! leader's forward operations as before. The rules, each as prod's:
//! - **Admission** (`admit`, batch.py `_admit` 383-447): foreground before background, FIFO (the host's queue order); a
//!   request needs a free slot; one with more than `long_prompt` tokens left to prefill waits while `concurrent` such
//!   prompts are admitted and still prefilling (a `.skip`: shorter prompts behind it go); the memory floor (`Floor`, when
//!   a memory source is set: the prompt's priced prefill transient plus what admitted prompts still owe, against the
//!   usable memory less the hard floor) else a `.wait` (FIFO: nothing behind it either), refused (503) only when nothing
//!   is live and it waited `refuse_s`; a request past the whole pool is refused; the pool's pages (`sess.admit` with the
//!   round's resumed entries kept and at most one round of spills) else a `.wait`.
//! - **Pieces** (`plan`, batch.py `_pieces` 476-501): prompts with at most `short` tokens left always, the others when the
//!   fair share allows (`Fairness`, GLM batchplan), fewest tokens left first (then the earliest submitted), as many as
//!   the round's rows hold (`rows`, or G19's adaptive step: `Adapt`, `roundNeed`); a piece ends at the prompt's replay
//!   point (its snapshot runs right after it, rounds.py `_save`) or at the prompt's last token less one, on the 16-token
//!   grid (`pieceEnd`); a prompt whose rows are all in is a final this round (the prompt's last token is the first
//!   verify window's pending row).
//! - **A damaged NVMe entry** (`started(.., damaged)`, batch.py `_sample` 561-568): the plan was made from the entry's
//!   length, so its piece is planned (the round's rows spent) and not run; after the round the request prefills from 0,
//!   its snapshot point unconditional (prod's reset).
//! - **After the round** (`after`): the fair share's debt from the pieces' and the decode's seconds.

const std = @import("std");
const sess = @import("sess.zig");

pub const grid: u64 = 16;
pub const GiB: f64 = @floatFromInt(@as(u64, 1) << 30);

/// plan.py `piece_end`: the end of a slot's next piece from `done` toward `target` with `budget` rows: `target` when it
/// fits, else the last grid point within the budget (at least one grid step: a piece is never empty).
pub fn pieceEnd(done: u64, target: u64, budget: u64, g: u64) u64 {
    if (target - done <= budget) return target;
    var end = (done + budget) / g * g;
    if (end <= done) end = @min(target, (done / g + 1) * g);
    return end;
}

/// sessions.snapshot_point: the last grid point strictly before a prompt's end.
pub fn snapshotPoint(n: u64) u64 {
    return if (n > 0) (n - 1) / grid * grid else 0;
}

/// GLM batchplan.Fairness: a piece of t seconds is followed by at least t (1 - share) / share seconds of decode rounds
/// (and at least one round) before the next piece; with nothing decoding, pieces run back to back.
pub const Fairness = struct {
    share: f64 = 0.5,
    debt: f64 = 0.0,
    rounds: u64 = 1,

    pub fn allow(f: *const Fairness, decoding: bool) bool {
        if (!decoding) return true;
        return f.rounds >= 1 and f.debt <= 0.0;
    }

    pub fn after(f: *Fairness, piece_s: f64, round_s: f64, prefilling: bool) void {
        if (piece_s > 0.0) {
            f.debt = @max(f.debt, 0.0) + piece_s * (1.0 - f.share) / f.share;
            f.rounds = 0;
        }
        if (round_s > 0.0) {
            f.debt -= round_s;
            f.rounds += 1;
        }
        if (!prefilling) {
            f.debt = 0.0;
            f.rounds = 1;
        }
    }
};

/// memory.py's prefill price knobs (TF_DSV41_FLOOR_PF_*_GIB, TF_DSV41_INDEX_BUDGET_MIB, TF_DSV41_FLOOR_PRICE).
pub const PriceKnobs = struct {
    on: bool = true,
    chunk_gib: f64 = 0.533,
    index_gib: f64 = 1.067,
    host_gib: f64 = 0.6,
    other_gib: f64 = 0.0,
    index_positions: u64 = 32768,
    budget_mib: u64 = 256,
};

/// memory.Price: a prompt's prefill transient in bytes (`once` with its first piece, `index` over its positions below
/// `index_positions`, from `start` to `end`).
pub const Price = struct {
    once: i64 = 0,
    index: i64 = 0,
    start: u64 = 0,
    end: u64 = 0,

    pub fn total(p: Price) i64 {
        return p.once + p.index;
    }

    /// What has not materialised once the prompt's state holds `done` tokens.
    pub fn outstanding(p: Price, done: u64) i64 {
        if (p.total() == 0) return 0;
        const once: i64 = if (done <= p.start) p.once else 0;
        const span = p.end - p.start;
        const left = p.end -| @max(done, p.start);
        // int(index * left / span): the exact quotient is at least 1/span from the next integer, so the integer
        // division is Python's truncated float
        const idx: i64 = if (span > 0) @intCast(@divTrunc(@as(i128, p.index) * left, span)) else 0;
        return once + idx;
    }
};

/// memory.prefill_price: the transient of prefilling tokens [cached, n - 1) in windows of `rows`.
pub fn prefillPrice(k: PriceKnobs, n: u64, cached: u64, rows: u64) Price {
    if (!k.on) return .{};
    if (n < 1 + cached + 1) return .{}; // todo = n - 1 - cached <= 0
    const todo = n - 1 - cached;
    const w = @as(f64, @floatFromInt(@min(rows, todo))) / 2048.0;
    const once = (k.chunk_gib + k.host_gib) * w + k.other_gib;
    const start = cached;
    const end = @min(n - 1, k.index_positions);
    const share = @as(f64, @floatFromInt(end -| start)) / @as(f64, @floatFromInt(k.index_positions));
    const index = k.index_gib * @as(f64, @floatFromInt(k.budget_mib)) / 256.0 * share;
    return .{ .once = @intFromFloat(once * GiB), .index = @intFromFloat(index * GiB), .start = start, .end = @max(start, end) };
}

/// memory.round_need: bytes the next prefill round takes at `rows` rows: one window's `once` + each prefilling prompt's
/// index growth still ahead (`left`: (prompt length, tokens done)).
pub fn roundNeed(k: PriceKnobs, rows: u64, left: []const [2]u64) i64 {
    if (left.len == 0) return 0;
    const once = prefillPrice(k, rows + 1, 0, rows).once;
    var index: i64 = 0;
    for (left) |x| index += prefillPrice(k, x[0], x[1], rows).index;
    return once + index;
}

/// memory.Adapt (G19): TF_DSV41_PREFILL_ADAPT (on), _GIB (what the tighter rank keeps after the round's transient; the
/// floor target), _UP_GIB (more to step back up), _MIN (the smallest rows a round).
pub const Adapt = struct {
    gib: f64 = 5.0,
    up_gib: f64 = 0.5,
    min_rows: u64 = 512,

    /// The rows a round may take, largest first: `full`, its half, ... down to `min_rows` (on the grid).
    pub fn steps(a: Adapt, full: u64, out: *[16]u64) []const u64 {
        out[0] = full;
        var n: usize = 1;
        var r = full;
        while (r / 2 >= a.min_rows and n < out.len) {
            r = @max(grid, r / 2 / grid * grid);
            out[n] = r;
            n += 1;
        }
        return out[0..n];
    }

    /// The largest step whose need leaves `usable` at or above `gib` (+ `up_gib` above `current`); else the smallest.
    pub fn choose(a: Adapt, k: PriceKnobs, full: u64, usable: i64, left: []const [2]u64, current: ?u64) u64 {
        var buf: [16]u64 = undefined;
        const st = a.steps(full, &buf);
        for (st) |r| {
            const keep = a.gib + (if (current != null and r > current.?) a.up_gib else 0.0);
            if (@as(f64, @floatFromInt(usable - roundNeed(k, r, left))) >= keep * GiB) return r;
        }
        return st[st.len - 1];
    }
};

/// The /proc/meminfo lines the floor reads (bytes).
pub const Meminfo = struct { free: i64, available: i64, dirty: i64 = 0, writeback: i64 = 0, mapped: i64 = 0 };

/// memory.Floor on rank 0 (rank 1's reports: none, as prod when they are stale): MemAvailable kept above the hard
/// floor by accounting. `usable` = MemFree + the page cache the kernel can hand over (memsafe.view "available").
pub const Floor = struct {
    target_gib: f64 = 5.0,
    hard_gib: f64 = 4.0,
    /// TF_DSV41_ADMIT_FREE_FLOOR_GIB: immediately free on top (default 0)
    free_gib: f64 = 0.0,
    /// TF_DSV41_FLOOR_REFUSE_S: nothing live this long -> refused
    refuse_s: f64 = 30.0,
    /// GLM53_TF_ADMIT_CACHE_KEEP_GB: page cache never counted as available
    keep_gib: f64 = 2.0,
    /// the usable bytes of the last check (the refusal's test)
    last: i64 = 0,

    pub fn usable(f: *const Floor, mi: Meminfo) i64 {
        const keep: i64 = @intFromFloat(f.keep_gib * GiB);
        const credit = @max(0, mi.available - mi.free - (mi.dirty + mi.writeback) - @max(mi.mapped, keep));
        return mi.free + credit;
    }

    /// Something new may start: usable less `need` stays at or above the hard floor, and MemFree at `free_gib`.
    pub fn check(f: *Floor, mi: Meminfo, need: i64) bool {
        const u = f.usable(mi);
        f.last = u;
        const want = need + @as(i64, @intFromFloat(f.hard_gib * GiB));
        return u >= want and mi.free >= @as(i64, @intFromFloat(f.free_gib * GiB));
    }

    pub fn fitsIdle(f: *const Floor, need: i64) bool {
        return @as(f64, @floatFromInt(f.last - need)) >= f.hard_gib * GiB;
    }
};

pub const Settings = struct {
    /// TF_DSV41_PREFILL_ROWS: rows of prompt a round (the forward's segment)
    rows: u64 = 2048,
    /// TF_DSV41_BATCH_SHORT: prompts with at most this many tokens left go every round
    short: u64 = 512,
    /// TF_DSV41_PREFILL_SHARE
    share: f64 = 0.5,
    /// TF_DSV41_LONG_PROMPT / TF_DSV41_PREFILL_CONCURRENT (0: no limit)
    long_prompt: u64 = 32768,
    concurrent: u64 = 1,
    /// rounds.py session_min: shorter prompts are not snapshotted
    session_min: u64 = 64,
    /// a session store is open (prompt snapshots at the replay point)
    sessions: bool = true,
    /// G19's adaptive rows (prod: on whenever the floor is): null = the fixed rows
    adapt: ?Adapt = null,
    /// the memory floor (null: no memory source, as prod's rank 1 / a floor-less batcher)
    floor: ?Floor = null,
    price: PriceKnobs = .{},

    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        if (envInt("TF_DSV41_PREFILL_ROWS")) |v| s.rows = v;
        if (envInt("TF_DSV41_BATCH_SHORT")) |v| s.short = v;
        if (envFloat("TF_DSV41_PREFILL_SHARE")) |v| s.share = v;
        if (envInt("TF_DSV41_LONG_PROMPT")) |v| s.long_prompt = v;
        if (envInt("TF_DSV41_PREFILL_CONCURRENT")) |v| s.concurrent = v;
        if (!(s.share > 0.0 and s.share <= 1.0) or s.long_prompt < 1) return error.BadSettings;
        if (envInt("TF_DSV41_INDEX_BUDGET_MIB")) |v| s.price.budget_mib = v;
        if (envFloat("TF_DSV41_FLOOR_PF_CHUNK_GIB")) |v| s.price.chunk_gib = v;
        if (envFloat("TF_DSV41_FLOOR_PF_INDEX_GIB")) |v| s.price.index_gib = v;
        if (envFloat("TF_DSV41_FLOOR_PF_HOST_GIB")) |v| s.price.host_gib = v;
        if (envFloat("TF_DSV41_FLOOR_PF_OTHER_GIB")) |v| s.price.other_gib = v;
        if (std.c.getenv("TF_DSV41_FLOOR_PRICE")) |v| s.price.on = !std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "0");
        return s;
    }

    /// The floor and G19's rows as prod's rank 0 runs them (memory.Floor.from_env, memory.Adapt.from_env).
    pub fn withFloorFromEnv(s: *Settings) !void {
        var f: Floor = .{};
        if (envFloat("TF_DSV41_FLOOR_GIB")) |v| f.target_gib = v;
        if (envFloat("TF_DSV41_FLOOR_HARD_GIB")) |v| f.hard_gib = v;
        if (!(f.hard_gib > 0 and f.hard_gib <= f.target_gib)) return error.BadSettings;
        if (envFloat("TF_DSV41_ADMIT_FREE_FLOOR_GIB")) |v| f.free_gib = v;
        if (envFloat("TF_DSV41_FLOOR_REFUSE_S")) |v| f.refuse_s = v;
        if (envFloat("GLM53_TF_ADMIT_CACHE_KEEP_GB")) |v| f.keep_gib = v;
        s.floor = f;
        const off = if (std.c.getenv("TF_DSV41_PREFILL_ADAPT")) |v| std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "0") else false;
        if (off) return;
        var a: Adapt = .{ .gib = f.target_gib };
        if (envFloat("TF_DSV41_PREFILL_ADAPT_GIB")) |v| a.gib = v;
        if (envFloat("TF_DSV41_PREFILL_ADAPT_UP_GIB")) |v| a.up_gib = v;
        if (envInt("TF_DSV41_PREFILL_ADAPT_MIN")) |v| a.min_rows = v;
        if (a.min_rows < 16) return error.BadSettings;
        s.adapt = a;
    }
};

fn envInt(name: [*:0]const u8) ?u64 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u64, t, 10) catch null;
}

fn envFloat(name: [*:0]const u8) ?f64 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    if (t.len == 0) return null;
    return std.fmt.parseFloat(f64, t) catch null;
}

/// A queued request as the planner sees it (`key`: the host's handle for it).
pub const Job = struct { key: usize, n: u64, max_new: u64, submitted: f64 };

/// The session store's side of an admission (sessions_gpu.zig on the GPU path; the host gate's store in the test).
pub const Store = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const Hit = struct { id: u32, pos: u64, ram: bool };

    pub const VTable = struct {
        /// The longest saved entry that strictly prefixes the job's prompt (null: none).
        find: *const fn (ctx: *anyopaque, job: usize) anyerror!?Hit,
        /// batch.py's pool decision for the job (`sess.admit`: the entries in `keep` never spilled, `extra` pages
        /// taken by the round's earlier admissions).
        pool: *const fn (ctx: *anyopaque, job: usize, hit: ?Hit, keep: []const u32, extra: u32, spills: *std.ArrayList(u32)) anyerror!sess.Plan,
    };
};

/// An admitted request's plan: its pages and the entry it resumes (the round's spills are in `Round.spills`: every
/// rank evicts them before the round's admissions run).
pub const Admission = struct { need: u32, hit: ?Store.Hit };

pub const Verdict = union(enum) {
    /// start it (its admission); the host begins the stream, then `started`
    admit: Admission,
    /// it and every request behind it stay queued this round (prod's FIFO `break`)
    wait,
    /// it stays queued; the next one may start (a long prefill in flight)
    skip,
    /// it cannot run on this server: the request fails with these words
    refuse: Refusal,
};

pub const Refusal = enum { too_large, memory };

/// One request in the slots: its prompt's progress (batch.py `Seq`).
pub const Seq = struct {
    key: usize,
    n: u64,
    submitted: f64,
    /// prompt tokens in the slot's state (prefill progress)
    done: u64,
    /// the tokens its saved prefix gave it at admission
    cached: u64 = 0,
    /// the replay point still to snapshot at
    save_at: ?u64,
    decoding: bool = false,
    long: bool = false,
    price: ?Price = null,
    /// its NVMe entry failed to restore: this round's plan stands, its pieces do not run, it starts over after it
    damaged: bool = false,
};

/// A prompt piece: rows [start, end) of `key`'s prompt; `save`: its snapshot right after; `run`: false for a damaged
/// entry's planned piece (prod plans it, the executor skips it).
pub const Piece = struct { key: usize, start: u64, end: u64, save: bool, run: bool = true };

pub const Counts = struct {
    long_waits: u64 = 0,
    mem_waits: u64 = 0,
    pool_waits: u64 = 0,
    mem_refused: u64 = 0,
    rows_down: u64 = 0,
    rows_up: u64 = 0,
    /// rounds at each adaptive step (rows_<rows>): (rows, count)
    rows: [8][2]u64 = @splat(.{ 0, 0 }),

    fn rowsAt(c: *Counts, r: u64) void {
        for (&c.rows) |*x| {
            if (x[0] == r) {
                x[1] += 1;
                return;
            }
            if (x[0] == 0) {
                x.* = .{ r, 1 };
                return;
            }
        }
    }
};

/// One admission pass's running state (batch.py `_admit`'s locals).
pub const Round = struct {
    longs: u64 = 0,
    owed: i64 = 0,
    /// entries resumed by this round's earlier admissions (never spilled for a later one)
    used: std.ArrayList(u32) = .empty,
    /// pages this round's earlier admissions take (`new_pages`: they run after the whole pass is decided)
    new_pages: u32 = 0,
    admitted: u32 = 0,
    /// the round's spills (plan.spills: every rank evicts them, in this order, before the admissions run)
    spills: std.ArrayList(u32) = .empty,
    one: std.ArrayList(u32) = .empty,

    pub fn deinit(r: *Round, gpa: std.mem.Allocator) void {
        r.used.deinit(gpa);
        r.spills.deinit(gpa);
        r.one.deinit(gpa);
    }
};

pub const Planner = struct {
    gpa: std.mem.Allocator,
    s: Settings,
    fair: Fairness,
    /// the requests in the slots, in admission order
    seqs: std.ArrayList(Seq) = .empty,
    /// the rows of the last prefill round (adaptive rows)
    rows_now: ?u64 = null,
    counts: Counts = .{},
    /// a request the floor holds back: since when (seconds)
    mem_since: std.AutoHashMapUnmanaged(usize, f64) = .empty,

    pub fn init(gpa: std.mem.Allocator, s: Settings) Planner {
        return .{ .gpa = gpa, .s = s, .fair = .{ .share = s.share } };
    }

    pub fn deinit(p: *Planner) void {
        p.seqs.deinit(p.gpa);
        p.mem_since.deinit(p.gpa);
    }

    fn find(p: *Planner, key: usize) ?*Seq {
        for (p.seqs.items) |*x| if (x.key == key) return x;
        return null;
    }

    pub fn seq(p: *Planner, key: usize) ?*Seq {
        return p.find(key);
    }

    /// The rows admission prices a prompt's transient at: the smallest adaptive step, else `rows`.
    fn priceRows(p: *const Planner) u64 {
        const a = p.s.adapt orelse return p.s.rows;
        var buf: [16]u64 = undefined;
        const st = a.steps(p.s.rows, &buf);
        return st[st.len - 1];
    }

    /// An admission pass starts (the host then asks `admit` for each queued request in its order while slots are free).
    pub fn beginAdmit(p: *Planner) Round {
        var r: Round = .{};
        for (p.seqs.items) |x| {
            if (x.long and !x.decoding) r.longs += 1;
            if (x.price) |pr| if (!x.decoding) {
                r.owed += pr.outstanding(x.done);
            };
        }
        return r;
    }

    /// batch.py `_admit` for one queued request: start it, keep it (and those behind) queued, look past it, or refuse
    /// it. `queued`: requests in the queue (the floor's log); `now`: seconds (the floor's refusal clock); `mem`: the
    /// memory reading when a floor is set.
    pub fn admit(p: *Planner, r: *Round, job: Job, store: ?Store, now: f64, mem: ?Meminfo) !Verdict {
        const hit: ?Store.Hit = if (store) |st| try st.vtable.find(st.ctx, job.key) else null;
        const cached: u64 = if (hit) |h| h.pos else 0;
        const long = job.n - cached > p.s.long_prompt;
        if (long and p.s.concurrent > 0 and r.longs >= p.s.concurrent) {
            p.counts.long_waits += 1;
            return .skip;
        }
        var price: ?Price = null;
        if (p.s.floor) |*f| if (mem) |mi| {
            const pr = prefillPrice(p.s.price, job.n, cached, p.priceRows());
            price = pr;
            if (!f.check(mi, r.owed + pr.total())) {
                p.counts.mem_waits += 1;
                if (try p.neverFits(job, pr, r, now)) return .{ .refuse = .memory };
                return .wait;
            }
            _ = p.mem_since.remove(job.key);
        };
        var pp: sess.Plan = .{ .need = 0, .want = 0 };
        if (store) |st| {
            r.one.clearRetainingCapacity();
            const at = r.used.items.len;
            if (hit) |h| try r.used.append(p.gpa, h.id);
            defer r.used.shrinkRetainingCapacity(at);
            pp = st.vtable.pool(st.ctx, job.key, hit, r.used.items, r.new_pages, &r.one) catch |e| switch (e) {
                error.RequestTooLarge => return .{ .refuse = .too_large },
                else => return e,
            };
            // running requests free pages as they end; one round's admissions spill once
            if (pp.wait or (r.one.items.len > 0 and r.spills.items.len > 0)) {
                p.counts.pool_waits += 1;
                return .wait;
            }
            try r.spills.appendSlice(p.gpa, r.one.items);
            r.new_pages = pp.want;
        }
        if (hit) |h| try r.used.append(p.gpa, h.id);
        r.longs += @intFromBool(long);
        if (price) |pr| r.owed += pr.total();
        r.admitted += 1;
        try p.seqs.append(p.gpa, .{ .key = job.key, .n = job.n, .submitted = job.submitted, .done = cached, .cached = cached, .save_at = p.savePoint(job.n, cached), .long = long, .price = price });
        return .{ .admit = .{ .need = pp.need, .hit = hit } };
    }

    fn savePoint(p: *const Planner, n: u64, cached: u64) ?u64 {
        if (!p.s.sessions) return null;
        const s = snapshotPoint(n);
        return if (s > cached and s >= p.s.session_min) s else null;
    }

    /// batch.py `_never_fits`: refused only when nothing is live (none admitted, none this round) and it has waited
    /// `refuse_s` while its price still exceeds the usable memory less the hard floor.
    fn neverFits(p: *Planner, job: Job, pr: Price, r: *const Round, now: f64) !bool {
        const f = &p.s.floor.?;
        _ = r;
        if (p.seqs.items.len > 0) {
            _ = p.mem_since.remove(job.key);
            return false;
        }
        const g = try p.mem_since.getOrPut(p.gpa, job.key);
        if (!g.found_existing) g.value_ptr.* = now;
        if (now - g.value_ptr.* < f.refuse_s or f.fitsIdle(pr.total())) return false;
        _ = p.mem_since.remove(job.key);
        p.counts.mem_refused += 1;
        return true;
    }

    /// A prompt whose client left: the grid point to keep its rows at for a retry, or null (nothing new, or saved).
    pub fn leaving(p: *const Planner, key: usize) ?u64 {
        for (p.seqs.items) |x| if (x.key == key) {
            if (x.decoding or x.damaged or x.save_at == null) return null;
            if (x.done % grid != 0 or x.done < p.s.session_min or x.done <= x.cached) return null;
            return x.done;
        };
        return null;
    }

    /// The admitted request's entry failed to restore (a damaged NVMe file): this round's plan stands, its pieces do
    /// not run; after the round it prefills from 0 (batch.py `_sample`).
    pub fn damaged(p: *Planner, key: usize) void {
        if (p.find(key)) |x| x.damaged = true;
    }

    /// batch.py `_pieces`: this round's pieces (`out`, in order) and the requests whose prompts are in (`finals`).
    /// `mem`: the memory reading when adaptive rows are on.
    pub fn plan(p: *Planner, mem: ?Meminfo, out: *std.ArrayList(Piece), finals: *std.ArrayList(usize)) !void {
        out.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        var decoding = false;
        for (p.seqs.items) |x| decoding = decoding or x.decoding;
        const allow = p.fair.allow(decoding);
        var waiting: std.ArrayList(*Seq) = .empty;
        defer waiting.deinit(p.gpa);
        for (p.seqs.items) |*x| if (!x.decoding and x.done < x.n - 1) try waiting.append(p.gpa, x);
        var budget: u64 = if (waiting.items.len > 0) try p.roundRows(waiting.items, mem) else p.s.rows;
        std.mem.sort(*Seq, waiting.items, {}, struct {
            fn less(_: void, a: *Seq, b: *Seq) bool {
                const la = a.n - 1 - a.done;
                const lb = b.n - 1 - b.done;
                return la < lb or (la == lb and a.submitted < b.submitted);
            }
        }.less);
        for (waiting.items) |x| {
            const left = x.n - 1 - x.done;
            if (budget == 0 or (left > p.s.short and !allow)) continue;
            const target = if (x.save_at) |s| (if (x.done < s) s else x.n - 1) else x.n - 1;
            const end = pieceEnd(x.done, target, budget, grid);
            const save = x.save_at != null and end == x.save_at.?;
            try out.append(p.gpa, .{ .key = x.key, .start = x.done, .end = end, .save = save, .run = !x.damaged });
            // a piece is at least one grid step, so it can cost more than the budget left: batch.py's budget goes
            // negative and the next `budget <= 0` ends the round; saturating to 0 does the same
            budget -|= end - x.done;
            x.done = end;
            if (save) x.save_at = null;
        }
        for (p.seqs.items) |*x| if (!x.decoding and x.done == x.n - 1) {
            if (!x.damaged) try finals.append(p.gpa, x.key);
            x.decoding = true;
        };
    }

    /// batch.py `_round_rows` (G19): the full rows while the usable memory covers the round's priced transient with
    /// `adapt.gib` to spare, else a smaller step.
    fn roundRows(p: *Planner, waiting: []const *Seq, mem: ?Meminfo) !u64 {
        const a = p.s.adapt orelse return p.s.rows;
        const f = p.s.floor orelse return p.s.rows;
        const mi = mem orelse return p.s.rows;
        var left: std.ArrayList([2]u64) = .empty;
        defer left.deinit(p.gpa);
        for (waiting) |x| try left.append(p.gpa, .{ x.n, x.done });
        const rows = a.choose(p.s.price, p.s.rows, f.usable(mi), left.items, p.rows_now);
        if (p.rows_now) |cur| if (rows != cur) {
            if (rows < cur) p.counts.rows_down += 1 else p.counts.rows_up += 1;
            std.log.scoped(.dsv41).info("prefill rows {d} -> {d} (usable {d:.2} GiB, the round's transient at {d} rows {d:.2} GiB, keep {d})", .{ cur, rows, @as(f64, @floatFromInt(f.usable(mi))) / GiB, rows, @as(f64, @floatFromInt(roundNeed(p.s.price, rows, left.items))) / GiB, a.gib });
        };
        p.counts.rowsAt(rows);
        p.rows_now = rows;
        return rows;
    }

    /// A request left the slots (finished, cancelled, failed).
    pub fn ended(p: *Planner, key: usize) void {
        _ = p.mem_since.remove(key);
        for (p.seqs.items, 0..) |x, i| if (x.key == key) {
            _ = p.seqs.orderedRemove(i);
            return;
        };
    }

    /// A queued request left the queue without starting (cancelled, refused).
    pub fn dropped(p: *Planner, key: usize) void {
        _ = p.mem_since.remove(key);
    }

    /// After the round (batch.py `_sample`'s damaged-entry reset, then the fair share): the pieces' and the decode's
    /// seconds. Ended requests are gone first (`ended`).
    pub fn after(p: *Planner, piece_s: f64, decode_s: f64) void {
        for (p.seqs.items) |*x| if (x.damaged) {
            x.damaged = false;
            x.done = 0;
            x.decoding = false;
            x.long = x.n > p.s.long_prompt;
            x.save_at = if (p.s.sessions) snapshotPoint(x.n) else null;
        };
        var prefilling = false;
        for (p.seqs.items) |x| prefilling = prefilling or !x.decoding;
        p.fair.after(piece_s, decode_s, prefilling);
    }
};

test "piece_end, snapshot_point and the fair share as plan.py / sessions.py / batchplan.py" {
    try std.testing.expectEqual(@as(u64, 300), pieceEnd(0, 300, 2048, 16));
    try std.testing.expectEqual(@as(u64, 2048), pieceEnd(0, 9000, 2048, 16));
    try std.testing.expectEqual(@as(u64, 2032), pieceEnd(5, 9000, 2040, 16));
    try std.testing.expectEqual(@as(u64, 16), pieceEnd(3, 9000, 4, 16)); // at least one grid step
    try std.testing.expectEqual(@as(u64, 10), pieceEnd(3, 10, 4, 16));
    try std.testing.expectEqual(@as(u64, 9488), snapshotPoint(9500));
    try std.testing.expectEqual(@as(u64, 0), snapshotPoint(1));
    var f: Fairness = .{};
    try std.testing.expect(f.allow(true));
    f.after(0.25, 0.0625, true);
    try std.testing.expect(!f.allow(true) and f.allow(false));
    f.after(0, 0.0625, true);
    f.after(0, 0.0625, true);
    f.after(0, 0.0625, true);
    try std.testing.expect(f.allow(true));
    var buf: [16]u64 = undefined;
    try std.testing.expectEqualSlices(u64, &.{ 2048, 1024, 512 }, (Adapt{}).steps(2048, &buf));
}

test "a round whose last piece rounds up past the budget left ends the round (batch.py: budget <= 0)" {
    // Prod 2026-10-10: a prompt mid-prefill and two arrivals in one round. The first piece left 4 rows; the next,
    // at row 3, rounds up to the grid (13 rows): batch.py's budget goes negative and the round takes no more
    // pieces, where an unsigned budget overflowed and the planner panicked.
    var p = Planner.init(std.testing.allocator, .{});
    defer p.deinit();
    try p.seqs.append(p.gpa, .{ .key = 1, .n = 2045, .submitted = 1, .done = 0, .save_at = null });
    try p.seqs.append(p.gpa, .{ .key = 2, .n = 9000, .submitted = 2, .done = 3, .save_at = null });
    try p.seqs.append(p.gpa, .{ .key = 3, .n = 9500, .submitted = 3, .done = 0, .save_at = null });
    var out: std.ArrayList(Piece) = .empty;
    defer out.deinit(p.gpa);
    var finals: std.ArrayList(usize) = .empty;
    defer finals.deinit(p.gpa);
    try p.plan(null, &out, &finals);
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqual(Piece{ .key = 1, .start = 0, .end = 2044, .save = false }, out.items[0]);
    try std.testing.expectEqual(Piece{ .key = 2, .start = 3, .end = 16, .save = false }, out.items[1]);
    try std.testing.expectEqualSlices(usize, &.{1}, finals.items);
}

test "a prompt that leaves mid-prefill is kept at its last grid point, if that is new and before its replay point" {
    var p = Planner.init(std.testing.allocator, .{});
    defer p.deinit();
    try p.seqs.append(p.gpa, .{ .key = 1, .n = 9000, .submitted = 0, .done = 4096, .cached = 0, .save_at = snapshotPoint(9000) });
    try p.seqs.append(p.gpa, .{ .key = 2, .n = 9000, .submitted = 0, .done = 4096, .cached = 4096, .save_at = snapshotPoint(9000) });
    try p.seqs.append(p.gpa, .{ .key = 3, .n = 9000, .submitted = 0, .done = 4100, .cached = 0, .save_at = snapshotPoint(9000) });
    try p.seqs.append(p.gpa, .{ .key = 4, .n = 9000, .submitted = 0, .done = 8992, .cached = 0, .save_at = null });
    try p.seqs.append(p.gpa, .{ .key = 5, .n = 9000, .submitted = 0, .done = 48, .cached = 0, .save_at = snapshotPoint(9000) });
    try p.seqs.append(p.gpa, .{ .key = 6, .n = 9000, .submitted = 0, .done = 4096, .cached = 0, .save_at = snapshotPoint(9000), .decoding = true });
    try std.testing.expectEqual(@as(?u64, 4096), p.leaving(1));
    try std.testing.expectEqual(@as(?u64, null), p.leaving(2)); // nothing past what it resumed from
    try std.testing.expectEqual(@as(?u64, null), p.leaving(3)); // off the grid
    try std.testing.expectEqual(@as(?u64, null), p.leaving(4)); // its replay point is saved already
    try std.testing.expectEqual(@as(?u64, null), p.leaving(5)); // under session_min
    try std.testing.expectEqual(@as(?u64, null), p.leaving(6)); // its prompt is in
    try std.testing.expectEqual(@as(?u64, null), p.leaving(99));
}

test {
    _ = @import("sched_test.zig");
    _ = @import("sched_crash_test.zig");
}
