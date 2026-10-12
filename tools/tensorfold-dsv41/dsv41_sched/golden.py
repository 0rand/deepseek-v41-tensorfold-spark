#!/usr/bin/env python3
"""Prefill scheduler golden: the reference batcher plans rounds on CPU; every round's decisions go to JSON for Zig."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types

GiB = 1 << 30


def _stub_triton() -> None:
    """Modules the batcher's imports reach import triton; on a host without it, stand-ins that only decorate."""

    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass

    class _Any(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _anything

    def _anything(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _anything

    for name in ("triton", "triton.language", "triton.language.extra", "triton.language.extra.cuda",
                 "triton.language.extra.libdevice", "triton.runtime", "triton.compiler", "triton.backends",
                 "triton.backends.compiler", "triton.tools", "triton.tools.tensor_descriptor"):
        sys.modules[name] = _Any(name)
    sys.modules["triton"].language = sys.modules["triton.language"]


# the scenario's knobs (the Zig test reads them from the JSON)
SETTINGS = {
    "rows": 2048,            # TF_DSV41_PREFILL_ROWS
    "short": 512,            # TF_DSV41_BATCH_SHORT
    "share": 0.5,            # TF_DSV41_PREFILL_SHARE
    "long_prompt": 6000,     # TF_DSV41_LONG_PROMPT (a smaller scenario than served, the same rule)
    "concurrent": 1,         # TF_DSV41_PREFILL_CONCURRENT
    "session_min": 64,
    "adapt_gib": 4.5,        # TF_DSV41_PREFILL_ADAPT_GIB
    "adapt_up_gib": 0.5,
    "adapt_min": 512,
    "floor_gib": 5.0,
    "hard_gib": 4.0,
    "refuse_s": 30.0,
    "cache_keep_gib": 2.0,   # GLM53_TF_ADMIT_CACHE_KEEP_GB
    "index_budget_mib": 256,
}

PAGES, SLOTS, CAPACITY, DISK_MIN = 72, 4, 16384, 1024
BOUNDED_BYTES = 1 << 20      # a snapshot's RAM measure (any fixed size: the tier trims at the same saves)
RAM_BYTES = 3 * BOUNDED_BYTES + (BOUNDED_BYTES >> 1)
WINDOW_S = 0.0625            # a decode round's seconds
TOKEN = 7                    # every reply token


def chat_ids(chat: int, n: int) -> list[int]:
    return [chat * 100_000 + i for i in range(n)]


def meminfo(free_gib: float) -> dict:
    """/proc/meminfo with ``free_gib`` free and a 3 GiB page cache of which 1 GiB is mapped (credit = 3 - 2 kept)."""

    return {"MemFree": int(free_gib * GiB), "MemAvailable": int((free_gib + 3) * GiB), "Dirty": 0, "Writeback": 0,
            "Mapped": GiB}


# arrivals: round -> [(id, chat, n, max_tokens, background)]; the same chat's later turns resume its snapshot
ARRIVALS = {
    0: [(1, 1, 1500, 24, False), (2, 2, 300, 40, False)],
    1: [(3, 3, 9000, 16, False)],                     # long: one long prefill at a time
    2: [(4, 4, 7000, 12, False), (5, 5, 200, 20, True)],   # long behind long; a short background request
    4: [(6, 1, 2100, 30, False)],                     # chat 1, turn 2: resumes from RAM
    6: [(7, 6, 5000, 3000, False)],                   # a large quota: the pool makes it wait
    9: [(8, 2, 1200, 10, False)],                     # chat 2's turn 2 (dropped from RAM by then? the tier decides)
    14: [(9, 3, 9500, 10, False)],                    # chat 3's turn 2: its snapshot is on NVMe (damaged below)
    16: [(10, 7, 3000, 10, False), (11, 8, 600, 10, False)],
}
CANCEL = {11: [4], 105: [7]}           # round -> ids cancelled (the HTTP side hung up)
DAMAGE = {45}                # rounds before which every NVMe file is damaged (a bad sector, a torn write)
MEMORY = {0: 24.0, 5: 4.0, 7: 3.2, 8: 2.0, 11: 24.0, 22: 2.0, 25: 24.0}   # round -> MemFree GiB from that round on
ROUNDS = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    _stub_triton()
    os.environ["TF_DSV41_PREFILL_ADAPT_GIB"] = str(SETTINGS["adapt_gib"])
    os.environ["TF_DSV41_LOOKUP"] = "0"
    os.environ["TF_DSV41_STALL_S"] = "0"
    os.environ["TF_DSV41_HOST_TRIM_BG"] = "0"
    os.environ["TF_DSV41_HOST_TRIM_ROWS"] = "0"
    import numpy as np

    from tensorfold.families.deepseek_v41.cuda import batch as B
    from tensorfold.families.deepseek_v41.cuda import memory as ME
    from tensorfold.families.deepseek_v41.cuda import pool as PO
    from tensorfold.families.deepseek_v41.cuda import sessdisk as SD
    from tensorfold.families.deepseek_v41.cuda import sessions as SE
    from tensorfold.families.deepseek_v41.cuda import topology as TO
    from tensorfold.families.deepseek_v41.cuda.protocol import Bounded

    clock = [1000.0]
    B.time = types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0])

    class Fwd:
        """What check_forward asks for; the batcher plans, nothing here runs."""

        vocab = 129280
        pchunk = 2048

        def reset(self, slot): ...
        def prefill(self, pieces, *, mode): ...
        def finish_prompt(self, slot, end, tail, *, mode): ...
        def window(self, windows, *, count, masks=None): ...
        def commit(self, slot, acc): ...
        def snapshot(self, slot): ...
        def restore(self, slot, snap): ...

    cfg = json.load(open(a.config))
    topo = TO.build(cfg)
    TAG = SE.Tag(prefill="fast", grid=1, ced="replay").code()
    page = PO.PAGE
    tmp = tempfile.mkdtemp(prefix="sched-golden-")
    pool = PO.Pool(topo, PAGES * page, page)
    disk = SD.DiskTier(tmp, 0, SE.compat_extra(SE.Tag(), topo), budget_gib=64.0, min_tokens=DISK_MIN, direct=False)
    store = SE.Store(pool, ram_bytes=RAM_BYTES, disk=disk)
    mem = [meminfo(MEMORY[0])]
    S = SETTINGS
    floor = ME.Floor(S["floor_gib"], S["hard_gib"], meminfo=lambda: mem[0], cached=lambda: 0, free_gib=0.0,
                     refuse_s=S["refuse_s"])
    b = B.Batcher(Fwd(), pool, n_slots=SLOTS, capacity=CAPACITY, store=store, floor=floor, mode="replay", tag=TAG,
                  prefill_rows=S["rows"], depth=0, start=False, long_prompt=S["long_prompt"],
                  prefill_concurrent=S["concurrent"])
    assert b.adapt is not None and b.adapt.gib == S["adapt_gib"]
    b.short = S["short"]
    ex = b.ex
    jobs: dict[int, B.Job] = {}
    rows_of: dict[int, int] = {}                        # job id -> the slot's rows written (emulated)
    out = {"python": "8474f31", "settings": S, "tag": TAG, "page": page, "pages": PAGES, "slots": SLOTS,
           "capacity": CAPACITY, "slack": pool.slack, "ram_bytes": RAM_BYTES, "disk_min": DISK_MIN,
           "bounded_bytes": BOUNDED_BYTES, "window_s": WINDOW_S, "jobs": [], "rounds": []}

    def jid(slot: int) -> int:
        return b.seqs[slot].job.id

    def save(slot: int) -> bool:
        """Save the slot's prompt: an entry for the slot's history (a fixed-size bounded state)."""

        h = ex.hist[slot]
        if len(h) < ex.session_min:
            return False
        bd = Bounded(len(h), {"x": np.zeros(BOUNDED_BYTES, dtype=np.uint8)}, {})
        store.save(ex.slots[slot], bd, h, TAG, "prompt")
        return True

    for r in range(ROUNDS):
        if r in MEMORY:
            mem[0] = meminfo(MEMORY[r])
        if r in DAMAGE:
            for key in list(disk.keys()):
                path = disk.path(key)
                with open(path, "r+b") as fh:
                    size = os.path.getsize(path)
                    fh.seek(size - 64)
                    fh.write(b"\xff" * 32)
        arrived = []
        for (i, chat, n, mt, bg) in ARRIVALS.get(r, []):
            j = B.Job(chat_ids(chat, n), mt, background=bg)
            j.id, j.submitted = i, clock[0]
            jobs[i] = j
            b.submit(j)
            arrived.append(i)
            out["jobs"].append({"id": i, "chat": chat, "n": n, "max_tokens": mt, "background": bg,
                                "submitted": clock[0]})
        cancelled = []
        for i in CANCEL.get(r, []):
            if not jobs[i].cancel:
                b.cancel(jobs[i])
                cancelled.append(i)
        before = dict(b.counts)
        live_before = {s.job.id for s in b.seqs if s is not None}
        plan = b._plan()
        # rounds.Executor.run without a forward
        res = types.SimpleNamespace(skipped=set(), errors={}, cand=None, rows=[], piece_s=0.0, window_s=0.0)
        for slot, save_flag, _ in plan.finishes:
            ex.slots[slot].release()
            ex.hist[slot] = []
            ex.prompts[slot] = None
        for key in plan.spills:
            store.evict(key)
        admits = []
        for ad in plan.admits:
            sl = ex.slots[ad.slot]
            assert not sl.pages
            sl.reserve(ad.quota)
            ex.prompts[ad.slot] = list(ad.prompt)
            cached = 0
            if ad.tier:
                try:
                    snap = store.restore(ad.tier, ad.key, sl)
                    assert snap.pos == ad.cached
                    cached = ad.cached
                except (ValueError, KeyError):
                    sl.truncate(0)
                    res.skipped.add(ad.slot)
            ex.hist[ad.slot] = list(ad.prompt[:cached])
            admits.append({"id": ad.job, "tier": ad.tier or "none", "cached": ad.cached, "need": ad.quota,
                           "damaged": ad.slot in res.skipped})
        pieces, rows = [], 0
        for slot, start, end in plan.pieces:
            if slot in res.skipped:
                continue
            ex.slots[slot].ensure(end)
            ex.hist[slot].extend(ex.prompts[slot][start:end])
            pieces.append([jid(slot), start, end])
            rows += end - start
        res.piece_s = rows / 8192.0
        saves = []
        for slot in plan.saves:
            if slot not in res.skipped and save(slot):
                saves.append(jid(slot))
        finals = [jid(slot) for slot in plan.finals if slot not in res.skipped]
        windows = []
        for w in plan.windows:
            if w.slot in res.skipped:
                continue
            ex.slots[w.slot].ensure(w.start + 1)
            windows.append(w.slot)
        window_ids = sorted(jid(s) for s in windows)
        res.window_s = WINDOW_S if windows else 0.0
        planned_pieces = [[jid(s), st, en] for s, st, en in plan.pieces]
        skipped = sorted(jid(s) for s in res.skipped)
        # the sampler's step: the damaged entries' reset, then each window's one token (no drafts)
        b._sample(plan, res)
        ended = []
        for slot in windows:
            s = b.seqs[slot]
            if s is None:
                continue
            b.commits.append((slot, 0, TOKEN))
            s.pos, s.pending = s.pos + 1, TOKEN
            s.out.append(TOKEN)
            if len(s.out) >= s.job.max_tokens:
                ended.append(s.job.id)
                b._end(s, None, cancelled=False)
        b.fair.after(res.piece_s, res.window_s, any(s is not None and not s.decoding for s in b.seqs))
        counts = {k: v - before.get(k, 0) for k, v in b.counts.items() if v != before.get(k, 0)}
        rec = {"round": r, "clock": clock[0], "free_gib": MEMORY[max(k for k in MEMORY if k <= r)],
               "arrive": arrived, "cancel": cancelled, "damage": r in DAMAGE, "admits": admits,
               "pieces": planned_pieces, "ran": pieces, "skipped": skipped, "saves": saves, "finals": finals,
               "windows": window_ids,
               "ended": ended, "rows": b.rows_now, "piece_s": res.piece_s, "window_s": res.window_s,
               "counts": counts, "queue": [j.id for j in b.queue],
               "ram": sorted(len(e.ids) for e in store.entries.values() if e.in_ram),
               "disk": sorted(disk.length(k) for k in disk.keys()), "available": pool.available(),
               "debt": b.fair.debt, "fair_rounds": b.fair.rounds}
        out["rounds"].append(rec)
        clock[0] += 1.0
        if not live_before and not b.queue and not any(s is not None for s in b.seqs) and r > max(ARRIVALS):
            break
    out["counts"] = dict(b.counts)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(out['rounds'])} rounds, counts {dict(b.counts)}, store {store.stats} -> {a.out}")


if __name__ == "__main__":
    main()
