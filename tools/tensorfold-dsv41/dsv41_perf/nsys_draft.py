#!/usr/bin/env python3
"""The drafter's share of a Zig row-mode decode round from an nsys SQLite export, per-phase microsecond stats."""
import sqlite3
import statistics as st
import sys


def idle(rows):
    t, prev = 0, rows[0][1]
    for s, e, _ in rows[1:]:
        if s > prev:
            t += s - prev
        prev = max(prev, e)
    return t / 1e3


def main(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    ks = [(s, e, names[k]) for s, e, k in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    picks = [i for i, r in enumerate(ks) if r[2] == "pick_pack_kernel"]
    out = {}

    def add(k, v):
        out.setdefault(k, []).append(v)

    for a, b in zip(picks, picks[1:]):
        R = ks[a:b + 1]
        nm = [r[2] for r in R]
        if "ds_stage_rows_kernel" not in nm or "ds_cands_kernel" not in nm or "gm2_kernel" in nm or "pfd_kernel" in nm:
            continue
        ist = nm.index("ds_stage_rows_kernel")
        ic = len(nm) - 1 - nm[::-1].index("ds_cands_kernel")
        if "ds_accept_rows_kernel" in nm:
            ia = nm.index("ds_accept_rows_kernel")
            add("ingest", (R[ist][0] - R[ia][0]) / 1e3)
            add("ingest_idle", idle(R[ia:ist + 1]))
        add("pass", (R[ic][1] - R[ist][0]) / 1e3)
        add("pass_roce", sum(e - s for s, e, n in R[ist:ic] if n == "tp_roce") / 1e3)
        heads = [r for r in R[ist:ic] if r[2] == "linear_kernel"]
        if heads:
            add("head", (heads[-1][1] - heads[-1][0]) / 1e3)
        j = ic + 1
        while j < len(R) and R[j][2] == "tp_roce":
            j += 1
        add("after_pass", (R[j][0] - R[j - 1][1]) / 1e3)
        k = j
        while k < len(R) and R[k][2] == "carry_kernel":
            k += 1
        add("before_window", (R[k][0] - R[k - 1][1]) / 1e3)
        W = R[k:]
        add("window", (W[-1][1] - W[0][0]) / 1e3)
        add("roce", sum(e - s for s, e, n in W if n == "tp_roce") / 1e3)
        add("nroce", sum(1 for r in W if r[2] == "tp_roce"))
        add("window_idle", idle(W))
        lds = [i for i, r in enumerate(R) if r[2] == "ld_kernel"]
        if len(lds) >= 4:
            add("tail", (R[-1][0] - R[lds[-4]][0]) / 1e3)
        add("round", (R[-1][0] - R[0][0]) / 1e3)
    print(f"{len(out.get('round', []))} decode rounds (us)")
    for k, v in out.items():
        print(f"{k:14} median {st.median(v):10.1f} mean {st.mean(v):10.1f} min {min(v):10.1f} max {max(v):10.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
