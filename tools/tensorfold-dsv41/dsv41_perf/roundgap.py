"""GPU-idle gaps inside a decode round from an nsys SQLite export, split by the kernels bounding each gap."""
import collections
import sqlite3
import statistics
import sys

f = sys.argv[1]
marker = sys.argv[2] if len(sys.argv) > 2 else "pick_pack_kernel"
min_gap = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 10e3
c = sqlite3.connect(f)
names = dict(c.execute("select id, value from StringIds"))
ev = []
for s, e, k, g in c.execute("select start, end, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"):
    ev.append((s, e, names.get(k, "?"), g is not None))
for tbl, lab in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    for s, e, kind in c.execute(f"select start, end, copyKind from {tbl}" if lab == "memcpy" else
                                f"select start, end, 0 from {tbl}"):
        ev.append((s, e, f"{lab}{kind}", False))
ev.sort()
api = c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
api = [(s, e, names.get(n, "?")) for s, e, n in api]

t0, t1 = ev[0][0], ev[-1][1]
lo, hi = t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0)
marks = [s for s, e, n, g in ev if n == marker and lo <= s <= hi]
print(f"{f.split('/')[-1]}: {len(marks) - 1} rounds (marker {marker})")

spans, busys, gap_cls, gap_api = [], [], collections.Counter(), collections.Counter()
per_round_gaps = []
j = 0
for a, b in zip(marks, marks[1:]):
    while j < len(ev) and ev[j][1] < a:
        j += 1
    k = j
    cur_end, busy, prev = a, 0, None
    gaps = []
    while k < len(ev) and ev[k][0] < b:
        s, e, n, g = ev[k]
        s2, e2 = max(s, a), min(e, b)
        if s2 > cur_end:
            gaps.append((cur_end, s2, prev, (n, g)))
        if e2 > cur_end:
            busy += e2 - max(s2, cur_end)
            cur_end = e2
            prev = (n, g)
        k += 1
    if b > cur_end:
        gaps.append((cur_end, b, prev, None))
    spans.append(b - a)
    busys.append(busy)
    big = 0
    for gs, ge, p, nx in gaps:
        d = ge - gs
        if d < min_gap:
            gap_cls["(gaps < min)"] += d
            continue
        big += d
        key = f"{(p[0][:26] + ('[g]' if p[1] else '')) if p else '-'} -> {(nx[0][:26] + ('[g]' if nx[1] else '')) if nx else '-'}"
        gap_cls[key] += d
        # host calls overlapping the gap (what the host was doing)
        for s, e, n in api:
            if e < gs:
                continue
            if s > ge:
                break
            ov = min(e, ge) - max(s, gs)
            if ov > 0:
                gap_api[n] += ov
    per_round_gaps.append(big)

n = len(spans)
ms = lambda v: v / 1e6
print(f"round span   mean {ms(statistics.mean(spans)):.2f} ms  median {ms(statistics.median(spans)):.2f}")
print(f"GPU busy     mean {ms(statistics.mean(busys)):.2f} ms")
idle = [s - b for s, b in zip(spans, busys)]
print(f"GPU idle     mean {ms(statistics.mean(idle)):.2f} ms  median {ms(statistics.median(idle)):.2f}"
      f"  ({100 * sum(idle) / sum(spans):.1f} % of the round)")
print(f"  in gaps >= {min_gap / 1e3:.0f} us: mean {ms(statistics.mean(per_round_gaps)):.2f} ms a round")
print("idle by bounding kernels (ms a round):")
for k, v in gap_cls.most_common(25):
    print(f"  {ms(v) / n:7.3f}  {k}")
print("host API overlapping the big gaps (ms a round):")
for k, v in gap_api.most_common(12):
    print(f"  {ms(v) / n:7.3f}  {k}")

# Pass -> window: from the pass's largest DtoH copy to the window's first graphed kernel, plus its host times.
copies = c.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 order by start").fetchall()
big = max((b for s, e, b in copies), default=0)
if big >= 65536:
    graphed = [s for s, e, nm, g in ev if g]
    launches = [(s, e) for s, e, nm in api if nm.startswith("cuGraphLaunch")]
    trans, lat = [], []
    gi = li = 0
    for s, e, b in copies:
        if b != big or not (lo <= s <= hi):
            continue
        while gi < len(graphed) and graphed[gi] < e:
            gi += 1
        if gi == len(graphed):
            break
        trans.append(graphed[gi] - e)
        while li < len(launches) and launches[li][1] < e:
            li += 1
        here = [x for x in launches[li:li + 4] if x[0] < graphed[gi]]
        if here:
            lat.append(sum(x[1] - x[0] for x in here))
    if trans:
        print(f"pass end ({big} B copy) -> next graphed kernel: median {ms(statistics.median(trans)):.3f} ms over {len(trans)} rounds")
    if lat:
        print(f"  the window's cuGraphLaunch host time before its first kernel (all parts): median {ms(statistics.median(lat)):.3f} ms")
