"""Per-round kernel time of two nsys SQLite exports side by side over the middle 60 % (optionally split by grid)."""
import argparse
import collections
import sqlite3


def load(path, marker, grid):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    ev = []
    for s, e, k, x, y, z in c.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        n = names.get(k, "?")
        ev.append((s, e, f"{n} ({x},{y},{z})" if grid else n, n))
    for s, e, k, b in c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append((s, e, f"memcpy{k} {b} B" if grid else f"memcpy{k}", "memcpy"))
    ev.sort()
    t0, t1 = ev[0][0], ev[-1][1]
    lo, hi = t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0)
    marks = [s for s, _, _, base in ev if base == marker and lo <= s <= hi]
    if len(marks) < 2:
        raise SystemExit(f"{path}: fewer than 2 '{marker}' launches in the middle of the capture")
    a, b = marks[0], marks[-1]
    rounds = len(marks) - 1
    tot, cnt = collections.Counter(), collections.Counter()
    for s, e, key, _ in ev:
        if a <= s < b:
            tot[key] += e - s
            cnt[key] += 1
    return {k: v / rounds / 1e6 for k, v in tot.items()}, {k: v / rounds for k, v in cnt.items()}, (b - a) / rounds / 1e6, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--marker", default="_chain")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--names", help="only kernels whose name starts with one of these (comma separated)")
    o = ap.parse_args()
    A = load(o.a, o.marker, o.grid)
    B = load(o.b, o.marker, o.grid)
    for tag, x, p in (("A", A, o.a), ("B", B, o.b)):
        print(f"{tag} {p.split('/')[-1]}: {x[3]} rounds, {x[2]:.2f} ms a round, kernel sum {sum(x[0].values()):.2f} ms")
    keys = set(A[0]) | set(B[0])
    if o.names:
        pre = tuple(o.names.split(","))
        keys = {k for k in keys if k.startswith(pre)}
    keys = sorted(keys, key=lambda k: -abs(A[0].get(k, 0) - B[0].get(k, 0)))
    print(f"{'kernel (by |A - B|)':52s} {'A ms':>8s} {'A n':>6s} {'B ms':>8s} {'B n':>6s} {'A - B':>8s}")
    for k in keys[: o.top]:
        da, db = A[0].get(k, 0), B[0].get(k, 0)
        print(f"{k[:52]:52s} {da:8.3f} {A[1].get(k, 0):6.1f} {db:8.3f} {B[1].get(k, 0):6.1f} {da - db:+8.3f}")


if __name__ == "__main__":
    main()
