"""Two bench_http.py --times cells side by side: a 1-stream rate split into tokens a chunk, typical and slow gaps."""
import argparse
import json
import math
import statistics


def cell(path):
    j = json.load(open(path))
    c = j["cells"][0] if "cells" in j else j
    times = c["times_ms"][0]
    toks = c["tokens"][0]
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"path": path.split("/")[-1], "rate": c["mean_tok_s"], "ttft": c["ttft_s"][0] * 1e3, "tokens": toks,
            "tpr": c.get("mean_tpr"), "times": times, "gaps": gaps}


def stats(c, slow):
    g = c["gaps"]
    med = statistics.median(g)
    excess = sum(x - med for x in g if x > slow * med)
    q = sorted(g)
    return {
        "chunks": len(c["times"]),
        "tok/chunk": (c["tokens"] - 1) / max(1, len(g)),
        "span ms": sum(g),
        "median gap": med,
        "mean gap": statistics.fmean(g),
        "p90 gap": q[int(0.9 * (len(q) - 1))],
        "slow gaps": sum(1 for x in g if x > slow * med),
        "slow excess ms": excess,
        "first gap": g[0],
        "last gap": g[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--slow", type=float, default=1.25)
    o = ap.parse_args()
    A, B = cell(o.a), cell(o.b)
    sa, sb = stats(A, o.slow), stats(B, o.slow)
    print(f"{'':16s} {A['path'][:22]:>22s} {B['path'][:22]:>22s}")
    for k in ("rate", "ttft", "tokens", "tpr"):
        print(f"{k:16s} {A[k] or 0:22.3f} {B[k] or 0:22.3f}")
    for k in sa:
        print(f"{k:16s} {sa[k]:22.3f} {sb[k]:22.3f}")
    # log-rate decomposition: ln(rA / rB) = ln(tpcA / tpcB) - ln(meanA / meanB), the mean gap split at the median
    tpc = math.log(sa["tok/chunk"] / sb["tok/chunk"])
    typ = -math.log(sa["median gap"] / sb["median gap"])
    total = math.log((A["tokens"] - 1) / sa["span ms"] / ((B["tokens"] - 1) / sb["span ms"]))
    rest = total - tpc - typ
    pct = lambda x: 100 * (math.exp(x) - 1)
    print(f"\nA vs B decode rate {pct(total):+.2f} % =")
    print(f"  tokens a chunk (drafting)        {pct(tpc):+.2f} %")
    print(f"  typical round (median gap)       {pct(typ):+.2f} %")
    print(f"  slow rounds and gap spread       {pct(rest):+.2f} %   (A slow excess {sa['slow excess ms']:.1f} ms, B {sb['slow excess ms']:.1f} ms)")
    print(f"TTFT (outside the rate): A {A['ttft']:.0f} ms, B {B['ttft']:.0f} ms")


if __name__ == "__main__":
    main()
