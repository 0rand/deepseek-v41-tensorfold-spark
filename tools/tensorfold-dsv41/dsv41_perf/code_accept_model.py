#!/usr/bin/env python3
"""Reproduce a conditional draft-depth sensitivity model from bench artifacts (offline model, not a benchmark)."""

import argparse
import json
import math
import re
from pathlib import Path


ORDER = [f"{work}-t{temp}-s{streams}" for work in ("code", "prose")
         for temp in ("0", "0.7") for streams in (1, 4)]


def expected(q, depth):
    return sum(q ** j for j in range(depth + 1))


def fit_q(tokens, depth):
    """Fit constant conditional q using linear interpolation at fractional mean depth."""
    lo, hi = 0.0, 1.0
    whole = math.floor(depth)
    for _ in range(80):
        q = (lo + hi) / 2
        estimate = expected(q, whole) + (depth - whole) * q ** (whole + 1)
        if estimate < tokens:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run", required=True, help="the bench run's directory name under --results")
    args = parser.parse_args()
    run = args.results / args.run
    bench = run / "bench-zig"
    cells = {name: json.loads((bench / "cells" / f"{name}.json").read_text())["cells"]
             for name in ORDER}
    log_path = run / "logs" / "r0-bench0.log"
    records = []
    for line in log_path.read_text().splitlines():
        match = re.search(r"prompt=(\d+).*tokens=384 .*rounds=(\d+) accepted=(\d+)/(\d+)", line)
        if match:
            records.append(tuple(map(int, match.groups())))

    # Completion order can differ within a concurrent cell: match each repetition's unordered round-derived TPRs.
    groups = [(name, cell) for name in ORDER for cell in cells[name]]
    count = sum(cell["streams"] for _, cell in groups)
    candidates = []
    for start in range(len(records) - count + 1):
        pos, error = start, 0.0
        for _, cell in groups:
            size = cell["streams"]
            got = sorted(1 + item[2] / item[1] for item in records[pos:pos + size])
            want = sorted(cell["tpr"])
            error = max(error, *(abs(a - b) for a, b in zip(got, want)))
            pos += size
        candidates.append((error, start))
    if not candidates:
        raise ValueError("Request log does not contain the full speed-cell run")
    error, start = min(candidates)
    if error > 0.002:
        raise ValueError(f"Cannot match speed cells to request log: max TPR error {error}")
    matched = {}
    pos = start
    for name, cell in groups:
        size = cell["streams"]
        matched.setdefault(name, []).extend(records[pos:pos + size])
        pos += size
    mix_cell = json.loads((bench / "mix-short.json").read_text())["cells"][0]
    mix_records = records[-4:]
    if max(abs(a - b) for a, b in zip(sorted(1 + r[2] / r[1] for r in mix_records),
                                     sorted(mix_cell["tpr"]))) > 0.002:
        raise ValueError("Final request records do not match mix-short cell")
    cells["mix4"] = [mix_cell]
    matched["mix4"] = mix_records

    assumptions = [
        "Observed TPR and verified depth are aggregate request-log ratios; peak cell throughput supplies cost proxy.",
        "TPR counts landed rows (rounds + accepted)/rounds, matching cell metrics; final rounds may land past output cap.",
        "Equivalent ms/round = streams * observed TPR * 1000 / peak aggregate tok/s; includes scheduling effects.",
        "Constant conditional acceptance q fitted to observed TPR and fractional mean verified depth.",
        "Unverified suffix acceptance equals fitted q: unsupported by these censored logs, a sensitivity assumption.",
        "Static-depth costs: +3.68 ms/extra row x1 (verify1/16 secant); +2.5 ms/extra row x4 (assumed).",
        "No additional draft pass: existing DSpark block generates five positions.",
        "Bucket fill ceiling assumes all four eligible streams reach k5 in their existing 24-row forward bucket.",
        "Actual gains depend on per-round padding, confidence, eligibility, context graph key, and expert traffic.",
        "Observed setup has row cap64; cap16 requires preserving each actual forward partition.",
        "Prose is frozen by policy eligibility; static prose rows are deliberately hypothetical comparisons.",
    ]
    print("Sources:", bench, log_path, sep="\n")
    print(f"Matched speed requests at 384-token record offset {start}; max TPR mismatch {error:.6f}")
    print("Assumptions:")
    for assumption in assumptions:
        print("-", assumption)
    print("\nAll TPRs below are per slot. Costs are per equivalent shared round.")
    print("| Cell | Policy | fitted q | TPR | ms/round | aggregate tok/s | +10% cost tok/s | +15% cost tok/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name in ("code-t0-s1", "code-t0-s4", "code-t0.7-s4", "mix4", "prose-t0-s4"):
        rs = matched[name]
        rounds = sum(r[1] for r in rs)
        tpr = 1 + sum(r[2] for r in rs) / rounds
        depth = sum(r[3] for r in rs) / rounds
        streams = cells[name][0]["streams"]
        speed = max(c["aggregate_tok_s"] or c["mean_tok_s"] * streams for c in cells[name])
        cost = streams * tpr * 1000 / speed
        q = fit_q(tpr, depth)

        def show(policy, tokens, ms, penalties=False):
            rate = streams * tokens * 1000 / ms
            p10, p15 = (f"{rate / 1.1:.1f}", f"{rate / 1.15:.1f}") if penalties else ("—", "—")
            print(f"| {name} | {policy} | {q:.4f} | {tokens:.3f} | {ms:.2f} | {rate:.1f} | {p10} | {p15} |")

        show(f"observed (mean k={depth:.3f})", tpr, cost)
        slope = 3.68 if streams == 1 else 2.5
        for k in (3, 4, 5):
            show(f"static k{k} surrogate", expected(q, k), cost + streams * (k - depth) * slope)
        if name.startswith("prose"):
            show("bucket policy frozen", tpr, cost)
        elif streams > 1:
            show("bucket fill k5 ceiling", expected(q, 5), cost, penalties=True)
        else:
            show("exact-row k5 surrogate", expected(q, 5), cost + (5 - depth) * slope)

    print("\nMissing replay evidence: per-round calibrated q, old/new k, accepted path length,")
    print("actual forward partitions and graph keys, E/C and historical lambda, measured phase costs.")
    print("Phase JSON counters are cumulative; subtract consecutive snapshots for cell phase deltas.")


if __name__ == "__main__":
    main()
