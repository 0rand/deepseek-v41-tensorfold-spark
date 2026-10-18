#!/usr/bin/env python3
"""Two op captures (TF_DSV41_GM_V2=1 vs 0): routed MoE outputs and phase ends must digest equal."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROUTED = {"tf_dsv41_x3gm_v1.gateup", "tf_dsv41_x3gm_v1.down", "tf_dsv41_x3gm_v1.gateup2", "tf_dsv41_x3gm_v1.down2"}


def ops(d: str) -> list[dict]:
    p = Path(d) / "ops.jsonl"
    if not p.exists():
        p = Path(d) / "ops.jsonl.gz"
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def written(op: dict) -> dict[int, str]:
    """buffer id -> after digest, for the buffers this launch changed."""

    return {b["id"]: b["after"] for b in op.get("buffers", []) if b.get("before") != b.get("after")}


def after_of(op: dict, arg: int) -> str | None:
    a = op["args"][arg]
    if "buf" not in a:
        return None
    for b in op.get("buffers", []):
        if b["id"] == a["buf"]:
            return b["after"]
    return None


def main() -> int:
    v2, v1 = ops(sys.argv[1]), ops(sys.argv[2])
    for name, cap in (("v2", v2), ("v1", v1)):
        n = {k: sum(o["name"] == k for o in cap) for k in sorted(ROUTED)}
        print(f"{name}: {len(cap)} launches, " + ", ".join(f"{k.split('.')[1]} {c}" for k, c in n.items()))
    a = [o for o in v2 if o["name"] not in ROUTED]
    b = [o for o in v1 if o["name"] not in ROUTED]
    if not any(o["name"] == "tf_dsv41_x3gm_v1.gateup2" for o in v2):
        print("FAIL gm2pf-same: the v2 capture launched no gateup2 (the twin ignored TF_DSV41_GM_V2=1?)")
        return 1
    if len(a) != len(b):
        print(f"FAIL gm2pf-same: {len(a)} vs {len(b)} other launches")
        return 1
    bad, info, combines, lasts = 0, 0, 0, 0
    for i, (x, y) in enumerate(zip(a, b)):
        where = f"#{i} {x.get('set')} {x.get('phase')} {x['name']}"
        if x["name"] != y["name"] or (x.get("set"), x.get("phase")) != (y.get("set"), y.get("phase")):
            print(f"FAIL gm2pf-same: {where} vs {y['name']} ({y.get('set')} {y.get('phase')})")
            return 1
        last = i + 1 == len(a) or (a[i + 1].get("set"), a[i + 1].get("phase")) != (x.get("set"), x.get("phase"))
        if x["name"] == "tensorfold_exl3_experts_v1.combine":
            combines += 1
            if after_of(x, 2) != after_of(y, 2):
                bad += 1
                print(f"  differ (gate): {where}: the routed partial")
        wx, wy = written(x), written(y)
        for k in sorted(set(wx) | set(wy)):
            if wx.get(k) == wy.get(k):
                continue
            if last:
                bad += 1
                print(f"  differ (gate): {where}: buffer {k} (the phase's last launch)")
            else:
                info += 1
                if info <= 20:
                    print(f"  differ (info): {where}: buffer {k}")
        lasts += last
    verdict = "PASS" if bad == 0 and combines > 0 else "FAIL"
    print(f"{verdict} gm2pf-same: {combines} routed partials and {lasts} phase ends equal-checked, {bad} differ; "
          f"{info} other written buffers differ (info)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
