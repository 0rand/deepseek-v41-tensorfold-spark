#!/usr/bin/env python3
"""Decode and prefill gaps of two engines' profiles as tables of milliseconds (phases/windows/prefill/prefill-ab)."""

from __future__ import annotations

import argparse
import json
import re
import sys

TOP = ("prefill", "draft", "draft.pass", "window", "commit")
ORDER = ("prefill", "draft", "draft.pass", "window", "forward", "graph.stage", "engram.wait", "graph.replay",
         "graph.capture", "candidates", "prefetch", "commit", "plan", "share", "execute", "sample", "spec")

FAMILIES = (
    ("exchange", r"nccl|allgather|all_gather|gather_kernel|glue\.exchange|glue\.embed|mailbox|roce|ds_embed"),
    ("engram", r"engram|fuse_dec|gather_cols|_dequant|gate_copy"),
    ("mHC", r"mhc|boundary|_site|_finish"),
    ("attention", r"attn|_chunks|_merge|_fused\b"),
    ("indexer", r"_scores|topk|dtopk|_keys|_index_k|_plain|_block_keys|_stream"),
    ("router", r"router|narrow|gemv|_select|_logits"),
    ("experts", r"x3ld|ld_kernel|grouped|gateup|down_combine|\bgroup|x3gm|gm_|prune|kit_weights|x3pf"),
    ("dense linears", r"lanes_linear|linear|seg_rot_in|x3seg|dense3|rot_in|_gemm|x3dn|unpack"),
    ("small ops", r"_rms|_rope|_kv_store|_pool_norm|carry|positions|cast|widen|kit_logits|memcpy|memset|copy"),
    ("L2 prefetch", r"l2p|segments_kernel|paced"),
    ("head", r"head"),
)


def family(name: str) -> str:
    low = name.lower()
    for fam, rx in FAMILIES:
        if re.search(rx, low):
            return fam
    return "other"


def find_phases(doc, cell: str | None):
    """Every phases report in a document: {label: report}. A bare report (has ``wall_ms`` and phase entries) is one."""

    out = {}

    def walk(x, path):
        if isinstance(x, dict):
            if "wall_ms" in x and any(isinstance(v, dict) and "ms" in v for v in x.values()):
                out[path or "run"] = x
                return
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(doc, "")
    if cell:
        out = {k: v for k, v in out.items() if cell in k}
    return out


def per_round(rep: dict) -> tuple[dict, int, float | None]:
    rounds = int(rep.get("rounds") or (rep.get("window") or {}).get("n") or 0)
    ms = {k: v["ms"] / rounds for k, v in rep.items() if isinstance(v, dict) and "ms" in v and rounds}
    tokens = rep.get("tokens")
    return ms, rounds, (tokens / rounds if tokens and rounds else None)


def phases_table(py: dict, zig: dict) -> list[str]:
    p, pr, ptok = per_round(py)
    z, zr, ztok = per_round(zig)
    names = [n for n in ORDER if n in p or n in z] + sorted((set(p) | set(z)) - set(ORDER))
    lines = [f"rounds: Python {pr}, Zig {zr}; tokens a round: Python {ptok or '-'}, Zig "
             f"{'%.2f' % ztok if ztok else '-'}",
             "| phase | Python ms/round | Zig ms/round | Zig - Python |", "|---|---:|---:|---:|"]
    for n in names:
        a, b = p.get(n), z.get(n)
        d = "" if a is None or b is None else f"{b - a:+.3f}"
        lines.append(f"| {'**' + n + '**' if n in TOP else n} | {'' if a is None else f'{a:.3f}'} | "
                     f"{'' if b is None else f'{b:.3f}'} | {d} |")
    ps, zs = sum(p.get(n, 0) for n in TOP), sum(z.get(n, 0) for n in TOP)
    lines.append(f"| top-level sum | {ps:.3f} | {zs:.3f} | {zs - ps:+.3f} |")
    pw = py.get("wall_ms", 0) / pr if pr else 0
    zw = zig.get("wall_ms", 0) / zr if zr else 0
    lines.append(f"| wall a round | {pw:.3f} | {zw:.3f} | {zw - pw:+.3f} |")
    return lines


def zig_rows(paths: list[str]) -> dict:
    """{(graphs, rows): (mean, min)} from the gate's log lines."""

    out = {}
    for path in paths:
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if '"prof_rows"' not in line:
                continue
            try:
                d = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            out[(bool(d.get("graphs")), int(d["prof_rows"]))] = (d["ms_mean"], d["ms_min"])
    return out


def windows_table(py: dict, zig_paths: list[str], zig_prof: dict | None) -> list[str]:
    zr = zig_rows(zig_paths)
    rows = sorted({int(n) for n in py.get("rows", {})} | {n for _, n in zr})
    lines = ["| rows | Python wall ms | Python GPU ms | Zig eager wall | Zig graphed wall | graphed - Python |",
             "|---:|---:|---:|---:|---:|---:|"]
    for n in rows:
        pw = py.get("rows", {}).get(str(n), {}).get("ms_mean")
        pg = sum(v["ms"] for v in py.get("kernels", {}).get(str(n), {}).values()) or None
        ze = zr.get((False, n), (None,))[0]
        zg = zr.get((True, n), (None,))[0]
        f = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        d = "" if pw is None or zg is None else f"{zg - pw:+.3f}"
        lines.append(f"| {n} | {f(pw)} | {f(pg)} | {f(ze)} | {f(zg)} | {d} |")
    # GPU time a window by family: the reference's per row count, Zig's over its profiled eager windows
    fam_py: dict[str, dict[str, float]] = {}
    for n, ks in py.get("kernels", {}).items():
        for name, v in ks.items():
            fam_py.setdefault(family(name), {}).setdefault(n, 0.0)
            fam_py[family(name)][n] += v["ms"]
    fam_zig: dict[str, float] = {}
    if zig_prof:
        w = max(1, int(zig_prof.get("windows") or 1))
        for name, v in zig_prof.get("calls", {}).items():
            fam_zig[family(name)] = fam_zig.get(family(name), 0.0) + v["ms"] / w
    if fam_py or fam_zig:
        ns = sorted(py.get("kernels", {}), key=int)
        lines += ["", "GPU ms a window by family (Python per row count; Zig: its profiled eager windows, event-timed)",
                  "| family | " + " | ".join(f"Python {n}-row" for n in ns) + " | Zig eager |",
                  "|---|" + "---:|" * (len(ns) + 1)]
        for fam in [f for f, _ in FAMILIES] + ["other"]:
            if fam not in fam_py and fam not in fam_zig:
                continue
            cells = [f"{fam_py.get(fam, {}).get(n, 0.0):.3f}" for n in ns]
            lines.append(f"| {fam} | " + " | ".join(cells) + f" | {fam_zig.get(fam, 0.0):.3f} |")
        if zig_prof:
            w = max(1, int(zig_prof.get("windows") or 1))
            lines.append(f"Zig eager: wall {zig_prof['wall_ms'] / w:.3f} ms a window, GPU {zig_prof['gpu_ms'] / w:.3f}, "
                         f"host share {(zig_prof['wall_ms'] - zig_prof['gpu_ms']) / w:.3f} ({w} windows)")
    return lines


def prefill_table(py: dict, zig: dict, top: int = 20) -> list[str]:
    """``py`` / ``zig``: the two profiles' ``prefill`` objects (one prompt each)."""

    lines = ["| | Python | Zig | Zig - Python |", "|---|---:|---:|---:|"]
    for label, pk, zk in (("prompt rows", "rows", None), ("segments", "segments", "segments"),
                          ("wall ms", "wall_ms", "wall_ms"), ("GPU ms (kernels / calls)", "gpu_ms", "gpu_ms")):
        pv, zv = py.get(pk), zig.get(zk) if zk else None
        f = lambda x: "" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))  # noqa: E731
        d = f"{zv - pv:+.1f}" if isinstance(pv, (int, float)) and isinstance(zv, (int, float)) else ""
        lines.append(f"| {label} | {f(pv)} | {f(zv)} | {d} |")
    fam_py: dict[str, float] = {}
    for name, v in py.get("kernels", {}).items():
        fam_py[family(name)] = fam_py.get(family(name), 0.0) + v["ms"]
    fam_zig: dict[str, float] = {}
    for name, v in zig.get("calls", {}).items():
        fam_zig[family(name)] = fam_zig.get(family(name), 0.0) + v["ms"]
    lines += ["", "GPU ms a prompt by family", "| family | Python | Zig | Zig - Python |", "|---|---:|---:|---:|"]
    for fam in [f for f, _ in FAMILIES] + ["other"]:
        if fam in fam_py or fam in fam_zig:
            a, b = fam_py.get(fam, 0.0), fam_zig.get(fam, 0.0)
            lines.append(f"| {fam} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
    for side, ks in (("Python", py.get("kernels", {})), ("Zig", zig.get("calls", {}))):
        lines += ["", f"{side}'s largest {top} (ms a prompt, calls)", "| name | family | ms | calls |", "|---|---|---:|---:|"]
        for name, v in sorted(ks.items(), key=lambda kv: -kv[1]["ms"])[:top]:
            lines.append(f"| {name[:90]} | {family(name)} | {v['ms']:.1f} | {v['calls']} |")
    return lines


def prefill_ab(a: dict, b: dict, labels: tuple[str, str], top: int = 20) -> list[str]:
    """Two Zig profiles' ``prefill`` objects, per 1K encoder rows."""

    ka = 1024.0 / max(1, a.get("rows") or 0)
    kb = 1024.0 / max(1, b.get("rows") or 0)
    la, lb = labels
    lines = [f"| per 1K encoder rows | {la} | {lb} | {lb} - {la} |", "|---|---:|---:|---:|"]
    lines.append(f"| encoder rows / tail rows / segments | {a.get('rows')} / {a.get('tail_rows')} / {a.get('segments')} | "
                 f"{b.get('rows')} / {b.get('tail_rows')} / {b.get('segments')} | |")
    for label, f in (("wall ms", lambda x: x["wall_ms"]), ("GPU ms", lambda x: x["gpu_ms"]),
                     ("host ms (wall - GPU)", lambda x: x["wall_ms"] - x["gpu_ms"])):
        va, vb = f(a) * ka, f(b) * kb
        lines.append(f"| {label} | {va:.1f} | {vb:.1f} | {vb - va:+.1f} |")
    fa: dict[str, float] = {}
    fb: dict[str, float] = {}
    for d, k, fam in ((a, ka, fa), (b, kb, fb)):
        for name, v in d.get("calls", {}).items():
            fam[family(name)] = fam.get(family(name), 0.0) + v["ms"] * k
    lines += ["", "GPU ms per 1K encoder rows by family", f"| family | {la} | {lb} | {lb} - {la} |", "|---|---:|---:|---:|"]
    for fam in sorted(set(fa) | set(fb), key=lambda x: -(fb.get(x, 0) - fa.get(x, 0))):
        lines.append(f"| {fam} | {fa.get(fam, 0):.1f} | {fb.get(fam, 0):.1f} | {fb.get(fam, 0) - fa.get(fam, 0):+.1f} |")
    ca, cb = a.get("calls", {}), b.get("calls", {})
    moved = []
    for name in set(ca) | set(cb):
        x = ca.get(name, {"ms": 0.0, "calls": 0})
        y = cb.get(name, {"ms": 0.0, "calls": 0})
        moved.append((y["ms"] * kb - x["ms"] * ka, name, x, y))
    lines += ["", f"the {top} calls that moved most (ms per 1K encoder rows)", f"| name | family | {la} | {lb} | delta | calls {la} / {lb} |",
              "|---|---|---:|---:|---:|---:|"]
    for d, name, x, y in sorted(moved, key=lambda t: -abs(t[0]))[:top]:
        lines.append(f"| {name[:80]} | {family(name)} | {x['ms'] * ka:.2f} | {y['ms'] * kb:.2f} | {d:+.2f} | {x['calls']} / {y['calls']} |")
    return lines


def selftest() -> int:
    py = {"mode": "1", "wall_ms": 1000.0, "window": {"ms": 600.0, "n": 20, "ms_avg": 30.0},
          "draft.pass": {"ms": 80.0, "n": 20, "ms_avg": 4.0}, "commit": {"ms": 20.0, "n": 20, "ms_avg": 1.0}}
    zig = {"mode": "1", "wall_ms": 1300.0, "rounds": 20, "tokens": 64, "window": {"ms": 800.0, "n": 20, "ms_avg": 40},
           "forward": {"ms": 700.0, "n": 20, "ms_avg": 35}, "draft.pass": {"ms": 120.0, "n": 20, "ms_avg": 6},
           "commit": {"ms": 30.0, "n": 20, "ms_avg": 1.5}}
    t = phases_table(py, zig)
    assert any(l.startswith("| **window** | 30.000 | 40.000 | +10.000") for l in t), t
    assert any(l.startswith("| top-level sum | 35.000 | 47.500 | +12.500") for l in t), t
    assert find_phases({"cells": {"code": {"phases": py}, "prose": {"phases": zig}}}, "code") == {"cells.code.phases": py}
    assert family("tf_dsv41_mhc.run") == "mHC" and family("ncclDevKernel_AllGather_RING") == "exchange"
    assert family("glue.exchange_f32") == "exchange" and family("tf_dsv41_x3ld.gate_up") == "experts"
    assert family("_rms2") == "small ops" and family("glue.engram_rows") == "engram"
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "rank0.log")
        with open(log, "w") as f:
            f.write('{"prof_rows": 1, "graphs": false, "ms_mean": 30.0, "ms_min": 29.0}\n')
            f.write('{"prof_rows": 1, "graphs": true, "ms_mean": 24.0, "ms_min": 23.5}\n')
        w = windows_table({"rows": {"1": {"ms_mean": 26.0, "ms_min": 25.0}},
                           "kernels": {"1": {"boundary_kernel": {"calls": 81, "ms": 2.3}}}},
                          [log], {"windows": 2, "wall_ms": 60.0, "gpu_ms": 40.0,
                                  "calls": {"tf_dsv41_mhc.run": {"calls": 162, "ms": 5.0}}})
        assert any(l.startswith("| 1 | 26.000 | 2.300 | 30.000 | 24.000 | -2.000") for l in w), w
        assert any(l.startswith("| mHC | 2.300 | 2.500") for l in w), w
    pt = prefill_table({"rows": 4096, "segments": 2, "wall_ms": 3000.0, "gpu_ms": 2900.0,
                        "kernels": {"gatherTopK": {"calls": 40, "ms": 50.0}, "_fused_kernel": {"calls": 40, "ms": 900.0}}},
                       {"segments": 2, "wall_ms": 6000.0, "gpu_ms": 5000.0,
                        "calls": {"glue.topk": {"calls": 40, "ms": 400.0}, "_fused": {"calls": 40, "ms": 900.0}}})
    assert "| wall ms | 3000.0 | 6000.0 | +3000.0 |" in pt and "| indexer | 50.0 | 400.0 | +350.0 |" in pt, pt
    ab = prefill_ab({"rows": 4096, "tail_rows": 127, "segments": 3, "wall_ms": 1600.0, "gpu_ms": 1500.0,
                     "calls": {"_fused": {"calls": 40, "ms": 100.0}, "glue.engram_rows": {"calls": 4, "ms": 40.0}}},
                    {"rows": 8192, "tail_rows": 127, "segments": 3, "wall_ms": 3600.0, "gpu_ms": 3000.0,
                     "calls": {"_fused": {"calls": 40, "ms": 200.0}, "glue.engram_rows": {"calls": 4, "ms": 400.0}}},
                    ("2K", "4K"), 5)
    assert "| wall ms | 400.0 | 450.0 | +50.0 |" in ab and "| host ms (wall - GPU) | 25.0 | 75.0 | +50.0 |" in ab, ab
    moved = [l for l in ab if l.startswith("| glue.engram_rows") or l.startswith("| _fused")]
    assert moved[0].startswith("| glue.engram_rows") and "+40.00" in moved[0], ab  # the call that moved most first
    print("gap.py selftest: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("phases")
    a.add_argument("--py", required=True)
    a.add_argument("--zig", required=True)
    a.add_argument("--cell", default=None, help="a substring of the Python report's cell path (e.g. code, prose)")
    b = sub.add_parser("windows")
    b.add_argument("--py", required=True)
    b.add_argument("--zig", action="append", default=[])
    b.add_argument("--zig-profile", default=None)
    e = sub.add_parser("prefill-ab")
    e.add_argument("--a", required=True)
    e.add_argument("--b", required=True)
    e.add_argument("--labels", default="A,B")
    e.add_argument("--top", type=int, default=20)
    c = sub.add_parser("prefill")
    c.add_argument("--py", required=True)
    c.add_argument("--zig-profile", required=True)
    c.add_argument("--top", type=int, default=20)
    sub.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "selftest":
        return selftest()
    if x.cmd == "phases":
        pys = find_phases(json.load(open(x.py)), x.cell)
        zigs = find_phases(json.load(open(x.zig)), None)
        if not pys or not zigs:
            print(f"no phases report in {x.py if not pys else x.zig}", file=sys.stderr)
            return 1
        for label, rep in pys.items():
            print(f"### Python {label} vs Zig {next(iter(zigs))}")
            print("\n".join(phases_table(rep, zigs[next(iter(zigs))])))
            print()
        return 0
    if x.cmd == "prefill-ab":
        pa, pb = json.load(open(x.a)).get("prefill"), json.load(open(x.b)).get("prefill")
        if not pa or not pb or not pa.get("rows") or not pb.get("rows"):
            print("both profiles need a \"prefill\" with \"rows\" (an engine with profile.prefillRows)", file=sys.stderr)
            return 2
        la, lb = (x.labels.split(",") + ["B"])[:2]
        print("\n".join(prefill_ab(pa, pb, (la, lb), x.top)))
        return 0
    if x.cmd == "prefill":
        py, zig = json.load(open(x.py)).get("prefill"), json.load(open(x.zig_profile)).get("prefill")
        if not py or not zig:
            print(f"no \"prefill\" in {x.py if not py else x.zig_profile}", file=sys.stderr)
            return 1
        print("\n".join(prefill_table(py, zig, x.top)))
        return 0
    prof = json.load(open(x.zig_profile)) if x.zig_profile else None
    print("\n".join(windows_table(json.load(open(x.py)), x.zig, prof)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
