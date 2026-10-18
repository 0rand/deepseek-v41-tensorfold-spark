#!/usr/bin/env python3
"""Checks of tools/tensorfold-dsv41 that need no GPU, model or optional package: every file compiles, every import
resolves inside this tree or names a documented dependency, the argparse tools print their usage, scripts keep their
executable bit, and the unit tests and self-tests pass. usage: python3 -I check.py"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# dependencies outside this tree (README.md, "Dependencies"); the reference package is the TensorFold Python engine
OPTIONAL = {"tensorfold", "torch", "triton", "numpy", "tokenizers", "jinja2"}
# modules a file imports from a directory it puts on sys.path itself, by name -> the file, relative to this tree
EXTRA = {"dsv41_decode_ref.py": {"engram_shards": "dsv41_perf/engram_shards.py",
                                 "grammar_ref": "../../engine/zig/tools/zig/dsv41_grammar/grammar_ref.py"}}
# tools whose --help needs nothing outside the standard library (heavy imports are deferred into main)
HELP = """dsv41_bench/bench_http.py dsv41_bench/glue_route_timeline.py dsv41_bench/kv_glue_gpu.py
dsv41_bench/kv_glue_timeline.py dsv41_bench/pieces_check.py dsv41_decode_ref.py dsv41_diverge/diverge.py
dsv41_engram_host.py dsv41_forward_digests.py dsv41_ops/http_golden.py dsv41_ops/wire_probe.py
dsv41_perf/acclog_replay.py dsv41_perf/code_accept_model.py dsv41_perf/gap.py dsv41_perf/kcmp.py
dsv41_perf/timescmp.py dsv41_rope_tables.py dsv41_samp_ref.py dsv41_sched/golden.py dsv41_session_slots/golden.py
dsv41_slots/ref.py dsv41_triton_capture.py dsv41_triton/check_dtopk_b.py dsv41_triton/check_scores_b.py
dsv41_triton/check_stream_rb.py dsv41_triton/check_topk_b_emu.py server_tests/gen_dsml.py server_tests/gen_template.py
triton_aot_manifest.py""".split()
EXECUTABLE = """dsv41_bench/peer.py dsv41_bench/pieces_check.py dsv41_diverge/diverge.py dsv41_session_slots/golden.py
kernels/sass_check.sh check.py""".split()
# directories holding unittest modules (pattern *_test.py or test_*.py)
TESTS = [("dsv41_perf", "*_test.py"), ("dsv41_bench", "*_test.py"), ("tests", "test_*.py")]


def sources() -> list[Path]:
    return sorted(p for p in HERE.rglob("*.py") if "__pycache__" not in p.parts)


def local_names(path: Path) -> set[str]:
    """Top-level module names importable from a file: its directory, its parent (dsv41_slots/ref.py) and this tree's
    root (the tools' sys.path entries), plus dsv41_triton (triton_fill_compile's own package directory)."""
    names: set[str] = set()
    for d in {path.parent, path.parent.parent, HERE, HERE / "dsv41_triton"}:
        if d.is_dir() and d.is_relative_to(HERE):
            names |= {p.stem for p in d.glob("*.py")} | {p.name for p in d.iterdir() if (p / "__init__.py").is_file()}
    return names


def imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def closure() -> list[str]:
    bad = []
    std = set(sys.stdlib_module_names)
    for p in sources():
        rel = p.relative_to(HERE)
        try:
            tree = ast.parse(p.read_text(), str(rel))
            compile(tree, str(rel), "exec")
        except SyntaxError as e:
            bad.append(f"{rel}: does not compile: {e}")
            continue
        local = local_names(p)
        for name, target in EXTRA.get(str(rel), {}).items():
            if (HERE / target).is_file():
                local.add(name)
            else:
                bad.append(f"{rel}: {name} expected at {target}, which is missing")
        for name in sorted(imports(tree)):
            if name not in std and name not in local and name not in OPTIONAL and name != "__future__":
                bad.append(f"{rel}: import {name} is neither standard, in this tree, nor a documented dependency")
    return bad


def run(cmd: list[str]) -> tuple[int, str]:
    """``cmd`` from a throwaway working directory: a tool that writes a relative path (sqlite3.connect("--help")
    creates the file) writes there, not into this tree."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TF_DSV41_")}
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def listing() -> set[str]:
    """Every file under this tree but bytecode caches."""
    return {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def main() -> int:
    fails = 0

    def check(ok: bool, what: str, detail: str = "") -> None:
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'} {what}")
        if not ok:
            fails += 1
            print("    " + detail.strip().replace("\n", "\n    ")[:2000])

    files = sources()
    before = listing()
    odd = sorted(f for f in before if any(part.startswith("-") for part in Path(f).parts))
    check(not odd, "no file or directory name starts with '-' (a stray --help output)", "\n".join(odd))
    bad = closure()
    check(not bad, f"{len(files)} files compile; imports closed over this tree and {sorted(OPTIONAL)}", "\n".join(bad))
    for rel in HELP:
        p = HERE / rel
        rc, out = run([sys.executable, "-I", "-B", str(p), "--help"])
        check(rc == 0 and out.startswith("usage:"), f"{rel} --help", out)
    for rel in EXECUTABLE:
        check(os.access(HERE / rel, os.X_OK), f"{rel} is executable")
    for d, pattern in TESTS:
        rc, out = run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", str(HERE / d), "-t", str(HERE / d), "-p", pattern])
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        check(rc == 0 and tail.startswith("OK"), f"unittest {d}/{pattern}: {tail}", out)
    rc, out = run([sys.executable, "-I", "-B", str(HERE / "dsv41_perf" / "gap.py"), "selftest"])
    check(rc == 0, "dsv41_perf/gap.py selftest", out)
    after = listing()
    changed = sorted(f"+ {f}" for f in after - before) + sorted(f"- {f}" for f in before - after)
    check(not changed, "the checks created or removed no file under this tree", "\n".join(changed))
    print(f"RESULT {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
