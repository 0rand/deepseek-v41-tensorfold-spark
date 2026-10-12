#!/usr/bin/env python3
"""Preflight of a Zig run against its container's env, mounts, assets and boot limits; exit 1 on any failure."""

import argparse
import json
import os
import struct
import sys

PATH_KNOBS = ("TF_DSV41_ENGRAM_DIR", "TF_DSV41_CALIB_DIR", "TF_DSV41_CALIB_FILE", "TF_DSV41_PF_DENSE_TABLE",
              "TF_DSV41_PREPARED", "TF_DSV41_SESSION_DISK", "TF_TP_KERNEL_IMAGE", "TF_DSV41_ASSETS")
OUT_KNOBS = ("TF_DSV41_PHASES_OUT", "TF_DSV41_PROFILE")
ROPE_EXTRA = 2048       # model.zig: rope_rows = TF_DSV41_CONTEXT + 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE, docker -e order")
    ap.add_argument("--map", action="append", default=[], help="/container=/host (the run's mounts)")
    ap.add_argument("--assets", required=True, help="the ASSETS argument (container path)")
    ap.add_argument("--refs", nargs="*", default=[], help="the reference dirs (container paths)")
    ap.add_argument("--fill-aot", help="the AOT fill's aot dir (host path)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--names", help="the engine's knob names, one a line")
    ap.add_argument("--pending", action="store_true", help="the stage has not built its own files yet")
    ap.add_argument("--prompt-lens", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="the references' prompt lengths (with --pending)")
    a = ap.parse_args()
    maps = sorted((m.split("=", 1) for m in a.map), key=lambda m: -len(m[0]))
    fails: list[str] = []

    def host(p: str) -> str:
        """A container path as this host sees it (symlinks inside resolved through the same mounts)."""
        for _ in range(16):
            for c, h in maps:
                if p == c or p.startswith(c.rstrip("/") + "/"):
                    p = h + p[len(c.rstrip("/")):]
                    break
            if os.path.islink(p):
                t = os.readlink(p)
                p = t if os.path.isabs(t) else os.path.normpath(os.path.join(os.path.dirname(p), t))
                continue
            return p
        return p

    def pend(what: str, path: str) -> bool:
        """With --pending, a file the stage builds: absent is fine (not checked yet)."""
        if a.pending and not os.path.exists(path):
            print(f"pieces_check: pend {what}: {path} (the stage builds it)", flush=True)
            return True
        return False

    def ok(what: str, cond: bool, detail: str = "") -> bool:
        print(f"pieces_check: {'ok  ' if cond else 'FAIL'} {what}{(': ' + detail) if detail else ''}", flush=True)
        if not cond:
            fails.append(what)
        return cond

    env: dict[str, str] = {}
    for line in open(a.env_file):
        line = line.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v
    for s in a.set:
        k, _, v = s.partition("=")
        env[k] = v
    print("pieces_check: effective env: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())), flush=True)

    if a.names and ok("knob names list", os.path.isfile(a.names), a.names):
        known = {l.strip() for l in open(a.names) if l.strip()}
        unknown = [k for k in env if k.split("_", 1)[0] in ("TF", "GLM53", "SESS4") and k not in known]
        ok("every knob set is one the engine reads", not unknown, ", ".join(unknown))

    context = int(env.get("TF_DSV41_CONTEXT", "4096"))
    rows_need = context + ROPE_EXTRA
    assets = host(a.assets)
    ok("assets dir", os.path.isdir(assets), f"{a.assets} -> {assets}")
    rope = host(os.path.join(a.assets, "rope.bin"))
    if ok("rope.bin", os.path.isfile(rope), rope):
        with open(rope, "rb") as fh:
            head = fh.read(16)
        magic, (rows, cols) = head[:8], struct.unpack("<II", head[8:16]) if len(head) == 16 else (0, 0)
        size = os.path.getsize(rope)
        ok("rope.bin magic", magic == b"DSV41RP1", repr(magic))
        ok("rope.bin rows >= TF_DSV41_CONTEXT + 2048", rows >= rows_need, f"{rows} rows, {rows_need} needed")
        ok("rope.bin size = its header's", size == 16 + 2 * rows * cols * 4, f"{size} bytes, {rows} x {cols}")
    ehb = host(os.path.join(a.assets, "engram-host.bin"))
    if not pend("engram-host.bin", ehb):
        ok("engram-host.bin", os.path.isfile(ehb) and os.path.getsize(ehb) > 0, ehb)
    aotj = host(os.path.join(a.assets, "aot", "aot.json"))
    if a.pending and not os.path.exists(aotj) and a.fill_aot:
        pend("aot/aot.json", aotj)
        aotj = os.path.join(a.fill_aot, "aot.json")      # the AOT fill's own set: its every cubin
    if ok("aot/aot.json", os.path.isfile(aotj), aotj):
        have = {k["hash"] for k in json.load(open(aotj))["kernels"]}
        cub = os.path.join(os.path.dirname(aotj), "cubins")
        missing_cub = [h for h in have if not os.path.isfile(os.path.join(cub, f"{h}.cubin"))]
        ok("every kernel's cubin present", not missing_cub, f"{len(have)} kernels, {len(missing_cub)} cubins missing")
        if a.fill_aot:
            fill = {k["hash"] for k in json.load(open(os.path.join(a.fill_aot, "aot.json")))["kernels"]}
            ok("the AOT fill's kernels are all in the run's set", fill <= have, f"{len(fill - have)} of {len(fill)} missing")
    for k in PATH_KNOBS:
        if env.get(k):
            ok(f"{k} exists", os.path.exists(host(env[k])), f"{env[k]} -> {host(env[k])}")
    for k in OUT_KNOBS:
        if env.get(k):
            d = os.path.dirname(host(env[k])) or "."
            ok(f"{k}'s directory exists", os.path.isdir(d), d)

    total = 0
    if a.pending:
        for n in a.prompt_lens:
            ok(f"prompt of {n}: prompt + steps within TF_DSV41_CONTEXT", n + a.steps <= context, f"{n + a.steps} <= {context}")
            total += n + a.steps
    for r in a.refs:
        j = host(os.path.join(r, "ref.json"))
        if pend(f"reference {r}", j):
            continue
        if not ok(f"reference {r}", os.path.isfile(j), j):
            continue
        ref = json.load(open(j))
        n = len(ref.get("prompt_ids", []))
        ok(f"reference {r}: prompt_ids and enough tokens", n > 0 and len(ref.get("tokens", [])) >= a.steps,
           f"{n}-token prompt, {len(ref.get('tokens', []))} tokens for {a.steps} steps")
        ok(f"reference {r}: prompt + steps within TF_DSV41_CONTEXT", n + a.steps <= context, f"{n + a.steps} <= {context}")
        total += n + a.steps
    if "TF_DSV41_POOL_TOKENS" in env:
        ok("TF_DSV41_POOL_TOKENS holds every chat at once", int(env["TF_DSV41_POOL_TOKENS"]) >= total,
           f"{env['TF_DSV41_POOL_TOKENS']} >= {total}")
    slots = int(env.get("TF_DSV41_SLOTS", "1"))
    chats = max(len(a.refs), len(a.prompt_lens))
    ok("TF_DSV41_SLOTS holds every chat", slots >= chats, f"{slots} slots, {chats} chats")
    if fails:
        print(f"pieces_check: FAIL {len(fails)} check(s): {'; '.join(fails)}", flush=True)
        return 1
    print("pieces_check: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
