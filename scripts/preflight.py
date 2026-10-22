#!/usr/bin/env python3
"""Preflight checks for one node of the Zig serving pair (standard library only, opt-in, read-only).

Run it on each node with the serving env (config/prod-zig.env), before a boot and, with --after-boot, once the rank is
up. scripts/run-zig.sh runs it before the container starts when PREFLIGHT=1 is set (default off); PREFLIGHT_ARGS adds
options there.

  python3 scripts/preflight.py --rank 0                     # reads CONFIG (default config/prod-zig.env) or the env
  python3 scripts/preflight.py --rank 1 --json              # one JSON object instead of the text report
  python3 scripts/preflight.py --rank 0 --after-boot        # + the engine's boot-check lines (docker logs) and a
                                                            #   suggested largest TF_DSV41_POOL_TOKENS

Checks (each ok / warn / fail / info; the exit code is 0 all ok, 1 a warning, 2 a failure):

  engram-fs       ENGRAM_DIR is on a local filesystem: not NFS / CIFS / sshfs / another network or FUSE mount, and not
                  tmpfs (95 GB of RAM). A decode round reads many random 4 KiB Engram rows; Mia's recipe measured 942 us
                  a read over NFS against 90 us from a local copy.
  engram-files    this rank's engram-l{L}-r{rank}of2.bin shards are there, their 4 KiB headers are valid
                  (scripts/pack_engram.py's format), each size is header + rows x 264 bytes, and, when MODEL_DIR's
                  config.json is readable, their row ranges are the ones the pack's Engram layout gives this rank.
                  Leftover *.bin.tmp files (an interrupted pack) warn.
  engram-latency  a random 4 KiB read probe on the largest shard (O_DIRECT when the filesystem allows it, so the page
                  cache does not answer): p50 / p99 in microseconds; warn above --latency-warn-us (300). --no-probe skips.
  memory          MemTotal / MemAvailable / /dev/shm use (shared memory counts against MemAvailable); before a boot an
                  ESTIMATE of what the rank needs: weights (MODEL_DIR's *.safetensors / 2 ranks) + the pool
                  (TF_DSV41_POOL_TOKENS x ~1,060 B a token with split KV, ~1,790 replicated) + the serving dip (3.0 GiB)
                  + the hard floor (4 GiB). The weights term is the files' size, not the resident layout, and the boot's
                  own growth is not in it: the engine's boot check is the authority. It warns only (--min-avail-gib N
                  makes "MemAvailable under N" a failure, as a start gate).
  pool            (--after-boot) the engine's boot-check lines from its log (--boot-log FILE, else docker logs of
                  tensorfold-zig-r{rank}): the pool, the bytes a token, the predicted worst MemAvailable; and the
                  largest TF_DSV41_POOL_TOKENS that keeps that worst point at the hard floor + --margin-gib (0.25, the
                  4.25 GiB rule of docs/FLASH-1017.md). Without the log it falls back to MemAvailable now (valid only
                  with the engine idle) and the documented bytes a token, and says so.
  health          (--after-boot --url URL) /health: ok, and graph caches under their memory floor.

The engine's model does not include the transient 4K prefill workspace (2,268 MiB a rank): on flash17 the model
allowed 3.0M tokens but 2.1M was the largest pool that held the 4 GiB floor under 4K prefills (docs/FLASH-1017.md).
Ideas from Mia's recipe (MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks: Engram tables on each Spark's own disk,
and a memory check at start that reports /dev/shm); this file is our own implementation.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GIB = 1 << 30
OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
RANK = {OK: 0, INFO: 0, WARN: 1, FAIL: 2}

# scripts/pack_engram.py's shard format
ENGRAM_MAGIC = 0x31344E4531565344  # b"DSV41EN1" little-endian
ENGRAM_HEADER = 4096
ENGRAM_ROW = 264
DEFAULT_LAYERS = (1, 14)  # docs/ZIG-SERVE.md: engram-l{1,14}-r{rank}of2.bin
NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "9p",
    "ceph",
    "glusterfs",
    "lustre",
    "afs",
    "davfs",
    "fuse.sshfs",
    "fuse.rclone",
    "fuse.s3fs",
    "fuse.gcsfuse",
    "fuse.glusterfs",
    "fuse.cephfs",
    "fuse.juicefs",
}
RAM_FS = {"tmpfs", "ramfs"}
# bytes a pool token on each rank (docs/ZIG-SERVE.md "Split KV and the pool"; engine kv/budget.zig): approximate
BYTES_A_TOKEN = {True: 1060, False: 1790}
SERVE_DIP_GIB = 3.0  # TF_DSV41_SERVE_DIP_GIB default (engine kv/budget.zig zig_dip_gib)
HARD_FLOOR_GIB = 4.0  # TF_DSV41_FLOOR_HARD_GIB default
PF_4K_WORKSPACE_MIB = 2268  # docs/FLASH-1017.md
SETTINGS = (
    "MODEL_DIR",
    "ENGRAM_DIR",
    "TF_DSV41_POOL_TOKENS",
    "TF_DSV41_KV_SPLIT",
    "TF_DSV41_FLOOR_HARD_GIB",
    "TF_DSV41_FLOOR_GIB",
    "TF_DSV41_SERVE_DIP_GIB",
    "TF_DSV41_POST_BOOT_GIB",
    "TF_DSV41_PF_4K",
    "TF_TP_RANK",
)


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str, **data) -> None:
        self.checks.append(
            {
                "check": name,
                "status": status,
                "detail": detail,
                **({"data": data} if data else {}),
            }
        )

    @property
    def code(self) -> int:
        return max((RANK[c["status"]] for c in self.checks), default=0)


# ---------------------------------------------------------------------------------------------------------- inputs
def read_env_file(path: Path) -> dict:
    """KEY=VALUE lines of a shell env file (quotes stripped, ``export`` allowed; no expansion)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        v = m[2].strip()
        if v[:1] in "\"'":
            q = v[0]
            end = v.find(q, 1)
            v = v[1:end] if end > 0 else v[1:]
        else:
            v = v.split(" #", 1)[0].strip()
        out[m[1]] = v
    return out


def settings(config: Path | None) -> tuple[dict, str]:
    """The serving settings: the environment wins (run-zig.sh exports the sourced config), else the config file."""
    file_vals = read_env_file(config) if config and config.is_file() else {}
    vals = {k: os.environ.get(k, file_vals.get(k)) for k in SETTINGS}
    src = (
        "environment"
        if any(k in os.environ for k in SETTINGS)
        else (str(config) if file_vals else "none")
    )
    return {k: v for k, v in vals.items() if v is not None}, src


def meminfo(path: str) -> dict:
    """/proc/meminfo in bytes."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                out[parts[0].rstrip(":")] = int(parts[1]) * (
                    1024 if len(parts) > 2 and parts[2] == "kB" else 1
                )
    return out


def unescape_mount(s: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)


def mount_of(path: str, mountinfo_path: str) -> dict | None:
    """The mount (mountpoint, fstype, source) holding ``path``, from a /proc/<pid>/mountinfo file."""
    best = None
    with open(mountinfo_path) as f:
        for line in f:
            left, sep, right = line.partition(" - ")
            if not sep:
                continue
            a, b = left.split(), right.split()
            if len(a) < 5 or len(b) < 2:
                continue
            mp = unescape_mount(a[4])
            inside = path == mp or mp == "/" or path.startswith(mp.rstrip("/") + "/")
            if inside and (best is None or len(mp) >= len(best["mountpoint"])):
                best = {
                    "mountpoint": mp,
                    "fstype": b[0],
                    "source": unescape_mount(b[1]),
                }
    return best


# ---------------------------------------------------------------------------------------------------------- engram
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):  # deterministic below 4,759,123,141
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_layout(cfg: dict, rank: int, world: int) -> dict[int, tuple[int, int, int]]:
    """{layer: (lo, hi, total rows)} of this rank's shard: the same arithmetic as scripts/pack_engram.py."""
    seen: set[int] = set()
    out = {}
    for layer in cfg["engram_layer_ids"]:
        primes = []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                primes.append(cur)
        part = (len(primes) + world - 1) // world
        h0, h1 = rank * part, min(rank * part + part, len(primes))
        out[layer] = (sum(primes[:h0]), sum(primes[:h1]), sum(primes))
    return out


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        head = f.read(48)
    if len(head) < 48:
        raise ValueError("shorter than its header")
    magic, layer, lo, hi, total, row = struct.unpack("<6Q", head)
    if magic != ENGRAM_MAGIC:
        raise ValueError("not an Engram shard (bad magic)")
    return {"layer": layer, "lo": lo, "hi": hi, "total": total, "row_bytes": row}


def check_engram(
    rep: Report,
    edir: str | None,
    rank: int | None,
    world: int,
    model_dir: str | None,
    mountinfo_path: str,
    probe: int,
    warn_us: float,
    seed: int,
) -> None:
    if not edir or "<" in edir:
        rep.add(
            "engram-fs",
            FAIL,
            "ENGRAM_DIR is not set (config/prod-zig.env or the environment)",
        )
        return
    real = os.path.realpath(edir)
    if not os.path.isdir(real):
        rep.add("engram-fs", FAIL, f"ENGRAM_DIR {edir} is not a directory")
        return
    try:
        m = mount_of(real, mountinfo_path)
    except OSError as exc:
        m = None
        rep.add("engram-fs", WARN, f"cannot read {mountinfo_path}: {exc}")
    if m:
        fs = m["fstype"]
        where = f"{real} is on {m['mountpoint']} ({fs}, {m['source']})"
        if fs in NETWORK_FS:
            rep.add(
                "engram-fs",
                FAIL,
                where
                + ": a network filesystem; copy the shards to this node's own disk "
                "(random 4 KiB reads: 942 us over NFS vs 90 us local in Mia's recipe)",
                **m,
            )
        elif fs in RAM_FS:
            rep.add(
                "engram-fs",
                FAIL,
                where
                + ": RAM-backed, the shards would take ~95 GB of the GPU's memory",
                **m,
            )
        elif fs.startswith("fuse"):
            rep.add(
                "engram-fs",
                WARN,
                where + ": a FUSE mount, cannot tell whether it is local",
                **m,
            )
        else:
            rep.add("engram-fs", OK, where, **m)
    files = sorted(Path(real).glob("engram-l*-r*of*.bin"))
    ranks = sorted({int(re.search(r"-r(\d+)of", p.name)[1]) for p in files})
    if rank is None:
        if len(ranks) == 1:
            rank = ranks[0]
        else:
            rep.add(
                "engram-files",
                FAIL,
                f"pass --rank: {real} holds shards of ranks {ranks or 'none'}",
            )
            return
    layout, layers, how = None, list(DEFAULT_LAYERS), "the documented layers 1 and 14"
    cfg_path = (
        Path(model_dir, "config.json") if model_dir and "<" not in model_dir else None
    )
    if cfg_path and cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            cfg = cfg.get("text_config", cfg) if "engram_layer_ids" not in cfg else cfg
            layout = engram_layout(cfg, rank, world)
            layers, how = sorted(layout), f"{cfg_path}'s Engram layout"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rep.add(
                "engram-files",
                WARN,
                f"{cfg_path}: no Engram layout ({type(exc).__name__}: {exc}); "
                "checking headers and sizes only",
            )
    problems, sizes, ok_files = [], {}, []
    for layer in layers:
        p = Path(real, f"engram-l{layer}-r{rank}of{world}.bin")
        if not p.is_file():
            problems.append(f"{p.name} is missing")
            continue
        try:
            h = read_header(p)
        except (OSError, ValueError) as exc:
            problems.append(f"{p.name}: {exc}")
            continue
        size = p.stat().st_size
        sizes[p.name] = size
        want = ENGRAM_HEADER + (h["hi"] - h["lo"]) * h["row_bytes"]
        if (
            h["layer"] != layer
            or h["row_bytes"] != ENGRAM_ROW
            or not h["lo"] < h["hi"] <= h["total"]
        ):
            problems.append(
                f"{p.name}: header says layer {h['layer']}, rows {h['lo']}..{h['hi']} of {h['total']}, "
                f"{h['row_bytes']} B a row"
            )
        elif size != want:
            problems.append(
                f"{p.name}: {size:,} bytes, the header's rows need {want:,} (truncated or still writing)"
            )
        elif layout and (h["lo"], h["hi"], h["total"]) != layout[layer]:
            problems.append(
                f"{p.name}: rows {h['lo']:,}..{h['hi']:,} of {h['total']:,}, the pack's layout gives "
                f"rank {rank} {layout[layer][0]:,}..{layout[layer][1]:,} of {layout[layer][2]:,} "
                "(shards of another release or rank?)"
            )
        else:
            ok_files.append(p)
    tmp = [p.name for p in Path(real).glob("*.bin.tmp")]
    total = sum(sizes.values())
    if problems:
        rep.add("engram-files", FAIL, "; ".join(problems), sizes=sizes)
    elif tmp:
        rep.add(
            "engram-files",
            WARN,
            f"rank {rank}: shards valid, but leftover {', '.join(tmp)} (an interrupted pack?)",
            sizes=sizes,
        )
    else:
        rep.add(
            "engram-files",
            OK,
            f"rank {rank}: {len(ok_files)} shards valid against {how}, "
            f"{total / GIB:.1f} GiB",
            sizes=sizes,
        )
    if probe > 0 and ok_files:
        target = max(ok_files, key=lambda p: p.stat().st_size)
        try:
            r = read_latency(target, probe, seed)
        except OSError as exc:
            rep.add("engram-latency", WARN, f"probe failed on {target.name}: {exc}")
            return
        status = WARN if r["p50_us"] > warn_us else OK
        mode = (
            "O_DIRECT"
            if r["direct"]
            else "buffered (O_DIRECT refused; the page cache may answer some reads)"
        )
        rep.add(
            "engram-latency",
            status,
            f"{r['reads']} random 4 KiB reads of {target.name}, {mode}: "
            f"p50 {r['p50_us']:.0f} us, p99 {r['p99_us']:.0f} us (warn above {warn_us:.0f} us p50)",
            **r,
        )


def read_latency(path: Path, reads: int, seed: int) -> dict:
    """``reads`` random 4 KiB-aligned reads past the header; O_DIRECT when allowed."""
    size = path.stat().st_size
    blocks = max(1, (size - ENGRAM_HEADER) // 4096)
    rng = random.Random(seed)
    direct = hasattr(os, "O_DIRECT")
    try:
        fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    except OSError:
        direct = False
        fd = os.open(path, os.O_RDONLY)
    buf = mmap.mmap(-1, 4096)  # page-aligned, as O_DIRECT needs
    lat = []
    try:
        if not direct and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        for _ in range(reads):
            off = ENGRAM_HEADER + rng.randrange(blocks) * 4096
            t0 = time.perf_counter_ns()
            try:
                os.preadv(fd, [buf], off)
            except OSError:
                if not direct:
                    raise
                os.close(fd)  # a filesystem that opens O_DIRECT but refuses the read
                fd, direct = os.open(path, os.O_RDONLY), False
                os.preadv(fd, [buf], off)
            lat.append((time.perf_counter_ns() - t0) / 1000)
    finally:
        os.close(fd)
        buf.close()
    lat.sort()
    return {
        "reads": len(lat),
        "direct": direct,
        "p50_us": lat[len(lat) // 2],
        "p99_us": lat[min(len(lat) - 1, int(len(lat) * 0.99))],
        "max_us": lat[-1],
    }


# ---------------------------------------------------------------------------------------------------------- memory
def num(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def weights_on_disk(model_dir: str | None) -> int | None:
    if not model_dir or "<" in model_dir or not os.path.isdir(model_dir):
        return None
    total = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    return total or None


def check_memory(
    rep: Report,
    env: dict,
    mem: dict,
    shm_path: str,
    world: int,
    min_avail: float | None,
    after_boot: bool,
) -> None:
    avail, total = mem.get("MemAvailable"), mem.get("MemTotal")
    if avail is None:
        rep.add("memory", FAIL, "no MemAvailable in meminfo")
        return
    shm = None
    try:
        st = os.statvfs(shm_path)
        shm = (st.f_blocks - st.f_bfree) * st.f_frsize
    except OSError:
        pass
    line = f"MemAvailable {avail / GIB:.1f} GiB of {(total or 0) / GIB:.1f} GiB"
    if shm is not None:
        line += f", {shm_path} holds {shm / GIB:.2f} GiB"
    data = {
        "mem_available_gib": round(avail / GIB, 2),
        "mem_total_gib": round((total or 0) / GIB, 2),
        "shm_gib": None if shm is None else round(shm / GIB, 2),
    }
    if min_avail is not None and avail / GIB < min_avail:
        rep.add("memory", FAIL, line + f": under --min-avail-gib {min_avail:g}", **data)
        return
    if shm is not None and shm > GIB:
        rep.add(
            "memory",
            WARN,
            line + ": shared memory counts against MemAvailable; free it before a boot",
            **data,
        )
    if after_boot:
        if shm is None or shm <= GIB:
            rep.add("memory", INFO, line, **data)
        return
    tokens = int(num(env.get("TF_DSV41_POOL_TOKENS"), 0))
    split = env.get("TF_DSV41_KV_SPLIT") == "1"
    hard = num(env.get("TF_DSV41_FLOOR_HARD_GIB"), HARD_FLOOR_GIB)
    dip = num(env.get("TF_DSV41_SERVE_DIP_GIB"), SERVE_DIP_GIB)
    reserve = num(env.get("TF_DSV41_POST_BOOT_GIB"), 0.0)
    pool = tokens * BYTES_A_TOKEN[split] / GIB
    w = weights_on_disk(env.get("MODEL_DIR"))
    data.update(
        pool_tokens=tokens,
        pool_gib_estimate=round(pool, 2),
        hard_floor_gib=hard,
        dip_gib=dip,
    )
    if w is None:
        rep.add(
            "memory",
            INFO,
            line
            + f"; pool {tokens:,} tokens ~ {pool:.2f} GiB a rank. The weights term cannot be "
            "estimated (MODEL_DIR has no readable *.safetensors here), so no need is computed: the engine's boot "
            "check decides",
            **data,
        )
        return
    need = w / world / GIB + pool + dip + hard + reserve
    data.update(
        weights_gib_estimate=round(w / world / GIB, 2), need_gib_estimate=round(need, 2)
    )
    detail = (
        f"{line}; estimated need {need:.1f} GiB = weights ~{w / world / GIB:.1f} (files / {world} ranks) + pool "
        f"{pool:.2f} ({tokens:,} tokens, {'split' if split else 'replicated'} KV) + dip {dip:g} + hard floor "
        f"{hard:g}"
        + (f" + post-boot reserve {reserve:g}" if reserve else "")
        + " (an estimate: the boot's growth is measured by the engine, the resident layout differs from the files)"
    )
    rep.add(
        "memory",
        WARN if avail / GIB < need else (INFO if shm is not None and shm > GIB else OK),
        detail,
        **data,
    )


BEFORE_POOL = re.compile(
    r"rank (\d+) pool ([\d,]+) tokens = ([\d.]+) GiB \(([\d.]+) B a token(, split KV)?\); "
    r"predicted worst MemAvailable (-?[\d.]+) GiB \(hard floor ([\d.]+)\)"
)
AFTER_BOOT = re.compile(
    r"rank (\d+): boot growth (-?[\d.]+) GiB measured \(MemAvailable ([\d.]+) at the check - pool "
    r"([\d.]+) - ([\d.]+) now\); worst serving point (-?[\d.]+) GiB \(post-boot reserve ([\d.]+)"
    r".*?dip ([\d.]+).*?hard floor ([\d.]+)\)"
)
DOES_NOT_FIT = re.compile(r"the largest that does: TF_DSV41_POOL_TOKENS=(\d+)")
WORKSPACE = re.compile(
    r"rank (\d+): the prefill workspace \(([\d.]+) GiB.*?leaves (-?[\d.]+) GiB"
)


def parse_boot_log(text: str, rank: int | None) -> dict:
    """The last boot's kv boot-check figures (engine kv_state.zig / kv/budget.zig log lines)."""
    out: dict = {}
    for line in text.splitlines():
        if m := BEFORE_POOL.search(line):
            if rank is not None and int(m[1]) != rank:
                continue
            out = {
                "rank": int(m[1]),
                "pool_tokens": int(m[2].replace(",", "")),
                "pool_gib": float(m[3]),
                "bytes_a_token": float(m[4]),
                "split": bool(m[5]),
                "predicted_worst_gib": float(m[6]),
                "hard_gib": float(m[7]),
            }
        elif m := AFTER_BOOT.search(line):
            if rank is not None and int(m[1]) != rank:
                continue
            out.update(
                growth_gib=float(m[2]),
                at_check_gib=float(m[3]),
                idle_gib=float(m[5]),
                worst_gib=float(m[6]),
                reserve_gib=float(m[7]),
                dip_gib=float(m[8]),
                hard_gib=float(m[9]),
            )
        elif m := DOES_NOT_FIT.search(line):
            out["refused_largest"] = int(m[1])
        elif m := WORKSPACE.search(line):
            out["workspace_gib"], out["workspace_leaves_gib"] = float(m[2]), float(m[3])
    return out


def suggest_pool(
    pool_tokens: int,
    worst_gib: float,
    hard_gib: float,
    margin_gib: float,
    bytes_a_token: float,
) -> int:
    """The largest pool (rounded down to 10,000 tokens) whose worst point stays at hard + margin."""
    room = (worst_gib - hard_gib - margin_gib) * GIB / bytes_a_token
    return max(0, int((pool_tokens + room) // 10_000 * 10_000))


def boot_log_text(path: str | None, container: str) -> tuple[str | None, str]:
    if path:
        try:
            return Path(path).read_text(errors="replace"), path
        except OSError as exc:
            return None, f"{path}: {exc}"
    docker = shutil.which("docker")
    if not docker:
        return None, "no --boot-log and no docker here"
    try:
        r = subprocess.run(
            [docker, "logs", container],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"docker logs {container}: {exc}"
    if r.returncode != 0:
        return None, f"docker logs {container}: {r.stderr.strip()[:200]}"
    return r.stdout + r.stderr, f"docker logs {container}"


def check_pool(
    rep: Report,
    env: dict,
    mem: dict,
    rank: int | None,
    log_path: str | None,
    container: str,
    margin: float,
) -> None:
    text, src = boot_log_text(log_path, container)
    b = parse_boot_log(text, rank) if text else {}
    pf4k = env.get("TF_DSV41_PF_4K") == "1"
    ws_note = (
        (
            f" The transient 4K prefill workspace ({PF_4K_WORKSPACE_MIB:,} MiB a rank, TF_DSV41_PF_4K=1) is not in "
            "this model: on flash17 2.1M was the largest pool that held the 4 GiB floor under 4K prefills "
            "(docs/FLASH-1017.md); measure under load before raising the pool."
        )
        if pf4k
        else ""
    )
    if b.get("refused_largest") and "worst_gib" not in b:
        rep.add(
            "pool",
            FAIL,
            f"the engine refused the pool at boot ({src}): the largest that fits is "
            f"TF_DSV41_POOL_TOKENS={b['refused_largest']:,}",
            **b,
        )
        return
    if "worst_gib" in b and "pool_tokens" in b:
        s = suggest_pool(
            b["pool_tokens"], b["worst_gib"], b["hard_gib"], margin, b["bytes_a_token"]
        )
        b["suggested_max_pool_tokens"] = s
        status = OK if b["worst_gib"] >= b["hard_gib"] + margin else WARN
        detail = (
            f"rank {b['rank']} ({src}): pool {b['pool_tokens']:,} tokens = {b['pool_gib']:.2f} GiB "
            f"({b['bytes_a_token']:.0f} B a token); after boot MemAvailable {b['idle_gib']:.2f} GiB, worst serving "
            f"point {b['worst_gib']:.2f} GiB (dip {b['dip_gib']:g}, hard floor {b['hard_gib']:g}); largest pool "
            f"keeping hard floor + {margin:g} GiB at that point: TF_DSV41_POOL_TOKENS ~ {s:,}."
        )
        if "workspace_leaves_gib" in b:
            detail += f" The engine warned the 4K workspace leaves {b['workspace_leaves_gib']:.2f} GiB at that point."
        rep.add("pool", status, detail + ws_note, **b)
        return
    avail = mem.get("MemAvailable")
    tokens = int(num(env.get("TF_DSV41_POOL_TOKENS"), 0))
    split = env.get("TF_DSV41_KV_SPLIT") == "1"
    if avail is None or not tokens:
        rep.add(
            "pool",
            WARN,
            f"no boot-check lines ({src}) and no TF_DSV41_POOL_TOKENS / MemAvailable: cannot "
            "suggest a pool",
        )
        return
    hard = num(env.get("TF_DSV41_FLOOR_HARD_GIB"), HARD_FLOOR_GIB)
    dip = num(env.get("TF_DSV41_SERVE_DIP_GIB"), SERVE_DIP_GIB)
    reserve = num(env.get("TF_DSV41_POST_BOOT_GIB"), 0.0)
    worst = avail / GIB - reserve - dip
    s = suggest_pool(tokens, worst, hard, margin, BYTES_A_TOKEN[split])
    rep.add(
        "pool",
        WARN,
        f"no boot-check lines ({src}); APPROXIMATE from MemAvailable now ({avail / GIB:.2f} GiB, "
        f"valid only with the engine booted and idle) and ~{BYTES_A_TOKEN[split]:,} B a token: worst point "
        f"{worst:.2f} GiB, largest pool ~ {s:,} tokens (pool now {tokens:,})."
        + ws_note,
        approximate=True,
        worst_gib=round(worst, 2),
        suggested_max_pool_tokens=s,
    )


def check_health(rep: Report, url: str) -> None:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url.rstrip("/") + "/health", timeout=10) as r:
            h = json.loads(r.read())
    except (OSError, ValueError) as exc:
        rep.add("health", WARN, f"{url}/health: {exc}")
        return
    under = [
        n
        for n, g in (h.get("graphs") or {}).items()
        if isinstance(g, dict) and g.get("under_floor")
    ]
    hits = {
        n: g.get("floor_hits")
        for n, g in (h.get("graphs") or {}).items()
        if isinstance(g, dict)
    }
    if h.get("ok") is False or h.get("fatal"):
        rep.add(
            "health",
            FAIL,
            f"/health not ok: status {h.get('status')}, fatal {h.get('fatal')}",
        )
    elif under:
        rep.add(
            "health",
            WARN,
            f"graph caches under their memory floor now: {', '.join(under)} (floor hits {hits})",
        )
    else:
        rep.add(
            "health",
            OK,
            f"/health ok, inflight {h.get('inflight')}, graph floor hits {hits or 'n/a'}",
        )


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rank",
        type=int,
        choices=(0, 1),
        default=None,
        help="this node's rank (default: TF_TP_RANK, "
        "else inferred from the Engram shards)",
    )
    ap.add_argument("--config", default=os.environ.get("CONFIG", "config/prod-zig.env"))
    ap.add_argument("--engram-dir", help="override ENGRAM_DIR")
    ap.add_argument("--model-dir", help="override MODEL_DIR")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument(
        "--probe", type=int, default=256, help="random 4 KiB reads in the latency probe"
    )
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--latency-warn-us", type=float, default=300.0)
    ap.add_argument(
        "--min-avail-gib",
        type=float,
        default=None,
        help="fail when MemAvailable is under this",
    )
    ap.add_argument(
        "--after-boot", action="store_true", help="check the booted rank's pool figures"
    )
    ap.add_argument(
        "--boot-log",
        help="the rank's log file (default: docker logs tensorfold-zig-r<rank>)",
    )
    ap.add_argument("--container", default=None)
    ap.add_argument("--margin-gib", type=float, default=0.25)
    ap.add_argument("--url", help="with --after-boot: also read <url>/health (rank 0)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ap.add_argument("--mountinfo", default="/proc/self/mountinfo")
    ap.add_argument("--shm", default="/dev/shm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    cfg = Path(a.config) if os.path.isabs(a.config) else root / a.config
    env, src = settings(cfg)
    if a.engram_dir:
        env["ENGRAM_DIR"] = a.engram_dir
    if a.model_dir:
        env["MODEL_DIR"] = a.model_dir
    rank = (
        a.rank
        if a.rank is not None
        else (int(env["TF_TP_RANK"]) if env.get("TF_TP_RANK", "").isdigit() else None)
    )
    rep = Report()
    rep.add(
        "settings",
        INFO,
        f"from {src}" + (f", rank {rank}" if rank is not None else ""),
        **{k: v for k, v in env.items()},
    )
    check_engram(
        rep,
        env.get("ENGRAM_DIR"),
        rank,
        a.world,
        env.get("MODEL_DIR"),
        a.mountinfo,
        0 if a.no_probe else a.probe,
        a.latency_warn_us,
        a.seed,
    )
    try:
        mem = meminfo(a.meminfo)
    except OSError as exc:
        mem = {}
        rep.add("memory", FAIL, f"{a.meminfo}: {exc}")
    if mem:
        check_memory(rep, env, mem, a.shm, a.world, a.min_avail_gib, a.after_boot)
    if a.after_boot:
        container = a.container or f"tensorfold-zig-r{rank if rank is not None else 0}"
        check_pool(rep, env, mem, rank, a.boot_log, container, a.margin_gib)
        if a.url:
            check_health(rep, a.url)
    if a.json:
        print(json.dumps({"exit": rep.code, "checks": rep.checks}, indent=1))
    else:
        for c in rep.checks:
            print(f"[{c['status'].upper():4s}] {c['check']}: {c['detail']}")
        print(f"preflight: {['ok', 'warnings', 'FAILED'][rep.code]} (exit {rep.code})")
    return rep.code


if __name__ == "__main__":
    sys.exit(main())
