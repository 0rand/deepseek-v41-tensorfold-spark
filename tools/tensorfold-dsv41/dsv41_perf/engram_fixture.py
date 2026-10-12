"""Goldens for engram_rows.zig from the reference Engram host code: the bf16 dequant table and packed-shard reads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

LO, ROWS, TOTAL = 1000, 3000, 9000           # the test shard: global rows [LO, LO + ROWS) of a TOTAL-row layer


def record(row: np.ndarray) -> np.ndarray:
    """[n, 264] bytes of global rows ``row``: values from a multiplicative hash, scales in 100..139 (finite)."""

    j = np.arange(264, dtype=np.uint64)[None, :]
    r = row.astype(np.uint64)[:, None]
    h = (r * np.uint64(2654435761) + j * np.uint64(40503)) % np.uint64(1 << 32)
    v = ((h >> np.uint64(7)) & np.uint64(0xFF)).astype(np.uint8)
    v[:, 256:] = (100 + (h[:, 256:] >> np.uint64(9)) % np.uint64(40)).astype(np.uint8)
    return v


def index_sets() -> list[np.ndarray]:
    a = np.array([LO, LO + 1, LO + 1, LO + 2999, LO + 7, LO + 1500, LO + 7], dtype=np.int64)
    b = (LO + (np.arange(96, dtype=np.int64) * 977) % ROWS)                     # spread: one extent each
    c = np.arange(LO + 200, LO + 260, dtype=np.int64)[::-1].copy()                # contiguous, reversed
    return [a, b, c]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True, help="the engine's src/ (engram_host.py is read from it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = Path(a.py_src) / "tensorfold/families/deepseek_v41/cuda/engram_host.py"
    spec = importlib.util.spec_from_file_location("engram_host", path)
    eh = importlib.util.module_from_spec(spec)
    sys.modules["engram_host"] = eh
    spec.loader.exec_module(eh)
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(256, 1)
    sc = torch.arange(256, dtype=torch.int32).to(torch.uint8)[:, None].repeat(1, 8)
    lut = eh.dequant(raw, sc).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).reshape(-1)
    lines = [f"lut {hashlib.sha256(lut.astype('<u2').tobytes()).hexdigest()}"]
    lines.append(f"shard {LO} {ROWS} {TOTAL}")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "engram-l1-r0of2.bin"
        eh.write_shard(f, 1, LO, record(np.arange(LO, LO + ROWS)), TOTAL)
        rows = eh.ShardRows(f, dim=256)
        for idx in index_sets():
            got = rows.rows(torch.from_numpy(idx)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            lines.append("rows " + ",".join(str(int(x)) for x in idx) + " " +
                         hashlib.sha256(got.astype("<u2").tobytes()).hexdigest())
        rows.close()
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
