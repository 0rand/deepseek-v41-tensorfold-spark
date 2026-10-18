"""Sparse full-size Engram shards with records only under the rows a prompt hashes to; other rows read as zeros."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x31344E4531565344
HEADER = 4096
ROW_BYTES = 264


def record(row: np.ndarray) -> np.ndarray:
    """[n, 264] bytes of global rows ``row`` (engram_fixture.record): hashed values, finite scales 100..139."""

    j = np.arange(264, dtype=np.uint64)[None, :]
    r = row.astype(np.uint64)[:, None]
    h = (r * np.uint64(2654435761) + j * np.uint64(40503)) % np.uint64(1 << 32)
    v = ((h >> np.uint64(7)) & np.uint64(0xFF)).astype(np.uint8)
    v[:, 256:] = (100 + (h[:, 256:] >> np.uint64(9)) % np.uint64(40)).astype(np.uint8)
    return v


def make(out: str | Path, hasher, prompts, rank: int, world: int) -> list[Path]:
    """This rank's sparse shard of every Engram layer under ``out``: records under the rows each prompt hashes to."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lay = hasher.layout
    hashes = [hasher(np.asarray(ids, dtype=np.int64)).numpy() for ids in prompts]     # [T, layers, cols] each
    made = []
    for li, layer in enumerate(lay.layer_ids):
        lo, hi, h0, nh = lay.head_shard(li, rank, world)
        rows = np.unique(np.concatenate([h[:, li, h0:h0 + nh].reshape(-1) for h in hashes]))
        path = out / f"engram-l{layer}-r{rank}of{world}.bin"
        with open(path, "wb") as f:
            head = bytearray(HEADER)
            struct.pack_into("<6Q", head, 0, MAGIC, layer, lo, hi, lay.total_rows(li), ROW_BYTES)
            f.write(head)
            f.truncate(HEADER + (hi - lo) * ROW_BYTES)
            recs = record(rows)
            for k, r in enumerate(rows.tolist()):
                f.seek(HEADER + (r - lo) * ROW_BYTES)
                f.write(recs[k].tobytes())
        os.sync()
        made.append(path)
    return made
