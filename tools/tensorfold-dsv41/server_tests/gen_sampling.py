"""DeepSeek-V4.1 sampling goldens: logit rows picked by the reference host rule (greedy, exact, nucleus)."""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import numpy as np


def top(half: np.ndarray, k: int) -> np.ndarray:
    """Columns of the best k: value descending, ties to the lower column."""

    order = np.lexsort((np.arange(half.size), -(half + np.float32(0.0))))
    return order[:k]


def candidates(row: np.ndarray, world: int, k: int, count: int):
    halves = np.array_split(np.arange(row.size), world)
    vals, ids = [], []
    for h in halves:
        cols = top(row[h], min(k, h.size))
        vals.append(row[h][cols])
        ids.append(h[cols])
    v, i = np.concatenate(vals), np.concatenate(ids).astype(np.int64)
    order = np.lexsort((i, -(v + np.float32(0.0))))[:count]
    return v[order], i[order]


def bf16(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-vocab", type=int, default=129280, help="smaller rows only (the committed fixture)")
    args = ap.parse_args()
    sys.path.insert(0, str(args.src))
    from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
    from tensorfold.families.deepseek_v41.cuda import nucleus as NU

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    full_rows = 0
    with open(args.out, "w") as f:
        for n in range(args.rows):
            vocab = rng.choice([64, 1000, 4096, 4096, 129280]) if n % 40 else 129280
            vocab = min(vocab, rng.choice([64, 600, args.max_vocab])) if vocab > args.max_vocab else vocab
            kind = rng.random()
            if kind < 0.1:
                row = np.full(vocab, rng.choice([0.0, 1.5, -3.0]), dtype=np.float32)         # flat: ties everywhere
            else:
                row = (nrng.standard_normal(vocab) * rng.choice([0.5, 2.0, 4.0, 8.0])).astype(np.float32)
                if rng.random() < 0.5:
                    row[rng.randrange(vocab)] += rng.choice([5.0, 15.0, 40.0])
                if rng.random() < 0.7:
                    row = bf16(row)
            world = rng.choice([1, 2, 2])
            temp = rng.choice([0.0, 0.3, 0.6, 0.7, 1.0, 1.0, 1.3])
            s = None
            if temp > 0:
                s = Sampling(rng.randrange(1 << 63), temp, rng.choice([0, 0, 0, 1, 20, 40]),
                             rng.choice([1.0, 0.95, 0.95, 0.8, 0.5, 0.99]), rng.choice([0.0, 0.0, 0.05]))
            position = rng.randrange(0, 1 << 20)
            nuc = NU.spec(s)
            if s is None:
                count = 1 + MARGIN
            elif s.top_k:
                count = s.top_k + MARGIN
            else:
                count = NU.DEFAULT if nuc else vocab
            count = min(count, vocab)
            k = count                                           # each rank's top count, as cand_gather asks
            vals, ids = candidates(row, world, k, count)
            stats, full = [], False
            if s is None:
                token = int(ids[0])
            elif nuc:
                t = max(float(s.temperature), 1e-6)
                for h in np.array_split(np.arange(vocab), world):
                    v = row[h].astype(np.float64)
                    m = v.max()
                    stats.append([float(m), float(np.exp(v / t - m / t).sum())])
                zm, z = NU.normalizer(np.asarray(stats), s.temperature)
                keep = NU.keep_count(vals, ids, nuc, zm, z)
                if keep is None:
                    full = True
                    fv, fi = candidates(row, world, vocab, vocab)
                    token = NU.choose(vals[None], ids[None], [position], s, {0: ("full", fi, fv)})[0]
                else:
                    token = NU.choose(vals[None], ids[None], [position], s, {0: ("keep", keep)})[0]
            else:
                token = choose_rows(vals[None].astype(np.float32), ids[None], [position], s)[0]
            full_rows += full
            f.write(json.dumps({"logits": base64.b64encode(row.astype("<f4").tobytes()).decode(), "world": world,
                                "sampling": None if s is None else [s.seed, s.temperature, s.top_k, s.top_p, s.min_p],
                                "position": position, "count": count, "stats": stats, "full": full,
                                "token": int(token)}) + "\n")
    print(f"{args.rows} rows ({full_rows} nucleus rows fell back to the whole row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
