#!/usr/bin/env python3
"""Sampled reference: TP=2 serial keyed sampling from fresh slots, one trace line per case (drafted == serial)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (seed, temperature, top_k, top_p, min_p): nucleus rows, the whole-vocabulary path (top_p 1), top-k, min-p
SAMPLINGS = [
    (4101, 0.6, 0, 0.95, 0.0),
    (77, 1.0, 0, 0.8, 0.0),
    ((1 << 63) + 12345, 1.3, 0, 0.95, 0.05),
    (9, 1.0, 20, 0.95, 0.0),
    (31337, 0.7, 0, 1.0, 0.0),
    (5, 1.0, 0, 0.9, 0.0),
]
PROMPTS = [37, 50, 66, 83, 128, 292]


def run_cases(fw, cfg, rank: int, out: Path, prompts, steps: int, seed: int = 4101, log=print) -> None:
    """Each prompt from a fresh slot, decoded serially with ``SAMPLINGS[i % len]``; rank 0 writes out/trace.jsonl."""

    import torch

    from tensorfold.engine.exact_sampling import MARGIN, Sampling
    from tensorfold.families.deepseek_v41.cuda import nucleus as NU
    from tensorfold.families.deepseek_v41.cuda.batch import Batcher
    from tensorfold.families.deepseek_v41.cuda.forward import Seg
    from tensorfold.families.deepseek_v41.cuda.slots import SlotsForward

    t0 = time.time()
    # the served forward's candidate path (slots.SlotsForward), bound to this forward: nucleus.finish calls it too
    for name in ("candidates", "cand_gather", "cand_fetch"):
        setattr(fw, name, types.MethodType(getattr(SlotsForward, name), fw))
    fw.cand_wait = SlotsForward.cand_wait
    nucleus = NU.candidates()

    def choose(lg, positions, s) -> tuple[list[int], int]:
        """``Batcher._decode``'s count, then ``_sample``'s path for one window (both ranks: the gathers pair up)."""
        nuc = NU.spec(s) if nucleus else []
        k = int(s.top_k) + MARGIN if s.top_k else (nucleus if nuc else fw.vocab)
        c = fw.candidates(lg, [lg.shape[0]], min(k, fw.vocab))
        if nuc:
            c = NU.finish(fw, lg, c, [lg.shape[0]], [nuc])
        whole = sum(1 for v in (getattr(c, "nucleus", None) or {}).values() if v[0] == "full")
        return Batcher._choose(None, c, positions, s), whole

    lines = []
    for n, plen in enumerate(prompts):
        seed_s, temp, top_k, top_p, min_p = SAMPLINGS[n % len(SAMPLINGS)]
        fw.slots[0] = fw.new_slot(contiguous=True)
        g = torch.Generator().manual_seed(seed + n)
        ids = torch.randint(3, cfg.vocab_size, (plen,), generator=g).tolist()
        s = Sampling(seed_s, temp, top_k, top_p, min_p)
        logits = fw.prompt(ids)
        toks, whole = choose(logits[-1:].float(), [len(ids)], s)
        tok = toks[0]
        tokens = [tok]
        for _ in range(steps - 1):
            start = fw.slot.pos
            lg = fw.run([Seg(0, start, (tok,))])
            fw.keep(0, 0)
            got, w = choose(lg.float(), [start + 1], s)
            whole += w
            tok = got[0]
            tokens.append(tok)
        lines.append({"prompt": ids, "tokens": tokens, "whole": whole,
                      "sampling": {"seed": seed_s, "temperature": temp, "top_k": top_k, "top_p": top_p, "min_p": min_p}})
        if rank == 0:
            log(f"sampled {n}: prompt {plen}, T {temp} top_k {top_k} top_p {top_p} min_p {min_p}: {len(tokens)} tokens"
                f" ({whole} rows whole), {len(set(tokens))} distinct, {time.time() - t0:.0f} s", flush=True)
    fw.slots[0] = fw.new_slot(contiguous=True)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        log(f"rank 0: {len(lines)} sampled replies into {out / 'trace.jsonl'}", flush=True)


def rank_main(rank: int, a) -> None:
    sys.path.insert(0, str(HERE))
    import torch

    torch.cuda.set_device(rank)
    from dsv41_triton_capture import IndexRows
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward

    t0 = time.time()
    dev = torch.device(f"cuda:{rank}")
    comm = open_comm(rank, 2, "127.0.0.1", a.port)
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, rank, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", rank * vh, (rank + 1) * vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = [loader.layer(L) for L in range(lo, hi + 1)]
    tree = W.RankW(rank, 2, rank * vh, embed, layers, norm, head, None, {})
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048)
    if rank == 0:
        print(f"loaded layers {lo}-{hi} in {time.time() - t0:.0f} s", flush=True)
    run_cases(fw, cfg, rank, Path(a.out), a.prompts, a.steps, a.seed)
    comm.barrier()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--prompts", type=lambda v: [int(x) for x in v.split(",") if x], default=PROMPTS,
                    help="prompt lengths, one sampled reply each (comma separated)")
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101, help="the prompts' ids")
    ap.add_argument("--port", type=int, default=29631)
    a = ap.parse_args()
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
