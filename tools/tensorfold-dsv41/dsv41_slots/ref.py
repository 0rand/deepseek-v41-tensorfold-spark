#!/usr/bin/env python3
"""Slots gate reference: TP=2, S live slots prefilled alone then decoded greedily together, one trace line a prompt."""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZT = HERE.parent                      # the tools root: dsv41_triton_capture, triton_aot_manifest


def rank_main(rank: int, a) -> None:
    sys.path.insert(0, str(ZT))
    import torch

    gpu = 0 if a.rank >= 0 else rank          # two hosts: each host's one GPU
    torch.cuda.set_device(gpu)
    from dsv41_triton_capture import IndexRows, pack_aot, specialization
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import pool as PO
    from tensorfold.families.deepseek_v41.cuda import rope as RO
    from tensorfold.families.deepseek_v41.cuda import rowgraphs as RG
    from tensorfold.families.deepseek_v41.cuda import rowtab as RT
    from tensorfold.families.deepseek_v41.cuda import topology as TO
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Seg, greedy
    from tensorfold.families.deepseek_v41.cuda.decode import DecodeForward    # serving's SlotsForward (Engram hooks)
    kvpool = PO.kvpool

    t0 = time.time()
    dev = torch.device(f"cuda:{gpu}")
    comm = open_comm(rank, 2, a.master, a.port)
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
    S = a.slots
    fw = DecodeForward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                      limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=S,
                      contiguous=False, pchunk=2048)
    page = PO.PAGE
    maxp = kvpool.pages_for(a.limit, page)
    index_kv = os.environ.get("TF_DSV41_INDEX_KV", "bf16").strip() or "bf16"
    # split / rank only where the tree's Pool has them (split KV trees); an unsplit pool is the default otherwise
    import inspect
    pkw = {k: v for k, v in (("split", 0), ("rank", rank)) if k in inspect.signature(PO.Pool.__init__).parameters}
    pool = PO.Pool(TO.load(pack), (S * maxp + 4) * page, index_kv=index_kv, device=dev, **pkw)
    pss = [pool.new_slot(a.limit) for _ in range(S)]
    fw.bind(pss)
    load_s = time.time() - t0
    import triton_aot_manifest as aot

    dumps = rank == 0 or a.rank >= 0          # two hosts: each host keeps its own AOT set (the same variants)
    rec = aot.Recorder().install() if dumps else None
    rg = RG.RowGraphs(fw) if a.graphs else None
    eos = int(getattr(cfg, "eos_token_id", 1) or 1)
    out = Path(a.out)
    # the prompts, one a slot, each alone from its empty slot
    g = torch.Generator().manual_seed(a.seed)
    prompts = [torch.randint(3, cfg.vocab_size, (n,), generator=g).tolist() for n in a.prompts]
    toks, live = [], []
    for s, ids in enumerate(prompts):
        pss[s].ensure(len(ids))
        lg = fw.prompt(ids, slot=s)
        toks.append([int(greedy(fw, lg)[0])])
        live.append(toks[s][-1] != eos)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"{S} prompts prefilled at {time.time() - t0:.0f} s, first tokens {[t[0] for t in toks]}", flush=True)
    # greedy decoding, every live slot in one run a step
    for step in range(a.steps):
        segs = [Seg(s, fw.slots[s].pos, (toks[s][-1],)) for s in range(S) if live[s]]
        if not segs:
            break
        for sg in segs:
            pss[sg.slot].ensure(sg.start + 1)
        lg = rg.run(segs) if rg is not None else fw.run(segs)
        picks = [int(x) for x in greedy(fw, lg[:len(segs)])]
        for sg, t in zip(segs, picks):
            fw.keep(sg.slot, 0)
            toks[sg.slot].append(t)
            if t == eos or len(toks[sg.slot]) > a.steps:
                live[sg.slot] = False
    torch.cuda.synchronize()
    run_s = time.time() - t0 - load_s
    # row-mode runs of every bucket (their Triton variants for the AOT set; nothing recorded comes from them)
    if rg is not None:
        for R in [b for b in RT.BUCKETS if b <= a.rows_cap]:
            k = min(S, R)
            sizes = [R // k + (1 if i < R % k else 0) for i in range(k)]
            if max(sizes) > RT.SEG_MAX:
                continue
            mix = [Seg(s, fw.slots[s].pos, tuple([11] * n)) for s, n in enumerate(sizes)]
            for sg in mix:
                pss[sg.slot].ensure(sg.start + len(sg.ids))
            key = RT.row_key([(sg.slot, sg.start, len(sg.ids)) for sg in mix], rg.cache.bucket, fw.limit, rg.buckets)
            if key is not None:
                rg.rows.run(key, mix)
            for sg in mix:
                fw.keep(sg.slot, len(sg.ids) - 1)
        torch.cuda.synchronize()
    if dumps:
        cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
        tabs = RO.tables(cfg, a.limit + 2048)
        (out / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<II", *tabs[0].cs.shape)
                                       + tabs[0].cs.contiguous().numpy().tobytes()
                                       + tabs[1].cs.contiguous().numpy().tobytes())
        rec.dump(out / "launches.json")
        aot.build(cache, out / "launches.json", out / "manifest.json", None)
        pack_aot(out / "manifest.json", cache, out / "aot", specialization())
    mine = {"rank": rank, "load_s": round(load_s, 1), "run_s": round(run_s, 1), "tokens": toks, "prompts": prompts,
            "graphs": bool(rg is not None)}
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")
    comm.barrier()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--prompts", default="45,173,9,640", help="prompt lengths, one a slot (random ids, --seed)")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--limit", type=int, default=4096, help="a slot's positions (the Zig engine's opts.limit)")
    ap.add_argument("--rows-cap", type=int, default=16, help="row-mode buckets warmed for the AOT set")
    ap.add_argument("--port", type=int, default=29640)
    ap.add_argument("--rank", type=int, default=-1, help="two hosts: this host's rank (one process, GPU 0); -1: both here")
    ap.add_argument("--master", default="127.0.0.1", help="rank 0's address (two hosts: the head's CX7 address)")
    ap.add_argument("--graphs", dest="graphs", action="store_true", default=True)
    ap.add_argument("--no-graphs", dest="graphs", action="store_false")
    a = ap.parse_args()
    a.prompts = [int(x) for x in a.prompts.split(",")]
    if len(a.prompts) != a.slots:
        ap.error(f"{len(a.prompts)} prompts for {a.slots} slots")
    if a.graphs:
        os.environ.setdefault("TF_DSV41_GRAPHS", "1")
        os.environ.setdefault("TF_DSV41_GRAPH_MODE", "rows")
    os.environ.pop("TF_DSV41_KV_SPLIT", None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.rank >= 0:                         # two hosts: this host's rank alone; its tokens, AOT set and rope tables
        rank_main(a.rank, a)
        mine = json.loads((out / f"rank{a.rank}.json").read_text())
        print(json.dumps({"rank": a.rank, "slots": a.slots, "tokens": [len(t) for t in mine["tokens"]]}), flush=True)
        return 0
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    r0 = json.loads((out / "rank0.json").read_text())
    r1 = json.loads((out / "rank1.json").read_text())
    if r0["tokens"] != r1["tokens"]:
        print("the ranks' greedy tokens differ", flush=True)
        return 1
    with open(out / "trace.jsonl", "w") as f:
        for s, (p, t) in enumerate(zip(r0["prompts"], r0["tokens"])):
            f.write(json.dumps({"slot": s, "prompt": p, "tokens": t}) + "\n")
    meta = {"layers": a.layers, "slots": a.slots, "prompts": a.prompts, "steps": a.steps, "seed": a.seed,
            "limit": a.limit, "graphs": a.graphs, "tokens": r0["tokens"],
            "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")},
            "ranks": [{k: r[k] for k in ("load_s", "run_s")} for r in (r0, r1)]}
    (out / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"slots": a.slots, "tokens": [len(t) for t in r0["tokens"]], "graphs": a.graphs}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
