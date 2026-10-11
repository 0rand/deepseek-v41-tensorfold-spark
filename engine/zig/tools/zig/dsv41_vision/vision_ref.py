#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The vision gate's Python reference (gate): prod's engine (8474f31, /dsv41-tf/src) on one load at TP=2.

1. Images: ``IMAGES`` generated with Pillow (seeded), each through prod's ``vision_prep`` (decode, preprocess, virtual
   ids) into VDIR: ``<name>.patches.bin`` (bf16) and ``manifest.json`` (grids, digest, ids), the Zig side's
   TF_DSV41_VISION_HOLD format.
2. The tower (rank 0, ``vision.Tower`` from the pack's vision tensors on this GPU): each image's stages as
   ``vision.Tower.vit`` / ``align`` compute them (the same helpers, so the same bits; checked against ``Tower.span``)
   into ``<name>.<stage>.bin`` (embed, block0, block1, block15, block31, vit, al.unfold, al.w1, gelu) and
   ``<name>.span.bin``; block 0's every op as ``b0.<op>``.
3. ``PROMPTS``: text ids around the images' spans of virtual ids; rank 0 holds the images' rows in ``vision.STORE``,
   then dsv41_m2b_ref.py's own ``one`` runs each prompt from a fresh slot (state dump, greedy steps, digests) into
   OUT/p<i>/ (``--prompts 1,2,3``: prompt i is ``PROMPTS[i - 1]``), so tf-dsv41-m1 m2b's prefill gate reads them as
   any M2b reference; ``OUT/trace.jsonl`` holds every prompt and its tokens for ``tf-dsv41-m1 generate``.
4. ``--prompts 1,2,3,4,5,6`` with ``--bias-vl DIR``: prompts 4..6 are 1..3 again (the same ids) after
   ``vision.attach_bias_vl(fw.blocks, DIR)`` on both ranks: image rows route with ``gate.bias_vl`` in their own MoE
   call (``vision.moe``), as prod with TF_DSV41_BIAS_VL; ``OUT/trace-bias.jsonl``.

Prompts 1..3 route image rows with the text bias (as Python without TF_DSV41_BIAS_VL).

    python -u -B vision_ref.py --vdir VDIR [--bias-vl DIR] -- --pack PACK --out OUT --layers 0-24 --steps 32 --prompts 1,2,3,4,5,6

Two hosts (the Sparks, spark.sh): the same with ``--rank R --master ADDR`` on each; each host prepares the images
(seeded, the same bits) and rank 0 alone dumps the tower.

``--tower-only``: steps 1 and 2 alone on one GPU (cuda:0; the pack's vision tensors, no model load, about 2 min), for
``tf-dsv41-m1 vision PACK VDIR`` on the same GPU:

    python -u -B vision_ref.py --tower-only --vdir VDIR --pack PACK [--sweep] [--light]

``--sweep``: ``SWEEP``'s images instead (every aligner row count regime: L 169 .. 1,021, strips, tile edges, every
unfold padding); ``--light``: no ``b0.*`` dumps (a sweep's disk).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # tools/zig: dsv41_m2b_ref and its helpers
import dsv41_m2b_ref as M2B  # noqa: E402

IMAGES = {  # name -> (width, height, format, mode)
    "photo": (640, 480, "JPEG", "RGB"),
    "wide": (1280, 400, "PNG", "RGB"),
    "small": (200, 150, "PNG", "RGBA"),
}
# the tower sweep (--sweep): name "<llm h>x<llm w>p<unfold's zero rows><zero cols>" -> (width, height), PNG RGB
SWEEP = {
    "13x13p00": (546, 546), "1x211p01": (8848, 42), "1x257p12": (10766, 28), "1x509p02": (21350, 42),
    "1x641p11": (26908, 28), "1x769p22": (32270, 14), "1x1019p00": (42798, 42), "1x1021p00": (42882, 42),
    "211x1p10": (42, 8848), "257x1p21": (28, 10766), "509x1p02": (14, 21378), "511x1p00": (42, 21462),
    "16x16p12": (644, 658), "17x15p21": (616, 686), "16x24p01": (994, 672), "16x32p20": (1344, 644),
    "32x16p10": (672, 1330), "20x32p12": (1316, 826), "24x32p22": (1316, 980), "28x32p01": (1330, 1176),
    "30x32p10": (1344, 1246), "31x31p00": (1302, 1302), "31x31p22": (1274, 1274), "27x36p00": (1512, 1134),
    "36x27p00": (1134, 1512), "29x33p21": (1372, 1190), "23x41p10": (1722, 952),
}
SWEEP_IMAGES = {**IMAGES, **{k: (w, h, "PNG", "RGB") for k, (w, h) in SWEEP.items()}}
# each prompt: text lengths and image names in order (ints: that many random text ids)
PROMPTS = [[37, "photo", 21], [15, "wide", 9, "small", 30], ["small", 12]]
STAGES = ("embed", "block0", "block1", "block15", "block31", "vit", "gelu")


def make_image(name: str):
    import numpy as np
    from PIL import Image

    w, h, fmt, mode = SWEEP_IMAGES[name]
    rng = np.random.default_rng(sum(map(ord, name)))
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 3) % 256], -1)
    for _ in range(6):                          # a few flat boxes (screenshot-like edges)
        x0, y0 = rng.integers(0, w // 2), rng.integers(0, h // 2)
        base[y0:y0 + h // 4, x0:x0 + w // 4] = rng.integers(0, 256, 3)
    arr = (base + rng.integers(0, 24, base.shape)).clip(0, 255).astype("u1")
    img = Image.fromarray(arr, "RGB")
    if mode == "RGBA":
        img.putalpha(Image.fromarray((x * 255 // max(w - 1, 1)).astype("u1"), "L"))
    b = io.BytesIO()
    img.save(b, fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return b.getvalue()


def prepare(vdir: Path, pack: Path, names=IMAGES):
    """Step 1: every image's Prepared (prod's front end) and the manifest."""

    from tensorfold.families.deepseek_v41.cuda import vision_prep as VP

    env = {k: v for k, v in os.environ.items() if k != "TF_DSV41_BIAS_VL"}
    s = VP.Settings.read(pack, env)
    reg = VP.Registry()
    vdir.mkdir(parents=True, exist_ok=True)
    out, man = {}, []
    for name in names:
        raw = make_image(name)
        (vdir / f"{name}.img").write_bytes(raw)
        p = VP.preprocess(VP.decode(raw, s), s)
        p.vids = reg.vids(p.digest, p.tokens)
        (vdir / f"{name}.patches.bin").write_bytes(p.patches.contiguous().view(__import__("torch").int16).numpy().tobytes())
        man.append({"name": name, "vit_h": p.n_vit_h, "vit_w": p.n_vit_w, "llm_h": p.n_llm_h, "llm_w": p.n_llm_w,
                    "digest": p.digest.hex(), "vids": p.vids, "patches": f"{name}.patches.bin", "size": list(p.size)})
        out[name] = p
    (vdir / "manifest.json").write_text(json.dumps({"images": man}) + "\n")
    return out


def tower_dumps(vdir: Path, pack: Path, images: dict, dev, light: bool = False):
    """Step 2 (rank 0): the stages and span rows of every image; returns the encoder (its tower) for step 3."""

    import torch
    import torch.nn.functional as F

    from tensorfold.families.deepseek_v41.cuda import vision as V

    enc = V.Encoder.load(pack, dev)
    t = enc.tower
    for name, img in images.items():
        dumps = {}
        with torch.inference_mode():
            tt = t.t
            x = t.lin(img.patches.to(t.device, torch.bfloat16).flatten(1), "vision.patch_embed.proj")
            dumps["embed"] = x.clone()
            n, dim = x.shape
            hd = dim // t.heads
            cos, sin = V._rope(img.n_vit_h, img.n_vit_w, hd // 2, t.theta, x.device)
            for i in range(t.layers):
                p = f"vision.blocks.{i}"
                # block 0's every op as its own stage ("b0.<op>", the Zig tower's layouts)
                b0 = (lambda k, v: dumps.__setitem__(f"b0.{k}", v.clone())) if i == 0 else (lambda k, v: None)
                b0 = (lambda k, v: None) if light else b0
                h = V._rms(x, tt[f"{p}.norm1.weight"])
                b0("norm1", h)
                qkv = t.lin(h, f"{p}.attn.wqkv")
                b0("qkv", qkv)
                q, k, v = (a.view(n, t.heads, hd) for a in qkv.chunk(3, dim=-1))
                q, k = V._rotate(q, cos, sin), V._rotate(k, cos, sin)
                b0("rope", torch.stack([q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)]))
                o = V._sdpa(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
                o = o.transpose(0, 1).reshape(n, -1)
                b0("attn", o)
                wo = t.lin(o, f"{p}.attn.wo")
                b0("wo", wo)
                x = x + wo
                b0("res1", x)
                h = V._rms(x, tt[f"{p}.norm2.weight"])
                b0("norm2", h)
                w1 = t.lin(h, f"{p}.mlp.w1")
                b0("w1", w1)
                g, u = w1.chunk(2, dim=-1)
                su = F.silu(g) * u
                b0("silu", su)
                w2 = t.lin(su, f"{p}.mlp.w2")
                b0("w2", w2)
                x = x + w2
                if f"block{i}" in STAGES:
                    dumps[f"block{i}"] = x.clone()
            vit = V._rms(x, tt["vision.norm.weight"])
            dumps["vit"] = vit
            r = t.ratio
            y = vit.view(img.n_vit_h, img.n_vit_w, -1).permute(2, 0, 1)
            y = F.pad(y, (0, -img.n_vit_w % r, 0, -img.n_vit_h % r))
            y = F.unfold(y.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
            dumps["al.unfold"] = y.t()                 # F.unfold's [9D, L]; aligner.w1 reads its transposed view
            w1 = t.lin(y, "aligner.w1")
            dumps["al.w1"] = w1.clone()
            a1 = F.gelu(w1)
            dumps["gelu"] = a1
            span = t.span(img)
            staged = t.lin(a1, "aligner.w2")
            if not torch.equal(span[1:-1][torch.tensor(img.types[1:-1]) == 1], staged.to(torch.bfloat16)):
                raise SystemExit(f"{name}: the staged tower differs from Tower.span (the reference is not prod's)")
        for k, v in dumps.items():
            (vdir / f"{name}.{k}.bin").write_bytes(v.contiguous().view(torch.int16).cpu().numpy().tobytes())
        (vdir / f"{name}.span.bin").write_bytes(span.contiguous().view(torch.int16).cpu().numpy().tobytes())
        print(f"vision ref {name}: {img.n_vit_h} x {img.n_vit_w} patches, span {span.shape[0]} rows, "
              f"sha256 {hashlib.sha256(span.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]}", flush=True)
    return enc


_ORIG_ONE = M2B.one
_STATE: dict = {}


def prompt_ids(spec, images, vocab, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    ids = []
    for part in spec:
        ids.extend(images[part].vids if isinstance(part, str) else torch.randint(3, vocab, (part,), generator=g).tolist())
    return ids


def one_image(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, *rest):
    """dsv41_m2b_ref.one on prompt ``PROMPTS[prompt - 1]``: its ids in place of the random ones."""

    import torch

    from tensorfold.families.deepseek_v41.cuda import vision as V

    vdir, pack = Path(os.environ["DSV41_VISION_DIR"]), Path(a.pack)
    if "images" not in _STATE:
        two_hosts = a.rank >= 0                  # rank 1 on its own host: the same seeded images prepared there
        _STATE["images"] = prepare(vdir, pack) if rank == 0 or two_hosts else None
        fw.comm.barrier() if hasattr(fw.comm, "barrier") else None
        if rank != 0 and not two_hosts:
            _STATE["images"] = _load_manifest(vdir)
        if rank == 0:
            enc = tower_dumps(vdir, pack, _STATE["images"], fw.device)
            imgs = list(_STATE["images"].values())
            V.STORE.hold(enc.table(imgs))
            del enc.tower.t                      # the rows are in the store; the tower's weights go
    k = (prompt - 1) % len(PROMPTS) + 1                 # 4..6: 1..3 again, with the image routing bias
    if prompt > len(PROMPTS) and not _STATE.get("bias"):
        folder = os.environ.get("DSV41_VISION_BIAS", "")
        if not folder:
            raise SystemExit("vision ref: prompts past 3 need --bias-vl")
        print(f"rank {rank}: gate.bias_vl of {V.attach_bias_vl(fw.blocks, folder)} routers from {folder}", flush=True)
        _STATE["bias"] = True
    ids = prompt_ids(PROMPTS[k - 1], _STATE["images"], cfg.vocab_size, a.seed + k)
    real = torch.randint

    def once(*x, **k):                                       # one() draws its prompt with torch.randint, first
        torch.randint = real
        return torch.tensor(ids)

    torch.randint = once
    try:
        return _ORIG_ONE(rank, a, fw, cfg, greedy, Seg, digest, len(ids), out, *rest)
    finally:
        torch.randint = real


class _Vids:
    def __init__(self, vids):
        self.vids = vids


def _load_manifest(vdir: Path) -> dict:
    import time

    for _ in range(600):                          # rank 0 writes it before its tower runs
        f = vdir / "manifest.json"
        if f.exists():
            return {e["name"]: _Vids(e["vids"]) for e in json.loads(f.read_text())["images"]}
        time.sleep(1)
    raise SystemExit("vision ref: no manifest from rank 0")


M2B.one = one_image


def main() -> int:
    argv = sys.argv[1:]
    if "--tower-only" in argv:
        import torch

        vdir, pack = Path(argv[argv.index("--vdir") + 1]), Path(argv[argv.index("--pack") + 1])
        os.environ.pop("TF_DSV41_BIAS_VL", None)
        names = SWEEP_IMAGES if "--sweep" in argv else IMAGES
        tower_dumps(vdir, pack, prepare(vdir, pack, names), torch.device("cuda:0"), "--light" in argv)
        return 0
    if "--vdir" not in argv or "--" not in argv:
        raise SystemExit(__doc__)
    os.environ["DSV41_VISION_DIR"] = argv[argv.index("--vdir") + 1]
    if "--bias-vl" in argv:
        os.environ["DSV41_VISION_BIAS"] = argv[argv.index("--bias-vl") + 1]
    os.environ.pop("TF_DSV41_BIAS_VL", None)             # the loader takes none: prompts 4..6 attach it themselves
    sys.argv = [sys.argv[0]] + argv[argv.index("--") + 1:]
    rc = M2B.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    for name, lo in (("trace.jsonl", 1), ("trace-bias.jsonl", len(PROMPTS) + 1)):
        lines = []
        for i in range(lo, lo + len(PROMPTS)):
            if (out / f"p{i}" / "ref.json").exists():
                ref = json.loads((out / f"p{i}" / "ref.json").read_text())
                lines.append(json.dumps({"prompt": ref["prompt_ids"], "tokens": ref["tokens"]}))
        if lines:
            (out / name).write_text("\n".join(lines) + "\n")
            print(f"vision ref: {len(lines)} image prompts, {out / name}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
