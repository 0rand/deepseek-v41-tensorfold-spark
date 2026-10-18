"""oracle.py's glue fixtures (plan, top-k, pointwise, prefill glue, window glue, mHC decode, KV store) and save."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tensorfold.families.deepseek_v41.cuda import x3gm as gm

DEV = torch.device("cuda:0")


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        (out / f"{name}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def plan_case(out: Path, g) -> dict:
    """x3gm.plan at full sizes (2,048 / 4,096-row blocks, 384 experts) with pairs past the table, for x3gm_plan.cu."""
    cases = [(2048, 6, 384, 64, 0.0), (4096, 6, 384, 128, 0.3), (1, 6, 384, 64, 0.0), (2048, 6, 128, 64, 0.5),
             (3, 7, 13, 64, 0.0), (700, 6, 384, 128, 0.9)]
    arrays, params = {}, {"kind": "plan", "cases": len(cases)}
    for i, (R, S, E, bm, past) in enumerate(cases):
        pk = torch.randint(0, E, (R, S), generator=g, dtype=torch.int32)
        pk[torch.rand(R, S, generator=g) < past] = E
        pk = pk.to(DEV)
        order, pe, poff, pcnt, npass = gm.plan(pk, E, bm)
        params.update({f"c{i}_E": E, f"c{i}_bm": bm, f"c{i}_P": R * S})
        arrays.update({f"c{i}_pk": pk, f"c{i}_order": order, f"c{i}_pe": pe, f"c{i}_poff": poff, f"c{i}_pcnt": pcnt,
                       f"c{i}_npass": npass})
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"cases": len(cases)}


def topk_case(out: Path, g) -> dict:
    """pick.top on a rank's vocabulary half (64,640 columns) of bf16-valued logits with many exact ties."""
    from tensorfold.families.deepseek_v41.cuda import pick
    R, C = 6, 64640
    lg = (torch.randint(-96, 97, (R, C), generator=g).float() / 16.0)          # bf16-exact, heavy ties
    lg[1] = torch.randn(C, generator=g).to(torch.bfloat16).float()               # a realistic row
    lg[2, ::7] = 6.0                                                              # one value at many columns
    lg[3] = -0.0
    lg = lg.to(DEV)
    arrays, params = {"lg": lg}, {"kind": "topk", "R": R, "C": C, "ks": 4}
    for i, k in enumerate((1, 5, 64, 1024)):
        vals, cols = pick.top(lg, k)
        arrays.update({f"vals{i}": vals, f"cols{i}": cols})
        params[f"k{i}"] = k
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"rows": R}


def pointwise_case(out: Path, g) -> dict:
    """prefill_moe's torch pointwise ops (clamps / silu / mul, adds) on edge values: past the limit, NaN, inf, -0.0."""
    R, n, limit = 7, 2304, 10.0
    gg = torch.randn(R, n, generator=g) * 8
    uu = torch.randn(R, n, generator=g) * 8
    gg[0, :6] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 100.0, -100.0])
    uu[0, :6] = torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf"), -0.0])
    gg, uu = gg.to(DEV), uu.to(DEV)
    act = torch.nn.functional.silu(gg.clone().clamp_(max=limit)).mul_(uu.clone().clamp_(min=-limit, max=limit))
    a = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    b = (torch.randn(R, 5120, generator=g) * 3).to(DEV)
    s32 = a.clone()
    s32 += b
    y16 = torch.empty((R, 5120), dtype=torch.bfloat16, device=DEV)
    torch.add(a, b, out=y16)
    torch.cuda.synchronize()
    save(out, {"kind": "pointwise", "R": R, "n": n, "D": 5120, "limit": limit},
         {"g": gg, "u": uu, "act": act, "a": a, "b": b, "sum32": s32, "sum16": y16})
    return {"rows": R}

KEY_NONE = -(2 ** 63)


def top_positions(k: torch.Tensor, count: int) -> torch.Tensor:
    """index.top_positions, verbatim."""
    R, n = k.shape
    top = torch.topk(k, min(count, n), dim=1, sorted=False).values
    pos = torch.where(top == KEY_NONE, torch.full_like(top, -1), 0x7FFFFFFF - (top & 0xFFFFFFFF))
    pos = torch.where(pos < 0, torch.full_like(pos, 2 ** 31 - 1), pos)
    pos = torch.sort(pos, dim=1).values
    pos = torch.where(pos == 2 ** 31 - 1, torch.full_like(pos, -1), pos)
    if pos.shape[1] < count:
        pos = torch.cat([pos, torch.full((R, count - pos.shape[1]), -1, dtype=pos.dtype, device=pos.device)], 1)
    return pos.to(torch.int32).contiguous()


def pfglue_case(out: Path, g) -> dict:
    """prefill_glue.cu's references: block_prefill.zig's glue steps as the reference runs them, on edge inputs."""
    arrays, params = {}, {"kind": "pfglue"}
    # top_positions: unique keys as sort_keys builds them, KEY_NONE pads, negative positions, short and long rows
    tp = [(4, 3000, 512), (3, 300, 512), (2, 20000, 2048), (2, 1500, 2048)]
    for i, (R, n, count) in enumerate(tp):
        hi = torch.randint(-(2 ** 31), 2 ** 31, (R, n), generator=g, dtype=torch.int64)
        hi[:, : n // 7] = hi[0, 0]                                    # many equal scores: the positions decide
        lo = torch.stack([torch.randperm(n, generator=g) for _ in range(R)]).to(torch.int64)
        k = (hi << 32) | (0x7FFFFFFF - lo)
        k[:, 1::13] = KEY_NONE
        k[0, 2::29] = (hi[0, 2::29] << 32) | 0xFFFFFFF0               # low half past 0x7FFFFFFF
        k = k.to(DEV)
        arrays[f"tp{i}_keys"] = k
        arrays[f"tp{i}_want"] = top_positions(k, count)
        params.update({f"tp{i}_R": R, f"tp{i}_n": n, f"tp{i}_count": count})
    params["tp"] = len(tp)
    # candidate_keys: ascending blocks, -1 padded
    R, nb, bs = 3, 2048, 8
    blocks = torch.sort(torch.randint(0, 50000, (R, nb), generator=g), dim=1).values.to(torch.int32)
    blocks[:, 1900:] = -1
    blocks = blocks.to(DEV)
    j = blocks.to(torch.int64)[:, :, None] * bs + torch.arange(bs, device=DEV)
    j = torch.where(blocks[:, :, None] >= 0, j, torch.full_like(j, -1))
    arrays.update(ck_blocks=blocks, ck_want=j.reshape(R, -1).to(torch.int32).contiguous())
    params.update(ck_R=R, ck_nb=nb, ck_bs=bs)
    # visible_counts over top_positions' selections, ratios 4 and 128
    sel = arrays["tp0_want"]
    pos = torch.tensor([0, 1500, 9000, 2 ** 20], dtype=torch.int64, device=DEV)
    for ratio in (4, 128):
        nvis = (pos + 1) // ratio
        arrays[f"vc{ratio}_want"] = ((sel >= 0) & (sel.to(torch.long) < nvis[:, None])).sum(1).to(torch.int32)
    arrays["vc_pos"] = pos
    # the prefill MoE's picks: 6 routed slots + the shared one, kit weights through fp16 (edges included)
    n, slots, topk = 37, 7, 6
    pick = torch.randint(0, 384, (n, slots), generator=g, dtype=torch.int32)
    pick[:, topk:] = 384
    wts = torch.rand(n, slots, generator=g) * 2
    wts[0, :6] = torch.tensor([65520.0, 1e-8, 6.1e-5, float("nan"), -0.0, 1.0 + 2 ** -11])
    pick, wts = pick.to(DEV), wts.to(DEV)
    arrays.update(gp_pick=pick, gp_wts=wts, gp_pk=pick[:, :topk].contiguous(),
                  gp_w6=wts.to(torch.float16).to(torch.float32)[:, :topk].contiguous())
    params.update(gp_n=n, gp_slots=slots, gp_topk=topk)
    # x3gm._run_ragged's width mask (picks of E: the masked shared slot)
    E, P = 384, 999
    tab = torch.tensor([[4, 6, 8, 10][i % 4] for i in range(E)], dtype=torch.int32)
    pk = torch.randint(0, E + 1, (P,), generator=g, dtype=torch.int32)
    tab, pk = tab.to(DEV), pk.to(DEV)
    k2w = torch.cat([tab, torch.zeros(1, dtype=torch.int32, device=DEV)])
    for k2 in (4, 10):
        arrays[f"wm{k2}_want"] = torch.where(k2w[pk.long()] == k2, pk, torch.full_like(pk, E))
    arrays.update(wm_pick=pk, wm_tab=tab)
    params.update(wm_E=E, wm_P=P)
    # the indexer head weights: fp64 -> bf16 -> fp32 (double rounding cases, NaN, inf, overflow, denormals)
    w = torch.randn(4096, generator=g, dtype=torch.float64) * 3
    w[:12] = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e300, -1e300, 1e-300, 4e-39,
                           1.0 + 2 ** -8 + 2 ** -30, 1.0 + 2 ** -8 - 2 ** -30, 1.0 + 2 ** -8, 3.4e38, -0.0],
                          dtype=torch.float64)
    w = w.to(DEV)
    arrays.update(f64_in=w, f64_want=w.to(torch.bfloat16).to(torch.float32))
    # blocks._projection: [kv | gate] fp32 and kv alone
    n2, hd = 33, 512
    kv = torch.randn(n2, hd, generator=g).to(torch.bfloat16).to(DEV)
    gate = torch.randn(n2, hd, generator=g).to(torch.bfloat16).to(DEV)
    arrays.update(pj_kv=kv, pj_gate=gate, pj_two=torch.cat([kv.float(), gate.float()], 1).contiguous(),
                  pj_one=kv.float().contiguous())
    params.update(pj_n=n2, pj_hd=hd)
    # blocks.stage_rows: a window of rows between rings of different sizes (4-byte and byte rows)
    for i, (sr, dr, rb, lo, hi) in enumerate([(4096, 256, 512, 1500, 1627), (256, 4096, 6, 9000, 9128)]):
        src = torch.randint(0, 256, (sr, rb), generator=g, dtype=torch.uint8).to(DEV)
        dst = torch.randint(0, 256, (dr, rb), generator=g, dtype=torch.uint8).to(DEV)
        want = dst.clone()
        p = torch.arange(lo, hi, dtype=torch.long, device=DEV)
        want[p % dr] = src[p % sr]
        arrays.update({f"rc{i}_src": src, f"rc{i}_dst": dst, f"rc{i}_want": want})
        params.update({f"rc{i}_sr": sr, f"rc{i}_dr": dr, f"rc{i}_rb": rb, f"rc{i}_lo": lo, f"rc{i}_hi": hi})
    torch.cuda.synchronize()
    save(out, params, arrays)
    return {"cases": 8}


def glue_case(out: Path, g) -> dict:
    """The window's torch glue on edge values: out-of-range ids, -0 / +0, NaN, inf, halfway and denormal values."""
    V, D, lo, W, n = 200, 512, 1000, 3, 7
    embed = torch.randn(V, D, generator=g).to(torch.bfloat16)
    embed[3, :4] = torch.tensor([-0.0, float("nan"), float("inf"), 1e-40]).to(torch.bfloat16)
    embed = embed.to(DEV)
    ids = torch.tensor([1003, 999, 1199, 1200, 1000, 5, 1003], dtype=torch.long, device=DEV)
    local = ids - lo                                                       # forward.embed_rows, then the bf16 send
    ok = (local >= 0) & (local < V)
    rows = embed.index_select(0, local.clamp(0, V - 1)).to(torch.float32)
    rows = torch.where(ok[:, None], rows, torch.zeros_like(rows))
    send = rows.to(torch.bfloat16)
    recv = torch.randn(W, n, D, generator=g).to(torch.bfloat16)
    recv[0, :, :8] = -0.0
    recv[1, :, :4] = 0.0
    recv[2, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")]).to(torch.bfloat16)
    recv = recv.to(DEV)
    parts = recv.float()                                                   # forward.allsum (parts fp32, rank order)
    acc = parts[0].clone()
    for r in range(1, W):
        acc += parts[r]
    summed = acc.to(torch.bfloat16).repeat(1, 4).contiguous()
    x = torch.randn(n, D, generator=g) * 3
    x[0, :8] = torch.tensor([float("nan"), float("inf"), -0.0, 1e-41, 70000.0, -70000.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9])
    x[1, :4] = torch.tensor([2 ** -20, 6.1e-5, 5.9e-8, 1.0 + 2 ** -11])
    x = x.to(DEV)
    cast = x.to(torch.bfloat16)
    kw = x.to(torch.float16).to(torch.float32)                             # pick.kit_weights
    kl = x.to(torch.bfloat16).to(x.dtype)                                  # pick.kit_logits
    gc = torch.randn(W, n, 24, generator=g).to(torch.bfloat16).to(DEV)
    cols = gc.view(W, n, 24).permute(1, 0, 2).reshape(n, -1).contiguous()  # forward.gather_cols
    src = torch.randn(5, 1100, generator=g).to(torch.bfloat16).to(DEV)[:, :1024]
    carry = src[2].to(torch.float32).contiguous()
    torch.cuda.synchronize()
    save(out, {"kind": "glue", "V": V, "D": D, "lo": lo, "W": W, "n": n, "c": 24, "ld": 1100, "row": 2, "start": 3000},
         {"embed": embed, "ids": ids, "send": send, "recv": recv, "summed": summed, "x": x, "cast": cast, "kw": kw,
          "kl": kl, "gc": gc, "cols": cols, "src": src.as_strided((5, 1100), (1100, 1)),   # the whole [5, 1100] rows (row stride 1,100)
          "carry": carry})
    return {"rows": n}


def mhcdec_case(out: Path, g, *, R: int, mode: str, tap: bool = False, in_place: bool = False) -> dict:
    """An mHC site of a decode window on the Triton decode path: TF_DSV41_MHC_PFDEC's reference (D 5,120, 2 ranks)."""
    from tensorfold.families.deepseek_v41.cuda import mhc
    D, W = 5120, 2
    bf, f32 = torch.bfloat16, torch.float32
    x = torch.randn(R, 4 * D, generator=g) * 2
    x[0, :4] = torch.tensor([-0.0, 0.0, 3.0e4, -2.5e-30])
    x = x.to(bf).to(DEV)
    gathered = (torch.randn(W, R, D, generator=g) * 0.5).to(bf).to(DEV)
    sig = lambda t: torch.sigmoid(t)
    ppre = (sig(torch.randn(R, 4, generator=g)) + 1e-6).to(f32).to(DEV)
    ppost = (2 * sig(torch.randn(R, 4, generator=g))).to(f32).to(DEV)
    pcomb = torch.softmax(torch.randn(R, 4, 4, generator=g), -1).reshape(R, 16).to(f32).to(DEV)
    fn = (torch.randn(24, 4 * D, generator=g) * 0.02).to(bf).to(DEV)
    base = (torch.randn(24, generator=g) * 0.5).to(f32).to(DEV)
    scale = (torch.rand(3, generator=g) + 0.05).to(f32).to(DEV)
    nw = (1 + 0.1 * torch.randn(D, generator=g)).to(f32).to(DEV)
    xout0 = torch.full((R, 4 * D), -7.0, dtype=bf, device=DEV)
    taps0 = torch.full((R, 3 * D), 5.0, dtype=bf, device=DEV)       # DSpark's tap buffer: column block 1 is this site's
    out_ = torch.full((R, D), -3.0, dtype=bf, device=DEV)
    scratch = mhc.Scratch(R, DEV)
    scratch.part.fill_(0.0)
    coefs = mhc.Coefs(R, DEV)
    prev = mhc.Coefs(R, DEV, pre=ppre.clone(), post=ppost.clone(), comb=pcomb.clone())
    hc = mhc.Hc(fn, base, scale)
    xs = x.clone()
    xout = xs if in_place else xout0.clone()
    taps = taps0.clone()
    saved = {k: os.environ.get(k) for k in ("TF_DSV41_MHC_CUDA", "TF_DSV41_MHC_PF")}
    os.environ["TF_DSV41_MHC_CUDA"] = "0"
    os.environ["TF_DSV41_MHC_PF"] = "0"
    try:
        if mode == "boundary":
            mhc.boundary(xs, xout, gathered, prev, hc, nw, out_, coefs, scratch, taps[:, D:2 * D] if tap else None)
        else:
            mhc.site(xs, hc, nw, out_, coefs, scratch, ppre if mode == "site2" else None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    torch.cuda.synchronize()
    arrays = {"x": x, "xout0": xout0, "g": gathered, "ppre": ppre, "ppost": ppost, "pcomb": pcomb, "fn": fn,
              "base": base, "scale": scale, "nw": nw, "taps0": taps0, "out0": torch.full((R, D), -3.0, dtype=bf),
              "xs": xs, "xout": xout, "taps": taps, "part": scratch.part, "c": scratch.c, "out": out_,
              "npre": coefs.pre, "npost": coefs.post, "ncomb": coefs.comb}
    save(out, {"kind": "mhcdec", "R": R, "D": D, "W": W, "mode": mhc_mode[mode], "tap": int(tap),
               "in_place": int(in_place), "eps": mhc.EPS, "hc_eps": mhc.HC_EPS, "post_alpha": mhc.POST_ALPHA,
               "iters": mhc.ITERS}, arrays)
    return {"rows": R, "mode": mode, "tap": tap, "in_place": in_place}


mhc_mode = {"boundary": 0, "site1": 1, "site2": 2}
