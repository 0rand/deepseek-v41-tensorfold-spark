#!/usr/bin/env python3
"""Check an EXL3 DeepSeek-V4.1-Flash pack (1017 layout) against what this recipe's engine loads, before booting it.

Reads ``model.safetensors.index.json``, ``config.json`` / ``quantization_config.json`` and each shard's safetensors
header; never the tensor data. Standard library only:

    python3 -I scripts/check_pack.py <model-dir>
    # before downloading: the index and configs in <model-dir>, each shard's header JSON in <headers-dir>
    python3 -I scripts/check_pack.py <model-dir> --headers <headers-dir>

It refuses (exit 1) what engine/zig (snapshot 9131e10a) would refuse at boot, or would load and run wrong:

- **Native tensor storage.** Each native tensor's dtype against its class in the engine's ``names.zig`` ``Store``:
  ``f32`` (mHC mixers ``hc_*``, attention sinks, router bias ``ffn.gate.bias``), ``norm`` (BF16 or F32), ``half``
  (BF16 or F16: router gate, indexer ``weights_proj``, embedding, Markov heads) and the DSpark confidence head (F32
  or BF16, ``plan.zig``). A router bias or confidence head stored F16 / BF16 is listed as *needs widening*:
  ``scripts/widen_pack.py`` widens exactly those to F32. An engine with the half-stored F32 natives loader change
  (TensorFold ``perf-1100/eaq-1``, store class ``f32_wide``) accepts them directly: pass ``--half-f32-natives``.
- **EXL3 groups.** int16 ``trellis`` [in/16, out/16, 16 K] with fp16 ``suh`` [in] / ``svh`` [out] and a ``mul1``
  marker. The engine's kernels are mul1 only and it never reads the codebook value, so a group without a ``mul1``
  marker (3-instruction default) or with an ``mcg`` one would decode with the wrong codebook; the declared
  ``codebook`` must be ``mul1`` too.
- **Routed experts' gate and up widths.** Prefill stores only the gate (w1) width for gate and up (w3); an expert
  whose widths differ would prefill wrong KV.
- **Coverage.** A tensor the engine neither reads nor ignores (vision tower, image routing bias, bf16 copies of EXL3
  groups) is refused by its loader.

It also lists the EXL3 widths (bits a weight) of each class, warns about widths outside the kernel tables of engine
9131e10a (a boot refusal, not a slow path), and reports the tensor bytes against flash17's
(203,755,654,072 B): at TP=2 each rank holds about half of the difference.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

INDEX = "model.safetensors.index.json"
FLASH17_BYTES = 203_755_654_072
GIB = 1 << 30
WIDTHS = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
GROUP_PARTS = ("trellis", "suh", "svh", "mul1", "mcg")
BLOCK = r"(?:layers|mtp)\.\d+\."

# (class, pattern, dtypes engine 9131e10a accepts, dtypes widen_pack.py turns into F32 for it)
CLASSES = [
    ("f32 mixer / sink", re.compile(BLOCK + r"(hc_(attn|ffn)_(fn|base|scale)|attn\.attn_sink)$"), {"F32"}, set()),
    ("f32 router bias", re.compile(BLOCK + r"ffn\.gate\.bias$"), {"F32"}, {"F16", "BF16"}),
    ("confidence head", re.compile(r"mtp\.\d+\.confidence_head\.proj\.weight$"), {"F32", "BF16"}, {"F16"}),
    (
        "norm",
        re.compile(
            BLOCK + r"(attn_norm|ffn_norm|attn\.q_norm|attn\.kv_norm|attn\.compressor\.norm"
            r"|attn\.indexer\.k_norm)\.weight$|" + BLOCK + r"engram\.[qk]_weight$|norm\.weight$"
            r"|mtp\.\d+\.(main_norm|norm)\.weight$"
        ),
        {"BF16", "F32"},
        set(),
    ),
    (
        "half",
        re.compile(
            BLOCK + r"(ffn\.gate|attn\.indexer\.weights_proj)\.weight$|embed\.weight$"
            r"|mtp\.\d+\.markov_head\.(embed|head)\.weight$"
        ),
        {"BF16", "F16"},
        set(),
    ),
]
# with the half-stored F32 natives loader change these classes also take F16 and BF16
WIDE_CLASSES = {"f32 router bias", "confidence head"}

# EXL3 width classes and the integer bits a weight engine 9131e10a has kernels for (a miss refuses to boot):
# routed and shared experts x3ld decode K2 2..12 and GM_V2 prefill K2 {3..8, 10}; dense3 K2 {4,6,8,10,12};
# upstream linear K2 {2..8, 10, 12, 14, 16}.
ROUTED_BITS = {2, 3, 4, 5}
EXPERT_BITS = {1, 2, 3, 4, 5, 6}
DENSE3_BITS = {2, 3, 4, 5, 6}
LINEAR_BITS = {1, 2, 3, 4, 5, 6, 7, 8}


class Refused(Exception):
    pass


def width_class(prefix: str) -> tuple[str, set[int]]:
    """The class of an EXL3 group (its stored prefix) and the widths the engine's kernels for it take."""
    if prefix == "head":
        return "lm head", LINEAR_BITS
    m = re.match(r"(layers|mtp)\.\d+\.(.*)$", prefix)
    if not m:
        return "other: " + prefix, LINEAR_BITS
    where, rest = ("dspark " if m.group(1) == "mtp" else ""), m.group(2)
    if rest.startswith("ffn.experts."):
        return where + "routed experts", ROUTED_BITS if not where else EXPERT_BITS
    if rest.startswith("ffn.shared_experts."):
        return where + "shared expert", EXPERT_BITS
    if rest == "main_proj":
        return "dspark main_proj", LINEAR_BITS
    if rest == "attn.indexer.wk":
        return "indexer wk", LINEAR_BITS
    if rest == "engram.wkv":
        return "engram wkv", LINEAR_BITS
    if rest.startswith("attn.indexer."):
        return "indexer wq_b", DENSE3_BITS
    if rest.startswith("attn.compressor."):
        return "compressor " + rest.rsplit(".", 1)[-1], DENSE3_BITS
    if rest.startswith("attn.wo_a."):
        return where + "attention wo_a", DENSE3_BITS
    if rest.startswith("attn."):
        return where + "attention " + rest[len("attn.") :], DENSE3_BITS
    return where + rest, LINEAR_BITS


def read_header(path: Path) -> dict:
    """A shard's header, checked against the file's size."""
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise Refused(f"{path.name}: shorter than a safetensors header")
        n = struct.unpack("<Q", prefix)[0]
        if not 2 <= n <= 256 << 20:
            raise Refused(f"{path.name}: header size {n} is not plausible")
        raw = f.read(n)
    header = json.loads(raw)
    end = max((t["data_offsets"][1] for k, t in header.items() if k != "__metadata__"), default=0)
    if path.stat().st_size != 8 + n + end:
        raise Refused(f"{path.name}: file size {path.stat().st_size} does not match its header ({8 + n + end})")
    return header


def load(model: Path, headers_dir: Path | None) -> tuple[dict, dict[str, dict], dict]:
    """(index, header of each shard, quantization settings) from the model folder or a folder of header JSON."""
    index = json.loads((model / INDEX).read_text())
    shards = sorted(set(index["weight_map"].values()))
    headers = {}
    for shard in shards:
        if "/" in shard or shard.startswith("."):
            raise Refused(f"index names an unsafe shard path {shard!r}")
        if headers_dir is not None:
            headers[shard] = json.loads((headers_dir / f"{shard}.header.json").read_text())
        else:
            headers[shard] = read_header(model / shard)
    quant = {}
    if (model / "quantization_config.json").is_file():
        quant = json.loads((model / "quantization_config.json").read_text())
    elif (model / "config.json").is_file():
        quant = json.loads((model / "config.json").read_text()).get("quantization_config") or {}
    return index, headers, quant


def check(index: dict, headers: dict[str, dict], quant: dict, half_f32_natives: bool = False) -> dict:
    refusals: list[str] = []
    widen: list[dict] = []
    warnings: list[str] = []
    tensors: dict[str, dict] = {}
    for shard, header in headers.items():
        pos = 0
        named = {k: v for k, v in header.items() if k != "__metadata__"}
        for name in sorted(named, key=lambda k: named[k]["data_offsets"][0]):
            t = named[name]
            a, b = t["data_offsets"]
            n = 1
            for d in t["shape"]:
                n *= d
            if t["dtype"] not in WIDTHS or a != pos or b - a != n * WIDTHS[t["dtype"]]:
                refusals.append(f"{shard}: {name}: offsets / dtype / shape do not agree")
            pos = b
            tensors[name] = dict(t, shard=shard, nbytes=b - a)
    mapped = index["weight_map"]
    if set(mapped) != set(tensors) or any(mapped[k] != tensors[k]["shard"] for k in tensors):
        missing = sorted(set(mapped) - set(tensors))[:3]
        extra = sorted(set(tensors) - set(mapped))[:3]
        refusals.append(
            f"index and shard headers disagree (first only in the index: {missing}; only in headers: {extra})"
        )
    total = sum(t["nbytes"] for t in tensors.values())
    declared = (index.get("metadata") or {}).get("total_size")
    if declared is not None and declared != total:
        refusals.append(f"index total_size {declared:,} != {total:,} bytes in the shard headers")

    # the declared codebook (the engine never reads it, and its kernels are mul1 only)
    codebook = quant.get("codebook")
    if codebook != "mul1":
        refusals.append(f"codebook is {codebook!r}, not 'mul1' (the engine's EXL3 kernels are mul1 only)")

    groups = {}
    for name, t in tensors.items():
        base, _, part = name.rpartition(".")
        if part == "trellis":
            groups[base] = t
    widths: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for base, t in groups.items():
        shape = t["shape"]
        if t["dtype"] != "I16" or len(shape) != 3 or shape[2] % 16:
            refusals.append(f"{base}.trellis: {t['dtype']} {shape}, expected int16 [in/16, out/16, 16 K]")
            continue
        for part, want in (("suh", shape[0] * 16), ("svh", shape[1] * 16)):
            s = tensors.get(f"{base}.{part}")
            if s is None or s["dtype"] != "F16" or s["shape"] != [want]:
                refusals.append(f"{base}.{part}: expected F16 [{want}]")
        if f"{base}.mcg" in tensors:
            refusals.append(f"{base}: mcg codebook marker (the engine's kernels are mul1 only)")
        elif f"{base}.mul1" not in tensors:
            refusals.append(f"{base}: no mul1 marker (3-instruction codebook; the engine's kernels are mul1 only)")
        bits = shape[2] // 16
        cls, supported = width_class(base)
        widths[cls][bits] += 1
        if bits not in supported:
            warnings.append(f"{base}: {bits}-bit, outside engine 9131e10a's kernels for {cls} (it refuses to boot)")
    for base, t in groups.items():
        if re.search(r"\.ffn\.experts\.\d+\.w1$", base):
            up = groups.get(base[:-2] + "w3")
            if up is None or up["shape"] != t["shape"]:
                refusals.append(
                    f"{base}: gate (w1) width differs from up (w3) {up and up['shape']} (prefill assumes gate = up)"
                )

    by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, t in sorted(tensors.items()):
        base, _, part = name.rpartition(".")
        if part in GROUP_PARTS and base in groups:
            continue
        if name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".ffn.gate.bias_vl"):
            continue
        if name.endswith(".weight") and name[: -len(".weight")] in groups:
            continue
        for cls, pattern, accepted, widenable in CLASSES:
            if pattern.fullmatch(name):
                by_class[cls][t["dtype"]] += 1
                if half_f32_natives and cls in WIDE_CLASSES:
                    accepted = accepted | {"F16", "BF16"}
                if t["dtype"] in accepted:
                    pass
                elif t["dtype"] in widenable:
                    widen.append({"name": name, "shard": t["shard"], "dtype": t["dtype"], "class": cls})
                else:
                    refusals.append(f"{name}: {t['dtype']}, class {cls} takes {'/'.join(sorted(accepted))}")
                break
        else:
            refusals.append(f"{name}: neither read nor ignored by the engine (its loader refuses leftovers)")
    for w in widen:
        refusals.append(f"{w['name']}: {w['dtype']} needs widening to F32 ({w['class']}; scripts/widen_pack.py)")
    return {
        "ok": not refusals,
        "refusals": refusals,
        "needs_widening": widen,
        "warnings": warnings,
        "tensors": len(tensors),
        "shards": len(headers),
        "total_bytes": total,
        "vs_flash17_bytes": total - FLASH17_BYTES,
        "vs_flash17_per_rank_gib": (total - FLASH17_BYTES) / 2 / GIB,
        "codebook": codebook,
        "exl3_widths": {c: dict(sorted(w.items())) for c, w in sorted(widths.items())},
        "native_dtypes": {c: dict(sorted(d.items())) for c, d in sorted(by_class.items())},
    }


def report(r: dict) -> str:
    out = [
        (
            f"{r['shards']} shards, {r['tensors']:,} tensors, {r['total_bytes']:,} tensor bytes "
            f"({r['total_bytes'] / GIB:.2f} GiB); codebook {r['codebook']}"
        ),
        (
            f"vs flash17 ({FLASH17_BYTES:,} B): {r['vs_flash17_bytes']:+,} B, "
            f"{r['vs_flash17_per_rank_gib']:+.2f} GiB a rank at TP=2 (less KV pool room on each rank)"
        ),
        "EXL3 widths (bits a weight: groups):",
    ]
    for cls, w in r["exl3_widths"].items():
        out.append(f"  {cls}: " + ", ".join(f"{b}: {n}" for b, n in w.items()))
    out.append("native tensors (dtype: count):")
    for cls, d in r["native_dtypes"].items():
        out.append(f"  {cls}: " + ", ".join(f"{k}: {n}" for k, n in d.items()))
    for w in r["warnings"]:
        out.append("warning: " + w)
    if r["needs_widening"]:
        shards = sorted({w["shard"] for w in r["needs_widening"]})
        out.append(
            f"needs widening: {len(r['needs_widening'])} tensor(s) in {', '.join(shards)}: run "
            "python3 -I scripts/widen_pack.py <model-dir> <served-dir> and serve <served-dir> "
            "(or an engine with the half-stored F32 natives change: --half-f32-natives)"
        )
    for x in r["refusals"]:
        out.append("REFUSED: " + x)
    out.append("pack check: " + ("ok" if r["ok"] else f"refused ({len(r['refusals'])} finding(s))"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", type=Path, help="the pack's folder (index, config.json, quantization_config.json)")
    p.add_argument("--headers", type=Path, help="read <shard>.header.json files here instead of the shards")
    p.add_argument(
        "--half-f32-natives",
        action="store_true",
        help="the engine accepts F16/BF16 router biases and confidence head (store class f32_wide)",
    )
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    a = p.parse_args(argv)
    try:
        r = check(*load(a.model, a.headers), half_f32_natives=a.half_f32_natives)
    except (OSError, ValueError, KeyError, TypeError, Refused) as e:
        print(f"pack check: cannot read the pack: {e}", file=sys.stderr)
        return 2
    print(json.dumps(r, indent=2) if a.json else report(r))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
