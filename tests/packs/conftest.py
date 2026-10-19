"""Offline tests of the pack tools (scripts/check_pack.py, scripts/widen_pack.py) on tiny synthetic packs.

    python -m pytest -q tests/packs

Standard library and pytest only: no network, no GPU, no real weights.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_shard(path: Path, specs, metadata=None) -> None:
    """specs: (name, dtype, shape, raw bytes) in file order."""
    h, data = {}, b""
    if metadata is not None:
        h["__metadata__"] = metadata
    for name, dtype, shape, raw in specs:
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    blob = json.dumps(h, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def group(prefix: str, k_in: int, n_out: int, bits: int, marker: str = "mul1"):
    """An EXL3 group's tensors: int16 trellis [in/16, out/16, 16 bits], fp16 suh / svh, a codebook marker."""
    words = 16 * bits
    out = [
        (f"{prefix}.trellis", "I16", [k_in // 16, n_out // 16, words], b"\0" * (k_in // 16 * n_out // 16 * words * 2)),
        (f"{prefix}.suh", "F16", [k_in], b"\0" * 2 * k_in),
        (f"{prefix}.svh", "F16", [n_out], b"\0" * 2 * n_out),
    ]
    if marker:
        out.append((f"{prefix}.{marker}", "I32", [], b"\0" * 4))
    return out


def native(name: str, dtype: str, n: int):
    return (name, dtype, [n], b"\x00\x3c" * (n * {"F32": 4, "F16": 2, "BF16": 2}[dtype] // 2))


def write_index(model: Path, total: bool = True) -> dict:
    wm, size = {}, 0
    for p in sorted(model.glob("*.safetensors")):
        raw = p.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        h = json.loads(raw[8 : 8 + n])
        wm.update({k: p.name for k in h if k != "__metadata__"})
        size += len(raw) - 8 - n
    index = {"metadata": {"total_size": size} if total else {}, "weight_map": wm}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    return index


@pytest.fixture
def tiny_pack(tmp_path: Path):
    """A two-shard pack in the 1017 layout's naming, stored like the EAQ pack: an F16 DSpark router bias and
    confidence head (widening needed), F16 router gates (accepted as half), everything else as the engine wants."""

    def make(bias_dtype="F16", conf_dtype="F16", up_bits=3, marker="mul1", codebook="mul1", extra=()):
        model = tmp_path / "pack"
        model.mkdir()
        write_shard(
            model / "model-00001-of-00002.safetensors",
            [
                native("embed.weight", "BF16", 32),
                native("layers.0.attn_norm.weight", "BF16", 16),
                native("layers.0.hc_attn_base", "F32", 8),
                native("layers.0.attn.attn_sink", "F32", 4),
                native("layers.0.ffn.gate.weight", "F16", 16),
                native("layers.0.ffn.gate.bias", "F32", 4),
                native("layers.0.ffn.gate.bias_vl", "F32", 4),
                *group("layers.0.attn.wq_a", 32, 32, 5),
                *group("layers.0.ffn.experts.0.w1", 32, 16, 3, marker),
                *group("layers.0.ffn.experts.0.w3", 32, 16, up_bits),
                *group("layers.0.ffn.experts.0.w2", 16, 32, 3),
                ("vision.patch_embed.weight", "BF16", [4], b"\0" * 8),
                *extra,
            ],
            {"format": "pt"},
        )
        write_shard(
            model / "model-00002-of-00002.safetensors",
            [
                native("mtp.0.ffn.gate.bias", bias_dtype, 4),
                native("mtp.0.main_norm.weight", "BF16", 16),
                native("mtp.0.confidence_head.proj.weight", conf_dtype, 8),
                native("mtp.0.markov_head.head.weight", "F16", 8),
                native("norm.weight", "BF16", 16),
                *group("head", 32, 64, 6),
            ],
        )
        write_index(model)
        (model / "quantization_config.json").write_text(json.dumps({"quant_method": "exl3", "codebook": codebook}))
        (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
        return model

    return make
