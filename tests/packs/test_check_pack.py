"""scripts/check_pack.py: what engine 9131e10a would refuse or load wrong, from the index and headers alone."""

from __future__ import annotations

import json
import struct
import subprocess
import sys

from conftest import ROOT, group, load_script, native, write_index, write_shard

cp = load_script("check_pack")


def run(model, *args):
    return cp.check(*cp.load(model, None), *args)


def test_eaq_like_pack_needs_exactly_the_half_stored_f32_natives(tiny_pack):
    r = run(tiny_pack())
    assert not r["ok"]
    assert sorted(w["name"] for w in r["needs_widening"]) == [
        "mtp.0.confidence_head.proj.weight",
        "mtp.0.ffn.gate.bias",
    ]
    assert {w["shard"] for w in r["needs_widening"]} == {"model-00002-of-00002.safetensors"}
    assert all("needs widening" in x for x in r["refusals"])
    # F16 router gates and Markov head are .half: accepted
    assert r["native_dtypes"]["half"] == {"BF16": 1, "F16": 2}
    assert r["exl3_widths"]["routed experts"] == {3: 3}
    assert r["exl3_widths"]["lm head"] == {6: 1}


def test_half_f32_natives_engine_accepts_and_f32_pack_passes(tmp_path, tiny_pack):
    model = tiny_pack()
    assert run(model, True)["ok"]
    (model / "model-00002-of-00002.safetensors").unlink()
    write_shard(
        model / "model-00002-of-00002.safetensors",
        [
            native("mtp.0.ffn.gate.bias", "F32", 4),
            native("mtp.0.main_norm.weight", "BF16", 16),
            native("mtp.0.confidence_head.proj.weight", "BF16", 8),
            native("mtp.0.markov_head.head.weight", "F16", 8),
            native("norm.weight", "BF16", 16),
            *group("head", 32, 64, 6),
        ],
    )
    write_index(model)
    r = run(model)
    assert r["ok"], r["refusals"]
    assert r["total_bytes"] == json.loads((model / cp.INDEX).read_text())["metadata"]["total_size"]


def test_strict_f32_and_norm_classes_are_refused_not_widened(tiny_pack):
    model = tiny_pack(extra=[native("layers.0.hc_ffn_scale", "F16", 3), native("layers.0.ffn_norm.weight", "F16", 16)])
    r = run(model, True)
    assert any("layers.0.hc_ffn_scale: F16, class f32 mixer / sink takes F32" in x for x in r["refusals"])
    assert any("layers.0.ffn_norm.weight: F16, class norm" in x for x in r["refusals"])
    assert r["needs_widening"] == []


def test_declared_codebook_must_be_mul1(tiny_pack):
    assert any("codebook is 'mcg'" in x for x in run(tiny_pack(codebook="mcg"), True)["refusals"])


def test_group_without_mul1_marker(tiny_pack):
    r = run(tiny_pack(marker=""), True)
    assert any("layers.0.ffn.experts.0.w1: no mul1 marker" in x for x in r["refusals"])


def test_mcg_marker(tiny_pack):
    r = run(tiny_pack(marker="mcg"), True)
    assert any("layers.0.ffn.experts.0.w1: mcg codebook marker" in x for x in r["refusals"])


def test_gate_up_width_mismatch(tiny_pack):
    r = run(tiny_pack(up_bits=4), True)
    assert any("gate (w1) width differs from up (w3)" in x for x in r["refusals"])


def test_leftover_tensor_and_unsupported_width(tiny_pack):
    model = tiny_pack(
        extra=[
            ("layers.0.attn.mystery", "F32", [1], b"\0" * 4),
            *group("layers.0.ffn.experts.1.w1", 32, 16, 7),
            *group("layers.0.ffn.experts.1.w3", 32, 16, 7),
        ]
    )
    r = run(model, True)
    assert any("layers.0.attn.mystery: neither read nor ignored" in x for x in r["refusals"])
    assert any("layers.0.ffn.experts.1.w1: 7-bit" in w for w in r["warnings"])


def test_index_disagreements(tiny_pack):
    model = tiny_pack()
    index = json.loads((model / cp.INDEX).read_text())
    index["metadata"]["total_size"] += 2
    index["weight_map"]["invented"] = "model-00001-of-00002.safetensors"
    (model / cp.INDEX).write_text(json.dumps(index))
    r = run(model, True)
    assert any("total_size" in x for x in r["refusals"])
    assert any("index and shard headers disagree" in x for x in r["refusals"])


def test_truncated_shard_cannot_be_read(tiny_pack):
    model = tiny_pack()
    p = model / "model-00001-of-00002.safetensors"
    p.write_bytes(p.read_bytes()[:-1])
    try:
        cp.load(model, None)
    except cp.Refused as e:
        assert "file size" in str(e)
    else:
        raise AssertionError("a truncated shard was read")


def test_headers_dir_mode_and_codebook_from_config(tmp_path, tiny_pack):
    model = tiny_pack()
    hdir = tmp_path / "headers"
    hdir.mkdir()
    for p in model.glob("*.safetensors"):
        raw = p.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        (hdir / f"{p.name}.header.json").write_bytes(raw[8 : 8 + n])
        p.unlink()  # header-only: the shards are not needed
    (model / "quantization_config.json").unlink()
    (model / "config.json").write_text(json.dumps({"quantization_config": {"codebook": "mul1"}}))
    r = cp.check(*cp.load(model, hdir), half_f32_natives=True)
    assert r["ok"], r["refusals"]
    assert r["vs_flash17_bytes"] == r["total_bytes"] - cp.FLASH17_BYTES


def test_cli_exit_codes(tiny_pack, tmp_path):
    model = tiny_pack()
    tool = [sys.executable, "-I", "-B", str(ROOT / "scripts" / "check_pack.py"), str(model)]
    refused = subprocess.run(tool, check=False, capture_output=True, text=True)
    assert refused.returncode == 1
    assert "scripts/widen_pack.py" in refused.stdout and "pack check: refused (2 finding(s))" in refused.stdout
    ok = subprocess.run([*tool, "--half-f32-natives", "--json"], check=False, capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)["ok"]
    missing = subprocess.run([*tool[:-1], str(tmp_path / "nothing")], check=False, capture_output=True, text=True)
    assert missing.returncode == 2
