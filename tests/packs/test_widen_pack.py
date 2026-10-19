"""scripts/widen_pack.py: exact F16 / BF16 -> F32 widening of the router bias and confidence head, served copy."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from conftest import ROOT, load_script, write_index, write_shard

w = load_script("widen_pack")
RAW = struct.pack("<6H", 0, 0x8000, 1, 0x7C00, 0x7C01, 0xFFFF)  # zeros, a subnormal, inf, NaNs


def test_every_f16_encoding_widens_exactly():
    raw = struct.pack("<65536H", *range(65536))
    out = w.widen(raw, "F16")
    for u, (bits,) in enumerate(struct.iter_unpack("<I", out)):
        value = struct.unpack("<e", struct.pack("<H", u))[0]
        if math.isnan(value):
            expected = ((u & 0x8000) << 16) | 0x7F800000 | ((u & 1023) << 13)
        else:
            expected = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits == expected, hex(u)
    assert w.narrow(out, "F16") == raw


def test_every_bf16_encoding_widens_exactly():
    raw = struct.pack("<65536H", *range(65536))
    out = w.widen(raw, "BF16")
    assert [x for (x,) in struct.iter_unpack("<I", out)] == [u << 16 for u in range(65536)]
    assert w.narrow(out, "BF16") == raw
    with pytest.raises(ValueError):
        w.widen(b"x", "F16")
    with pytest.raises(ValueError, match="not exactly representable"):
        w.narrow(struct.pack("<f", 0.1), "F16")


@pytest.fixture
def pack(tmp_path):
    src = tmp_path / "download"
    src.mkdir()
    write_shard(
        src / "a.safetensors",
        [
            ("mtp.0.ffn.gate.bias", "F16", [6], RAW),
            ("untouched", "U8", [7], b"\x00\xffhello"),
            ("mtp.0.ffn.gate.bias_vl", "F32", [1], b"\x11\x22\x33\x44"),
        ],
        {"format": "pt", "unicode": "δ"},
    )
    write_shard(
        src / "b.safetensors",
        [
            ("mtp.2.confidence_head.proj.weight", "BF16", [1, 6], RAW),
            ("layers.0.hc_attn_base", "F16", [2], b"\x00\x3c" * 2),  # not a target: stays as stored
            ("scalar", "I32", [], b"\xff" * 4),
        ],
    )
    write_shard(src / "c.safetensors", [("keep", "F16", [6], RAW)])
    (src / "config.json").write_text('{"hello": "world"}')
    (src / ".cache").mkdir()  # a download tool's folder: skipped
    write_index(src)
    return src, tmp_path / "served"


def test_served_copy_rewrites_only_target_shards_and_links_the_rest(pack):
    src, dst = pack
    before = {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()}
    m = w.run(src, dst)
    assert [t["name"] for t in m["targets"]] == ["mtp.0.ffn.gate.bias", "mtp.2.confidence_head.proj.weight"]
    assert sorted(m["files"]) == ["a.safetensors", "b.safetensors", w.INDEX]
    for name in ("c.safetensors", "config.json"):
        assert (dst / name).samefile(src / name)
    assert not (dst / ".cache").exists()
    h, base = w.header(dst / "a.safetensors")
    assert h["__metadata__"] == {"format": "pt", "unicode": "δ"}
    assert h["mtp.0.ffn.gate.bias"] == {"dtype": "F32", "shape": [6], "data_offsets": [0, 24]}
    out = (dst / "a.safetensors").read_bytes()
    assert out[base : base + 24] == w.widen(RAW, "F16")
    assert out[base + 24 : base + 31] == b"\x00\xffhello" and out[base + 31 :] == b"\x11\x22\x33\x44"
    hb, _ = w.header(dst / "b.safetensors")
    assert hb["mtp.2.confidence_head.proj.weight"]["dtype"] == "F32"
    assert hb["layers.0.hc_attn_base"]["dtype"] == "F16"
    idx = json.loads((dst / w.INDEX).read_text())
    assert idx["metadata"]["total_size"] == json.loads((src / w.INDEX).read_text())["metadata"]["total_size"] + 24
    assert before == {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()}


def test_idempotent_and_check_is_read_only(pack):
    src, dst = pack
    m = w.run(src, dst)
    state = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dst.iterdir()}
    assert w.run(src, dst, check=True) == m
    assert w.run(src, dst) == m
    assert state == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dst.iterdir()}


def test_check_without_output_writes_nothing(pack):
    src, dst = pack
    with pytest.raises(ValueError, match="does not exist"):
        w.run(src, dst, check=True)
    assert not dst.exists()


def test_tampered_output_fails_even_with_a_matching_manifest_hash(pack):
    src, dst = pack
    w.run(src, dst)
    p = dst / "a.safetensors"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="hash differs"):
        w.run(src, dst, check=True)
    mp = dst / w.MANIFEST
    m = json.loads(mp.read_text())
    m["files"]["a.safetensors"]["output_sha256"] = w.digest(p)
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="tensor bytes differ"):
        w.run(src, dst, check=True)


def test_widened_value_must_round_trip(pack):
    src, dst = pack
    w.run(src, dst)
    p = dst / "a.safetensors"
    _, base = w.header(p)
    raw = bytearray(p.read_bytes())
    raw[base + 4 : base + 8] = struct.pack("<f", 0.1)  # not an F16 value
    p.write_bytes(raw)
    m = json.loads((dst / w.MANIFEST).read_text())
    m["files"]["a.safetensors"]["output_sha256"] = w.digest(p)
    (dst / w.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="not exactly representable"):
        w.run(src, dst, check=True)


def test_changed_source_and_broken_link_are_refused(pack):
    src, dst = pack
    w.run(src, dst)
    (dst / "config.json").unlink()
    (dst / "config.json").write_text('{"hello": "world"}')
    with pytest.raises(ValueError, match="not a hard link"):
        w.run(src, dst, check=True)


def test_incomplete_output_is_refused(pack):
    src, dst = pack
    dst.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        w.run(src, dst)


def test_failure_removes_the_partial_output(pack):
    src, dst = pack
    link_fails = mock.patch.object(w.os, "link", side_effect=OSError(18, "Invalid cross-device link"))
    with link_fails, pytest.raises(ValueError, match="filesystem"):
        w.run(src, dst)
    assert not dst.exists()


def test_separation_and_index_safety(pack):
    src, dst = pack
    for bad in (src, src / "inner", src.parent):
        with pytest.raises(ValueError, match="separate"):
            w.run(src, bad)
    idx = json.loads((src / w.INDEX).read_text())
    for name in ("../x.safetensors", "sub/x.safetensors"):
        broken = dict(idx, weight_map=dict(idx["weight_map"], keep=name))
        (src / w.INDEX).write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="unsafe filename"):
            w.run(src, dst)
    broken = dict(idx, weight_map=dict(idx["weight_map"], invented="a.safetensors"))
    (src / w.INDEX).write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="index and header differ"):
        w.run(src, dst)
    broken = dict(idx, metadata={"total_size": idx["metadata"]["total_size"] + 1})
    (src / w.INDEX).write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="total_size"):
        w.run(src, dst)
    assert not dst.exists()


def test_corrupt_headers(pack):
    src, _ = pack
    path = src / "a.safetensors"
    original = path.read_bytes()
    for raw in (b"", original[:7], original[:30], original[:-1], original + b"x", struct.pack("<Q", 2**40)):
        path.write_bytes(raw)
        with pytest.raises((ValueError, UnicodeError)):
            w.header(path)
    with pytest.raises(ValueError, match="duplicate"):
        w.load_json('{"a": 1, "a": 2}')


def test_chunk_boundaries_and_index_without_total_size(pack):
    src, dst = pack
    idx = json.loads((src / w.INDEX).read_text())
    idx["metadata"] = {}
    (src / w.INDEX).write_text(json.dumps(idx))
    with mock.patch.object(w, "CHUNK", 4):
        w.run(src, dst)
        w.run(src, dst, check=True)
    assert json.loads((dst / w.INDEX).read_text())["metadata"] == {}


def test_already_f32_pack_is_all_links(tmp_path):
    src = tmp_path / "download"
    src.mkdir()
    write_shard(src / "a.safetensors", [("mtp.0.ffn.gate.bias", "F32", [6], b"\0" * 24)])
    write_index(src)
    m = w.run(src, tmp_path / "served")
    assert m["targets"] == [] and sorted(m["files"]) == [w.INDEX]
    assert (tmp_path / "served" / "a.safetensors").samefile(src / "a.safetensors")


def test_cli(pack):
    src, dst = pack
    tool = [sys.executable, "-I", "-B", str(ROOT / "scripts" / "widen_pack.py"), str(src), str(dst)]
    for flags in ([], ["--check"]):
        proc = subprocess.run([*tool, *flags], check=True, capture_output=True, text=True)
        assert json.loads(proc.stdout)["verified"]
    raw = bytearray((dst / "b.safetensors").read_bytes())
    raw[-1] ^= 1
    (dst / "b.safetensors").write_bytes(raw)
    proc = subprocess.run([*tool, "--check"], check=False, capture_output=True, text=True)
    assert proc.returncode == 1 and "refused:" in proc.stderr


def test_real_shapes_round_trip_through_check_pack(pack):
    src, dst = pack
    w.run(src, dst)
    cp = load_script("check_pack")
    for d, n in ((src, 2), (dst, 0)):
        _, headers, _ = cp.load(Path(d), None)
        widen_rows = [k for h in headers.values() for k, t in h.items() if k != "__metadata__" and w.needs(k, t)]
        assert len(widen_rows) == n
