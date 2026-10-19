#!/usr/bin/env python3
"""Widen a pack's half-stored F32 natives to F32, exactly, into a served copy of the pack folder.

Engine snapshot 9131e10a (this recipe's ``engine/zig``) accepts the router bias ``*.ffn.gate.bias`` only as F32 and
the DSpark confidence head ``*.confidence_head.proj.weight`` only as F32 or BF16. Some converters store them F16
(the DeepSeek-V4.1-Flash-1017 EXL3 EAQ pack: ``mtp.{0,1,2}.ffn.gate.bias`` and ``mtp.2.confidence_head.proj.weight``,
in its last two shards). Every F16 and BF16 value is exactly representable in F32, so widening changes no value:

    python3 -I scripts/widen_pack.py <downloaded-pack> <served-pack>          # once, on each node
    python3 -I scripts/widen_pack.py <downloaded-pack> <served-pack> --check  # read-only re-verification

``<served-pack>`` is a new folder on the same filesystem as the download: the shards holding such a tensor are
rewritten there (header and data; every other tensor's bytes copied unchanged) and every other file is a hard
link to the download's (no extra space; hard links, unlike symlinks, still resolve inside the serving container).
The index's ``total_size`` is updated. ``widen-manifest.json`` (written last) records the source and the sha256
of each rewritten file. Running it again on a complete output only verifies it; ``--check`` never writes. The
verification reads back every widened tensor, narrows it to its source dtype and requires the source bytes, and
requires every other tensor's bytes unchanged. Only the router bias and the confidence head are widened: the
mixers, sinks and norms stay as stored (check_pack.py refuses them in half precision).

Point ``MODEL_DIR`` at ``<served-pack>``. Keep the download unchanged afterwards: the links share its files.

An engine with the half-stored F32 natives loader change (TensorFold ``perf-1100/eaq-1``: store class
``f32_wide``) accepts these tensors as stored and widens them at load, so this step is only needed for the current
engine snapshot; it stays harmless afterwards (the values loaded are identical).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

INDEX = "model.safetensors.index.json"
MANIFEST = "widen-manifest.json"
SUFFIXES = (".ffn.gate.bias", ".confidence_head.proj.weight")
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
CHUNK = 1 << 20


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_json(raw):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, f"duplicate JSON key: {k}")
            out[k] = v
        return out

    return json.loads(raw, object_pairs_hook=unique)


def safe_name(name) -> str:
    require(
        isinstance(name, str)
        and name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name,
        f"unsafe filename: {name!r}",
    )
    return name


def header(path: Path) -> tuple[dict, int]:
    """A shard's header (validated: shapes, dtypes, contiguous offsets, file size) and its data offset."""
    with path.open("rb") as f:
        prefix = f.read(8)
        require(len(prefix) == 8, f"short prefix: {path}")
        n = struct.unpack("<Q", prefix)[0]
        require(2 <= n <= 256 << 20, f"invalid header size: {path}")
        raw = f.read(n)
        require(len(raw) == n and raw.startswith(b"{"), f"short or invalid header: {path}")
    h = load_json(raw)
    require(isinstance(h, dict), "header must be an object")
    meta = h.get("__metadata__", {})
    require(
        isinstance(meta, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()),
        f"invalid metadata: {path}",
    )
    pos = 0
    for name in tensor_order(h):
        t = h[name]
        require(isinstance(t, dict) and set(t) == {"dtype", "shape", "data_offsets"}, f"invalid descriptor: {name}")
        shape, offsets, dtype = t["shape"], t["data_offsets"], t["dtype"]
        require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), f"invalid shape: {name}")
        require(dtype in WIDTHS, f"unknown dtype: {name}")
        require(
            isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets),
            f"invalid offsets: {name}",
        )
        a, b = offsets
        require(a == pos and b - a == math.prod(shape) * WIDTHS[dtype], f"offsets or size do not match: {name}")
        pos = b
    require(path.stat().st_size == 8 + n + pos, f"file size does not match the header: {path}")
    return h, 8 + n


def tensor_order(h: dict) -> list[str]:
    names = [k for k in h if k != "__metadata__"]
    for k in names:
        require(
            isinstance(h[k], dict) and isinstance(h[k].get("data_offsets"), list) and len(h[k]["data_offsets"]) == 2,
            f"invalid descriptor: {k}",
        )
    return sorted(names, key=lambda k: tuple(h[k]["data_offsets"]))


def needs(name: str, t: dict) -> bool:
    return name.endswith(SUFFIXES) and t["dtype"] in ("F16", "BF16")


def f16_bits(u: int) -> int:
    """F32 bits of F16 bits u: exact, NaN sign and payload kept."""
    sign, exp, frac = (u & 0x8000) << 16, (u >> 10) & 31, u & 1023
    if exp == 31:
        return sign | 0x7F800000 | (frac << 13)
    if exp:
        return sign | ((exp + 112) << 23) | (frac << 13)
    if not frac:
        return sign
    shift = frac.bit_length() - 1
    return sign | ((shift + 103) << 23) | ((frac - (1 << shift)) << (23 - shift))


_WIDE: dict[str, tuple[bytes, ...]] = {}
_NARROW: dict[str, dict[bytes, bytes]] = {}


def widen(raw: bytes, dtype: str) -> bytes:
    require(dtype in ("F16", "BF16") and len(raw) % 2 == 0, "invalid widening input")
    if dtype not in _WIDE:
        _WIDE[dtype] = tuple(struct.pack("<I", f16_bits(u) if dtype == "F16" else u << 16) for u in range(65536))
    table = _WIDE[dtype]
    return b"".join(table[u] for (u,) in struct.iter_unpack("<H", raw))


def narrow(raw: bytes, dtype: str) -> bytes:
    """The inverse of widen: refuses any F32 value that is not exactly a widened `dtype` value."""
    require(len(raw) % 4 == 0, "invalid narrowing input")
    if dtype not in _NARROW:
        widen(b"\0\0", dtype)
        _NARROW[dtype] = {w: struct.pack("<H", u) for u, w in enumerate(_WIDE[dtype])}
    table = _NARROW[dtype]
    out = []
    for i in range(0, len(raw), 4):
        half = table.get(raw[i : i + 4])
        require(half is not None, f"value not exactly representable in {dtype}")
        out.append(half)
    return b"".join(out)


def chunks(f, count: int):
    while count:
        raw = f.read(min(CHUNK, count))
        require(bool(raw), "short tensor read")
        count -= len(raw)
        yield raw


def widened(h: dict) -> tuple[dict, int]:
    """The output header and its data size."""
    out, pos = copy.deepcopy(h), 0
    for name in tensor_order(h):
        t = h[name]
        size = t["data_offsets"][1] - t["data_offsets"][0]
        if needs(name, t):
            out[name]["dtype"] = "F32"
            size *= 2
        out[name]["data_offsets"] = [pos, pos + size]
        pos += size
    return out, pos


def rewrite(src: Path, dst: Path, h: dict, base: int) -> None:
    out, _ = widened(h)
    blob = json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode()
    blob += b" " * (-len(blob) % 8)
    with src.open("rb") as fi, dst.open("xb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for name in tensor_order(h):
            t = h[name]
            a, b = t["data_offsets"]
            fi.seek(base + a)
            for raw in chunks(fi, b - a):
                fo.write(widen(raw, t["dtype"]) if needs(name, t) else raw)
        fo.flush()
        os.fsync(fo.fileno())


def equivalent(src: Path, dst: Path, h: dict, base: int) -> None:
    """dst holds src's tensors with the targets widened: each one narrows back to the source bytes exactly."""
    dh, dbase = header(dst)
    require(dh == widened(h)[0], f"output header differs: {dst.name}")
    with src.open("rb") as fi, dst.open("rb") as fo:
        for name in tensor_order(h):
            t = h[name]
            a, b = t["data_offsets"]
            fi.seek(base + a)
            fo.seek(dbase + dh[name]["data_offsets"][0])
            for raw in chunks(fi, b - a):
                if needs(name, t):
                    out = fo.read(2 * len(raw))
                    require(narrow(out, t["dtype"]) == raw, f"widened tensor does not round-trip: {name}")
                else:
                    require(fo.read(len(raw)) == raw, f"tensor bytes differ: {name}")


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inspect(src: Path) -> tuple[dict, dict, list[dict], list[str]]:
    """(index, header of each shard, targets, the folder's files to serve)."""
    idx = load_json((src / INDEX).read_bytes())
    require(isinstance(idx, dict) and isinstance(idx.get("weight_map"), dict) and idx["weight_map"], "invalid index")
    wm = idx["weight_map"]
    shards = sorted({safe_name(v) for v in wm.values()})
    headers, targets, total = {}, [], 0
    for name in shards:
        h, base = header(src / name)
        require(
            set(h) - {"__metadata__"} == {k for k, v in wm.items() if v == name}, f"index and header differ: {name}"
        )
        headers[name] = (h, base)
        total += (src / name).stat().st_size - base
        targets += [
            {"name": k, "shard": name, "dtype": h[k]["dtype"], "shape": h[k]["shape"]}
            for k in tensor_order(h)
            if needs(k, h[k])
        ]
    meta = idx.get("metadata")
    require(isinstance(meta, dict), "invalid index metadata")
    require("total_size" not in meta or meta["total_size"] == total, "index total_size does not match the shards")
    files = []
    for p in sorted(src.iterdir()):
        name = safe_name(p.name)
        if name.startswith(".") and p.is_dir():
            continue  # e.g. a download tool's .cache/: not part of the pack
        require(name != MANIFEST, f"the source already holds {MANIFEST}: point at the downloaded pack")
        require(p.is_file(), f"source entry is not a file: {name}")
        require(not name.endswith(".safetensors") or name in headers, f"shard not in the index: {name}")
        files.append(name)
    return idx, headers, targets, files


def rewritten(headers: dict, name: str) -> bool:
    return name in headers and any(needs(k, t) for k, t in headers[name][0].items() if k != "__metadata__")


def out_index(idx: dict, headers: dict) -> dict:
    out = copy.deepcopy(idx)
    if "total_size" in out["metadata"]:
        out["metadata"]["total_size"] = sum(widened(h)[1] for h, _ in headers.values())
    return out


def verify(src: Path, dst: Path) -> dict:
    idx, headers, targets, files = inspect(src)
    require((dst / MANIFEST).is_file(), f"{dst} is incomplete (no {MANIFEST}): remove it and run again")
    m = load_json((dst / MANIFEST).read_bytes())
    require(
        m.get("version") == 2 and m.get("source") == str(src) and m.get("targets") == targets,
        "manifest does not describe this source",
    )
    require({p.name for p in dst.iterdir()} == set(files) | {MANIFEST}, "output file set differs from the source")
    require(load_json((dst / INDEX).read_bytes()) == out_index(idx, headers), "output index differs")
    for name in files:
        a, b = src / name, dst / name
        if name == INDEX or rewritten(headers, name):
            record = m["files"].get(name, {})
            require(not os.path.samefile(a, b), f"rewritten file must be its own copy: {name}")
            require(record == {"source_sha256": digest(a), "output_sha256": digest(b)}, f"hash differs: {name}")
            if name != INDEX:
                equivalent(a, b, *headers[name])
        else:
            require(os.path.samefile(a, b), f"not a hard link to the source file: {name}")
    return m


def run(source, destination, check: bool = False) -> dict:
    src, dst = Path(source).resolve(strict=True), Path(destination).absolute()
    resolved = dst.resolve()
    require(
        src != resolved and src not in resolved.parents and resolved not in src.parents,
        "source and destination must be separate folders",
    )
    if os.path.lexists(dst):
        return verify(src, dst)
    require(not check, f"{dst} does not exist (--check does not create it)")
    idx, headers, targets, files = inspect(src)
    os.mkdir(dst)  # exclusive: a concurrent run or an existing folder fails here
    try:
        manifest = {"version": 2, "source": str(src), "targets": targets, "files": {}}
        for name in files:
            if name == INDEX:
                with (dst / name).open("x") as f:
                    json.dump(out_index(idx, headers), f, indent=2)
            elif rewritten(headers, name):
                rewrite(src / name, dst / name, *headers[name])
            else:
                try:
                    os.link(src / name, dst / name, follow_symlinks=True)
                except OSError as e:
                    raise ValueError(f"cannot hard-link {name} ({e.strerror}): put {dst} on the source's filesystem")
                continue
            manifest["files"][name] = {"source_sha256": digest(src / name), "output_sha256": digest(dst / name)}
        tmp = dst / f".{MANIFEST}.tmp"
        with tmp.open("x") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, dst / MANIFEST)
        fd = os.open(dst, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return verify(src, dst)
    except BaseException:
        shutil.rmtree(dst)
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("source", type=Path, help="the downloaded pack folder (left unchanged)")
    p.add_argument("destination", type=Path, help="the served folder to create (same filesystem)")
    p.add_argument("--check", action="store_true", help="verify an existing output, read-only")
    a = p.parse_args(argv)
    try:
        m = run(a.source, a.destination, a.check)
    except (OSError, ValueError, TypeError, KeyError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"verified": True, "widened": m["targets"], "rewritten_files": sorted(m["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
