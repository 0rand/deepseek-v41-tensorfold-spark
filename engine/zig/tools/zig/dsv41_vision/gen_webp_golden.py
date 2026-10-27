# SPDX-License-Identifier: MIT
"""WebP goldens for the Zig image front end: Pillow 12.3's RGB of each file's first frame (``--out DIR``)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path


def chunk(tag: bytes, body: bytes) -> bytes:
    """One RIFF chunk with its pad byte."""
    return tag + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")


def riff(chunks: bytes) -> bytes:
    """A RIFF WEBP file around ``chunks``."""
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WEBP" + chunks


def le24(v: int) -> bytes:
    """Three little-endian bytes."""
    return struct.pack("<I", v)[:3]


def chunks_of(data: bytes) -> dict[bytes, bytes]:
    """The top-level chunks of a RIFF WEBP file by tag (the first of each)."""
    out, at = {}, 12
    while at + 8 <= len(data):
        tag, size = data[at : at + 4], struct.unpack("<I", data[at + 4 : at + 8])[0]
        out.setdefault(tag, data[at + 8 : at + 8 + size])
        at += 8 + size + (size & 1)
    return out


def vp8x(w: int, h: int, flags: int) -> bytes:
    """A VP8X chunk for a w x h canvas."""
    return chunk(b"VP8X", struct.pack("<I", flags) + le24(w - 1) + le24(h - 1))


def filtered(alpha, f: int):
    """The ALPH deltas of a plane under filter ``f`` (libwebp filters.c: none, horizontal, vertical, gradient)."""
    import numpy as np

    a = alpha.astype(np.int32)
    h, w = a.shape
    d = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            if f == 0:
                p = 0
            elif y == 0:
                p = a[y, x - 1] if x else 0
            elif f == 1:
                p = a[y, x - 1] if x else a[y - 1, 0]
            elif f == 2:
                p = a[y - 1, x]
            else:
                p = (
                    a[y - 1, x]
                    if x == 0
                    else min(255, max(0, a[y, x - 1] + a[y - 1, x] - a[y - 1, x - 1]))
                )
            d[y, x] = (a[y, x] - p) & 255
    return d.astype(np.uint8)


def images(full: bool):
    """(name, file bytes) for every golden image; ``full`` adds large ones (the TF_DSV41_WEBP_GOLDEN set)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(4102)

    def field(h, w, c, hi=256):
        y, x = np.mgrid[0:h, 0:w]
        base = (x * 7 + y * 3)[..., None] + np.arange(c)[None, None, :] * 50
        return ((base + rng.integers(0, 40, (h, w, c))) % hi).astype(np.uint8)

    def pil(img, **kw):
        b = io.BytesIO()
        img.save(b, "WEBP", **kw)
        return b.getvalue()

    out = []
    h, w = 23, 37
    rgb = Image.fromarray(field(h, w, 3), "RGB")
    rgba = Image.fromarray(field(h, w, 4), "RGBA")
    out.append(("lossy.webp", pil(rgb, quality=80)))
    for q in (1, 50, 100):
        out.append((f"lossy-q{q}.webp", pil(rgb, quality=q)))
    for m in (0, 6):
        out.append((f"lossy-m{m}.webp", pil(rgb, quality=75, method=m)))
    out.append(
        ("lossy-flat.webp", pil(Image.new("RGB", (40, 24), (200, 30, 90)), quality=90))
    )
    for sw, sh in ((1, 1), (7, 1), (1, 7), (83, 61), (130, 70)):
        out.append(
            (
                f"lossy-{sw}x{sh}.webp",
                pil(Image.fromarray(field(sh, sw, 3), "RGB"), quality=85),
            )
        )
    out.append(("lossy-alpha.webp", pil(rgba, quality=70)))
    out.append(("lossy-alpha-q50.webp", pil(rgba, quality=70, alpha_quality=50)))
    out.append(("lossy-icc.webp", pil(rgb, quality=80, icc_profile=b"\0" * 7)))
    # an ALPH chunk built here for each filter, raw (method 0) and as a VP8L stream's green channel (method 1)
    vp8 = chunks_of(out[0][1])[b"VP8 "]
    plane = field(h, w, 1)[..., 0]
    for f, fname in enumerate(("none", "h", "v", "g")):
        d = filtered(plane, f)
        raw_alph = bytes([f << 2]) + d.tobytes()
        out.append(
            (
                f"alph-raw-{fname}.webp",
                riff(vp8x(w, h, 0x10) + chunk(b"ALPH", raw_alph) + chunk(b"VP8 ", vp8)),
            )
        )
        grey = Image.fromarray(np.repeat(d[..., None], 3, axis=2), "RGB")
        ll = chunks_of(pil(grey, lossless=True))[b"VP8L"]
        out.append(
            (
                f"alph-ll-{fname}.webp",
                riff(
                    vp8x(w, h, 0x10)
                    + chunk(b"ALPH", bytes([1 | f << 2]) + ll[5:])
                    + chunk(b"VP8 ", vp8)
                ),
            )
        )
    out.append(("lossless.webp", pil(rgb, lossless=True)))
    out.append(("lossless-alpha.webp", pil(rgba, lossless=True)))
    clear = field(h, w, 4)
    clear[::3, :, 3] = 0
    out.append(
        (
            "lossless-exact.webp",
            pil(Image.fromarray(clear, "RGBA"), lossless=True, exact=True),
        )
    )
    for q, m in ((0, 0), (100, 6)):
        out.append(
            (f"lossless-q{q}-m{m}.webp", pil(rgba, lossless=True, quality=q, method=m))
        )
    for n in (2, 4, 16, 256):
        idx = field(h, w, 1)[..., 0].astype(np.int64) % n
        pal = rng.integers(0, 256, (n, 3)).astype(np.uint8)
        out.append(
            (
                f"lossless-pal{n}.webp",
                pil(Image.fromarray(pal[idx], "RGB"), lossless=True),
            )
        )
    out.append(
        (
            "lossless-1x1.webp",
            pil(Image.fromarray(field(1, 1, 4), "RGBA"), lossless=True),
        )
    )
    frames = [Image.fromarray(field(h, w, 4), "RGBA") for _ in range(3)]
    out.append(
        (
            "anim-lossy.webp",
            pil(frames[0], save_all=True, append_images=frames[1:], duration=50),
        )
    )
    out.append(
        (
            "anim-lossless.webp",
            pil(frames[0], save_all=True, append_images=frames[1:], lossless=True),
        )
    )
    # an animation whose first frame covers part of the canvas, at an offset: the rest stays transparent black
    small = chunks_of(pil(Image.fromarray(field(9, 11, 4), "RGBA"), lossless=True))[
        b"VP8L"
    ]
    anmf = (
        le24(2)
        + le24(1)
        + le24(10)
        + le24(8)
        + le24(100)
        + b"\0"
        + chunk(b"VP8L", small)
    )
    out.append(
        (
            "anim-offset.webp",
            riff(vp8x(w, h, 0x12) + chunk(b"ANIM", b"\0" * 6) + chunk(b"ANMF", anmf)),
        )
    )
    out += variants(dict(out))
    if not full:
        return out
    big = Image.fromarray(field(1500, 2000, 3), "RGB")
    out.append(("big-lossy.webp", pil(big, quality=85)))
    out.append(("big-lossy-m0.webp", pil(big, quality=30, method=0)))
    out.append(
        (
            "big-lossless.webp",
            pil(Image.fromarray(field(700, 1300, 4), "RGBA"), lossless=True),
        )
    )
    out.append(
        (
            "big-lossy-alpha.webp",
            pil(Image.fromarray(field(481, 641, 4), "RGBA"), quality=60),
        )
    )
    out.append(
        (
            "tall-lossy.webp",
            pil(Image.fromarray(field(3000, 199, 3), "RGB"), quality=75),
        )
    )
    wide = [Image.fromarray(field(120, 2601, 4), "RGBA") for _ in range(2)]
    out.append(
        (
            "wide-anim.webp",
            pil(wide[0], save_all=True, append_images=wide[1:], lossless=True),
        )
    )
    return out


# vision_prep.to_rgb's rule; golden.jsonl: {generator}, then {name, size, rgb_sha256} or {name, refused} per file
def variants(files: dict[str, bytes]) -> list[tuple[str, bytes]]:
    """Container variants of committed files: the demuxer's refusals, and a still image whose ALPH the demuxer drops."""
    out = []
    lossless, alpha, anim = (
        files["lossless.webp"],
        files["lossy-alpha.webp"],
        files["anim-lossless.webp"],
    )
    out.append(("bad-tail-short.webp", riff(lossless[12:] + b"JUNK")))
    out.append(
        (
            "bad-tail-overflow.webp",
            riff(lossless[12:] + b"JUNK" + struct.pack("<I", 100) + b"xx"),
        )
    )
    out.append(("ok-tail-chunk.webp", riff(lossless[12:] + chunk(b"JUNK", b"xyz"))))
    out.append(
        (
            "bad-two-images.webp",
            riff(alpha[12:] + chunk(b"VP8 ", chunks_of(alpha)[b"VP8 "])),
        )
    )
    out.append(("bad-two-vp8x.webp", riff(alpha[12:30] + alpha[12:])))
    for bit in (0x01, 0x40, 0x80):
        flipped = bytearray(alpha)
        flipped[20] ^= bit
        out.append((f"bad-vp8x-flag-{bit:02x}.webp", bytes(flipped)))
    cleared = bytearray(alpha)
    cleared[20] &= ~0x10
    out.append(("ok-alpha-flag-clear.webp", bytes(cleared)))
    c = chunks_of(alpha)
    out.append(
        (
            "bad-alph-after-vp8.webp",
            riff(
                alpha[12:30] + chunk(b"VP8 ", c[b"VP8 "]) + chunk(b"ALPH", c[b"ALPH"])
            ),
        )
    )
    head = anim[12:30]
    out.append(
        (
            "bad-empty-anim.webp",
            riff(head + chunk(b"ANIM", b"") + chunk(b"ANMF", chunks_of(anim)[b"ANMF"])),
        )
    )
    out.append(
        (
            "bad-anmf-tail.webp",
            riff(
                head
                + chunk(b"ANIM", b"\0" * 6)
                + chunk(b"ANMF", chunks_of(anim)[b"ANMF"] + b"junk")
            ),
        )
    )
    out.append(
        ("bad-anmf-empty-last.webp", riff(anim[12:] + chunk(b"ANMF", b"\0" * 16)[:24]))
    )
    return out


def to_rgb(img):
    """``vision_prep.to_rgb``: transparency composited on white, anything else ``convert("RGB")``."""
    from PIL import Image

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, img.convert("RGBA")).convert("RGB")
    return img.convert("RGB")


def main() -> int:
    """Write the files and golden.jsonl (refused unless Pillow is 12.3 with libwebp 1.6.0, the versions ported)."""
    import PIL
    from PIL import Image, features

    pillow, libwebp = PIL.__version__, features.version("webp")
    if not pillow.startswith("12.3.") or libwebp != "1.6.0":
        raise SystemExit(
            f"the goldens are Pillow 12.3 with libwebp 1.6.0; this is Pillow {pillow} with libwebp {libwebp}"
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "golden.jsonl", "w") as g:
        g.write(
            json.dumps({"generator": {"pillow": pillow, "libwebp": libwebp}}) + "\n"
        )
        for name, raw in images(a.full):
            (out / name).write_bytes(raw)
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
            except (OSError, SyntaxError, ValueError):
                g.write(json.dumps({"name": name, "refused": True}) + "\n")
                continue
            rgb = to_rgb(img)
            rec = {
                "name": name,
                "size": list(rgb.size),
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
            g.write(json.dumps(rec) + "\n")
    print(f"wrote {out}/golden.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
