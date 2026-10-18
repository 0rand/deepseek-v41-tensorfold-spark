"""DeepSeek-V4.1 tokenizer goldens: a varied corpus encoded by HF ``tokenizers`` with the release tokenizer.json."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

SENTENCES = [
    "The quick brown fox jumps over the lazy dog. It's 3:45pm on 2026-10-06, and the price is $1,234.56!",
    "我们今天去公园散步，天气非常好。人工智能正在改变世界，深度学习模型越来越大。",
    "東京は日本の首都です。カタカナとひらがなとカンジ。ラーメンを食べたい！",
    "한국어 텍스트도 포함합니다. 서울은 아름다운 도시입니다.",
    "Привет, мир! Это тест токенизатора с кириллицей и цифрами 12345.",
    "مرحبا بالعالم، هذا اختبار للنص العربي ١٢٣٤٥٦.",
    "שלום עולם, זה מבחן.",
    "नमस्ते दुनिया, यह एक परीक्षण है। संख्या ४२",
    "Ελληνικά: Καλημέρα κόσμε! αβγδε ΑΒΓΔΕ",
    "Emoji: 😀😃😄 👨‍👩‍👧‍👦 🏳️‍🌈 👍🏽 ❤️‍🔥 🇯🇵🇺🇸",
    "Math: ∑_{i=1}^{n} x_i² ≤ ∫₀^∞ e^{-x} dx = 1; α≈β±γ → ∞ ≠ ∅ ⊂ ℝ",
    "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n\nprint(fib(30))\n",
    "  leading spaces\tand\ttabs\r\nwindows lines\r\n\r\n\n\n   trailing   \n",
    "Ünïcödé çømbïnïng: é à ñ ö ́alone ​zero‌width‍",
    "Numbers: 1 12 123 1234 12345 123456 1234567 3.14159 -42 1e10 0x1F ١٢٣ ⅷ ① ²³",
    "URLs: https://example.com/path?q=1&r=2#frag, mail: someone@example.org",
    '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "units": ["c", "f"]}}',
    "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75<｜User｜>hi<｜Assistant｜><think>ok</think>yes<｜end▁of▁sentence｜>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"true\">1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "<​｜deepseek_image｜> escaped, <｜deepseek_image｜> real, <dsml:x></dsml:x> <|EOT|> <｜place▁holder▁no▁7｜>",
    "'s 't 're 've 'm 'll 'd 'S 'T it's I'M they'LL",
    "!!!??? ... --- *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}} <<< >>> ||| ~~~ ``` ;;; ::: ,,,",
    "a.b,c;d:e!f?g(h)i[j]k{l}m<n>o/p\\q|r'_s\"t`u~v@w#x$y%z^&*+-=",
    "Ⅳ ⅳ ½ ¾ ⁴ ₄ ° ℃ ™ © ® § ¶ † ‡ • … ‰ ′ ″ ‹ › « » € £ ¥ ₩ ₹ ₿",
    "　全角スペース　と、句読点。「かぎかっこ」・中黒ー長音",
    "\x00\x01\x02\x7f control \x1b[31mred\x1b[0m \x0b\x0c",
]


def code_files(dirs: list[Path], limit: int) -> list[str]:
    out = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix not in (".py", ".zig", ".md", ".json", ".cu", ".c", ".h", ".sh", ".txt", ".toml"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if 0 < len(text) < 200_000:
                out.append(text)
            if len(out) >= limit:
                return out
    return out


def random_codepoints(rng: random.Random, n: int) -> str:
    ranges = [(0x20, 0x7E), (0x09, 0x0D), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
              (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x2000, 0x206F), (0x2070, 0x22FF),
              (0x2460, 0x27BF), (0x3000, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3),
              (0xE000, 0xE0FF), (0xFE00, 0xFE0F), (0xFF00, 0xFFEF), (0x10000, 0x1007F), (0x1D400, 0x1D7FF),
              (0x1F300, 0x1FAFF), (0x20000, 0x2A6DF), (0xE0000, 0xE007F), (0x0, 0x10FFFF)]
    out = []
    for _ in range(n):
        lo, hi = rng.choice(ranges)
        cp = rng.randint(lo, hi)
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0x41
        out.append(chr(cp))
    return "".join(out)


def mixed(rng: random.Random, pool: list[str], specials: list[str]) -> str:
    parts = []
    for _ in range(rng.randint(1, 12)):
        r = rng.random()
        if r < 0.35:
            s = rng.choice(pool)
            a = rng.randint(0, len(s))
            parts.append(s[a:a + rng.randint(1, 400)])
        elif r < 0.5:
            parts.append(rng.choice(specials))
        elif r < 0.75:
            parts.append(random_codepoints(rng, rng.randint(1, 60)))
        elif r < 0.85:
            parts.append(rng.choice([" ", "  ", "\n", "\n\n", "\t", "\r\n", " \n ", "   \n\n  ", "　", "\xa0"]))
        else:
            parts.append(str(rng.randint(0, 10 ** rng.randint(1, 15))))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--docs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--src", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    added = json.loads((args.model_dir / "tokenizer.json").read_text())["added_tokens"]
    specials = [a["content"] for a in added]
    rng = random.Random(args.seed)
    files = code_files(args.src, 600)
    pool = SENTENCES + files
    docs = list(SENTENCES) + files
    docs += [s * rng.randint(2, 20) for s in SENTENCES]
    while len(docs) < args.docs:
        docs.append(mixed(rng, pool, specials))
    docs.append("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.perf_counter()
    encoded = [tok.encode(d, add_special_tokens=False).ids for d in docs]
    dt = time.perf_counter() - t0
    with open(args.out_dir / "tokenizer.golden", "wb") as f:
        for d, ids in zip(docs, encoded):
            b = d.encode("utf-8", "surrogatepass")
            total += len(b)
            f.write(struct.pack("<I", len(b)) + b + struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids))
    # decode goldens: every doc's ids decoded back (skip_special_tokens=False), as the server's detokenizer must give
    with open(args.out_dir / "decode.golden", "wb") as f:
        for ids in encoded[:2000]:
            text = tok.decode(ids, skip_special_tokens=False).encode("utf-8", "surrogatepass")
            f.write(struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids) + struct.pack("<I", len(text)) + text)
        for _ in range(500):      # random id runs: partial UTF-8 bytes, specials, anything
            ids = [rng.randrange(0, 129280) for _ in range(rng.randint(1, 40))]
            text = tok.decode(ids, skip_special_tokens=False).encode("utf-8", "surrogatepass")
            f.write(struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids) + struct.pack("<I", len(text)) + text)
    n = sum(len(e) for e in encoded)
    print(f"{len(docs)} docs, {total / 1e6:.2f} MB, {n} tokens; HF tokenizers single-thread encode "
          f"{total / dt / 1e6:.2f} MB/s ({n / dt / 1e6:.2f} M tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
