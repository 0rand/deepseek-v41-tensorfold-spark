#!/usr/bin/env python3
"""Speed, long-context retrieval and exactness through the serving path (standard library only).

Against a running OpenAI-compatible server (the Zig server of docs/ZIG-SERVE.md, or any other). Every request is
temperature 0 with a fixed seed, thinking off (--thinking turns it on), streamed with stream_options.include_usage
unless noted. The suites (--suite, comma separated; default prefill,decode,multi,needle):

  prefill     a cold prompt of each --lengths tokens (default 1K,8K,32K,131K), max_tokens 1: TTFT, prefill tok/s =
              (prompt - cached) tokens / TTFT, and the server's own prefill seconds when the reply carries them
              (the Zig server's ``tensorfold`` block). Random filler (--family prose|code) from --seed and a salt that
              starts the prompt, so the prefix cache cannot serve it; cached tokens are reported and flagged.
  decode      one stream of --tokens tokens for each --prompts prompt: TTFT, tok/s = (completion - 1) / (last token -
              first token), and tokens per decode round / accepted over drafted when the server reports rounds.
  multi       --streams concurrent streams released together (a barrier) for each prompt. Per stream: its own tok/s.
              Over the cell: wall_tok_s = (sum of tokens - N) / (latest last token - earliest first token), the same
              definition as Mia's tools/bench.py; live_tok_s = tokens streamed while all N are emitting, over that span;
              first_spread_s = latest first token - earliest (admission / prefill stagger). sum_per_stream_tok_s, the
              sum of each stream's own rate, is kept only to show how far it overstates the total.
  needle      a unique code hidden at each --depths (default 0.1,0.5,0.9) of a haystack of each --needle-lengths
              tokens, asked for; a cell passes when the reply contains the code. A fresh haystack a cell (cold TTFT).
  interleave  one stream decodes (ignore_eos, --interleave-tokens); once it is emitting, a cold prompt of each
              --interleave-lengths tokens (default 131072, max_tokens 1) is sent. Reported: the decoding stream's
              longest and p99 gap between tokens while that prompt runs, its tokens in that time, the long prompt's
              TTFT. --cancel-at S closes the decoding client S seconds into the long prompt and polls /health (100 ms)
              for how long the server takes to drop it; the output says when that is not measurable.
  exact       a fixed prompt set (code, prose, the printing-press essay, a 2K document, 4 concurrent and the same 4
              alone, a repeat), non-streamed, T0: SHA-256 of each reply's text (and of its token ids when the server
              returns them for return_token_ids). Without --exact-ref the run is the reference (write it with --out);
              with --exact-ref FILE each case is compared with FILE's: equal or different, exit 1 on any difference.
  reuse       an agent conversation (a shared ~2K-token system prompt, --reuse-history tokens of history), then
              --reuse-rounds rounds of another conversation with the same system prompt followed by the agent's next
              turn: each agent turn must report >= --reuse-min (0.9) of its prompt as cached
              (usage.prompt_tokens_details.cached_tokens, else tensorfold.cached). No counter: "unmeasured", exit 0
              unless --require-counter.
  anthropic   /v1/messages: the message shape and usage; /v1/messages/count_tokens within --count-tolerance tokens
              of usage.input_tokens; a streamed tool_use whose input_json_delta fragments join into valid JSON with
              an array argument; a tool_result turn; stop_sequences (stop_reason stop_sequence, stop_sequence set);
              the 400 error shape (type error, error.type invalid_request_error).

  python3 bench/speed.py --url http://127.0.0.1:8000 --out runs/base.json
  python3 bench/speed.py --suite decode,multi --prompts mia --streams 1,2,4 --reps 3 --out runs/x.json
  python3 bench/speed.py --suite interleave --interleave-lengths 131072 --cancel-at 10 --health
  python3 bench/speed.py --suite exact --out ref.json; python3 bench/speed.py --suite exact --exact-ref ref.json
  python3 bench/speed.py --compare runs/base.json runs/x.json [more.json ...]    # no server needed

--compare prints each cell's value per file and its change against the first file, and lists every setting that
differs between the files (model, prompts, tokens, thinking, filler, ...) so mismatched runs are not compared
silently; --strict makes such a difference exit 1. It also reads Mia's tools/bench.py JSON (its decode and multi
cells use the same printing-press prompt as --prompts mia here; its prefill filler differs from ours).

Not in the default list: interleave, exact, reuse, anthropic (name them in --suite).

--health / --metrics store GET /health and /metrics before and after the run, and /health's counter deltas (rounds,
drafted, accepted, graph eager steps and floor hits on the Zig server). The JSON (--out) holds this tool's SHA-256
and git revision, the settings, every request's raw figures and the medians. Exit 1 on a request error, a missed
needle, an exact difference, a reuse turn under --reuse-min or a failed anthropic check.

The suites, the wall-window aggregate and the needle / exactness / interleave checks follow Mia's recipe
(MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks: tools/bench.py, tools/needle.py, tools/exact.py; the interleave
measurement is from Coinupbtc's prompt-interleave work there; the reuse check from tools/prompt_reuse.py, which
came from Mia's GLM-5.3-Flash recipe; the anthropic checks from tools/anthropiccheck.py, written by Eduardo Florencio
(eduffd), Apache-2.0). This file is our own implementation; the printing-press
prompt text is from tools/bench.py in MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks, Apache-2.0, Copyright 2026
MiaAI-Lab (see NOTICE), kept verbatim so decode numbers are comparable with hers.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import itertools
import json
import math
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from redact import known_secrets, mask_values, redact_argv, redact_text

SCHEMA = 1
TOOL = "bench/speed.py"
FILLER_VERSION = 1  # bump when filler() or the needle text changes: --compare flags it
EXACT_SET = 1  # bump when the exact prompt set changes
DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-TF"

CODE = (
    "Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked "
    "list, with docstrings and type hints, then three unit tests with pytest."
)
CODE_MORE = (
    (
        "Write a Python class `Trie` with `insert(word)`, `search(word)` and `starts_with(prefix)`, with docstrings and "
        "type hints, then three unit tests with pytest."
    ),
    (
        "Write a Python class `TokenBucket` rate limiter with `allow(now: float) -> bool`, refill rate and burst size, "
        "with docstrings and type hints, then three unit tests with pytest."
    ),
    (
        "Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs in O(n log n), "
        "with a docstring and type hints, then three unit tests with pytest."
    ),
)
PROSE = (
    "Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what "
    "replaced them. Plain prose, no lists or headings."
)
PROSE_MORE = (
    (
        "Write a 400-word essay on why canals were dug where they were, how the people who worked them lived, and what "
        "replaced them. Plain prose, no lists or headings."
    ),
    (
        "Write a 400-word essay on why early observatories were built where they were, how their astronomers lived, and "
        "what replaced them. Plain prose, no lists or headings."
    ),
    (
        "Write a 400-word essay on why windmills were built where they were, how their millers lived, and what replaced "
        "them. Plain prose, no lists or headings."
    ),
)
# Prompt text from tools/bench.py in MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks (Apache-2.0, Copyright 2026
# MiaAI-Lab), verbatim; she appends "<salt>-<rep>" (decode) or "<salt>-m<N>-<rep>-<i>" (multi), and so do we.
MIA_DECODE = (
    "Write a long, detailed essay about the history of the printing press, with many paragraphs. "
    "Do not stop early. Variation tag: "
)
PROMPTS = {
    "code": (CODE, *CODE_MORE),
    "prose": (PROSE, *PROSE_MORE),
    "mia": (MIA_DECODE,),
}

WORDS = [
    "river",
    "mountain",
    "lantern",
    "copper",
    "harbor",
    "meadow",
    "quartz",
    "signal",
    "orchard",
    "canvas",
    "thunder",
    "velvet",
    "compass",
    "ember",
    "glacier",
    "timber",
    "falcon",
    "prairie",
    "marble",
    "beacon",
    "willow",
    "cobalt",
    "summit",
    "ferry",
    "saddle",
    "tundra",
    "pepper",
    "violet",
    "anchor",
    "bramble",
    "cinder",
    "delta",
    "fjord",
    "granite",
    "hollow",
    "ivory",
    "jasmine",
    "kettle",
    "lagoon",
    "mosaic",
    "nectar",
    "oyster",
]
VERBS = [
    "load",
    "parse",
    "merge",
    "scale",
    "shift",
    "split",
    "count",
    "fold",
    "build",
    "check",
]
VAULTS = [
    "amber",
    "basalt",
    "cedar",
    "dune",
    "elm",
    "flint",
    "garnet",
    "heron",
    "iris",
    "juniper",
]
# Tokens a filler unit (a word of prose, a line of code) when /tokenize is not available: an estimate only.
EST_TOKENS_PER_UNIT = {"prose": 1.2, "code": 14.0}

_STOP = threading.Event()


# ---------------------------------------------------------------------------------------------------------- helpers
def parse_ints(text: str) -> list[int]:
    """'1K,8k,32768,1M' -> [1024, 8192, 32768, 1048576] (K = 1024, M = 1024 * 1024)."""
    out = []
    for x in (t.strip() for t in str(text).split(",")):
        if not x:
            continue
        mult = 1
        if x[-1] in "kK":
            x, mult = x[:-1], 1024
        elif x[-1] in "mM":
            x, mult = x[:-1], 1024 * 1024
        out.append(int(float(x) * mult))
    return out


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.median(xs) if xs else None


def percentile(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(1, math.ceil(q / 100 * len(s)))
    return s[k - 1]


def rnd(x, n: int = 4):
    return None if x is None else round(x, n)


def sha256(data: str | bytes) -> str:
    return hashlib.sha256(
        data if isinstance(data, bytes) else data.encode("utf-8")
    ).hexdigest()


def tool_identity() -> dict:
    here = Path(__file__).resolve()
    ident = {"tool": TOOL, "schema": SCHEMA, "sha256": sha256(here.read_bytes())}
    try:
        rev = subprocess.run(
            ["git", "-C", str(here.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(here.parent), "status", "--porcelain", "--", here.name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if rev.returncode == 0:
            ident["git"] = rev.stdout.strip()
            ident["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return ident


# ----------------------------------------------------------------------------------------------------------- filler
def filler(family: str, units: int, seed: str) -> str:
    """Deterministic random text: ``units`` words of prose or lines of code, from a string seed."""
    rng = random.Random(seed)
    if family == "code":
        lines = []
        for i in range(units):
            a, b = rng.sample(WORDS, 2)
            lines.append(
                f"def {rng.choice(VERBS)}_{a}_{i}({a}, {b}): return {a} {rng.choice('+-*')} {b} * "
                f"{rng.randint(2, 97)}"
            )
        return "\n".join(lines)
    out: list[str] = []
    while len(out) < units:
        n = rng.randint(6, 16)
        sentence = [rng.choice(WORDS) for _ in range(n)]
        sentence[0] = sentence[0].capitalize()
        sentence[-1] += "."
        out.extend(sentence)
    return " ".join(out[:units])


def insert_at(text: str, family: str, depth: float, fact: str) -> str:
    """``fact`` inserted at a sentence (prose) or line (code) boundary at fraction ``depth`` of ``text``."""
    sep = "\n" if family == "code" else ". "
    parts = text.split(sep)
    at = min(len(parts), max(0, round(len(parts) * depth)))
    if family == "code":
        fact = "# " + fact
    elif at < len(parts):
        fact = fact.rstrip(".")
    parts.insert(at, fact)
    return sep.join(parts)


def needle_fact(seed: str) -> tuple[str, str, str]:
    """(vault name, code, the sentence): a 7-digit code a cell, from its seed."""
    rng = random.Random("needle-" + seed)
    vault, code = rng.choice(VAULTS), str(rng.randrange(1_000_000, 10_000_000))
    return vault, code, f"The access code for the {vault} vault is {code}."


def needle_score(reply: str, code: str) -> dict:
    """found: the code appears as a whole number in the reply; exact: the reply is the code and nothing else."""
    numbers = re.findall(r"\d+", reply or "")
    return {
        "found": code in numbers,
        "exact": (reply or "").strip().strip(".`'\"*").strip() == code,
        "numbers": numbers[:8],
    }


def make_prompt(name: str, index: int, tag: str) -> str:
    """Prompt ``name`` (variant ``index``) with the run's tag: Mia's appends it, ours lead with it."""
    variants = PROMPTS[name]
    text = variants[index % len(variants)]
    return text + tag if name == "mia" else f"[{tag}] {text}"


# ----------------------------------------------------------------------------------------------------------- client
class Client:
    def __init__(
        self,
        url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float = 3600,
        thinking: bool = False,
        seed: int = 1234,
        ignore_eos: bool = False,
    ):
        u = urllib.parse.urlsplit(url.rstrip("/"))
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SystemExit(f"--url must be http(s)://host[:port], not {url!r}")
        self.url, self.https, self.host = (
            url.rstrip("/"),
            u.scheme == "https",
            u.hostname,
        )
        self.port = u.port or (443 if self.https else 80)
        self.prefix = u.path.rstrip("/")
        self.model, self.timeout, self.thinking, self.seed, self.ignore_eos = (
            model,
            timeout,
            thinking,
            seed,
            ignore_eos,
        )
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def conn(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout or self.timeout)

    def get(self, path: str, timeout: float = 15) -> tuple[int, str]:
        c = self.conn(timeout)
        try:
            c.request("GET", self.prefix + path, headers=self.headers)
            r = c.getresponse()
            return r.status, r.read().decode("utf-8", "replace")
        finally:
            c.close()

    def get_json(self, path: str, timeout: float = 15):
        try:
            status, text = self.get(path, timeout)
        except (OSError, http.client.HTTPException) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        try:
            return json.loads(text)
        except ValueError:
            return {"error": f"HTTP {status}: not JSON", "text": text[:2000]}

    def post_json(self, path: str, body: dict, timeout: float | None = None) -> dict:
        c = self.conn(timeout)
        try:
            c.request(
                "POST",
                self.prefix + path,
                body=json.dumps(body).encode(),
                headers=self.headers,
            )
            r = c.getresponse()
            text = r.read().decode("utf-8", "replace")
            if r.status >= 400:
                return {"error": f"HTTP {r.status}: {text[:300]}"}
            return json.loads(text)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            c.close()

    def anthropic(self, path: str, body: dict) -> tuple[int, dict]:
        """POST to an Anthropic Messages route: (status, the JSON body); status 0 when the request failed."""
        headers = {**self.headers, "anthropic-version": "2023-06-01"}
        if "Authorization" in headers:
            headers["x-api-key"] = headers["Authorization"].split(" ", 1)[1]
        c = self.conn()
        try:
            c.request(
                "POST",
                self.prefix + path,
                body=json.dumps(body).encode(),
                headers=headers,
            )
            r = c.getresponse()
            text = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(text) if text.strip() else {}
            except ValueError:
                return r.status, {"text": text[:300]}
        except (OSError, http.client.HTTPException) as exc:
            return 0, {"error_text": f"{type(exc).__name__}: {exc}"}
        finally:
            c.close()

    def anthropic_stream(self, path: str, body: dict) -> list[dict]:
        """A streamed Anthropic Messages request: its SSE data events (a failure is one {"type": "client_error"})."""
        headers = {**self.headers, "anthropic-version": "2023-06-01"}
        if "Authorization" in headers:
            headers["x-api-key"] = headers["Authorization"].split(" ", 1)[1]
        c = self.conn()
        events: list[dict] = []
        try:
            c.request(
                "POST",
                self.prefix + path,
                body=json.dumps(body).encode(),
                headers=headers,
            )
            r = c.getresponse()
            if r.status >= 400:
                return [
                    {
                        "type": "client_error",
                        "error": f"HTTP {r.status}: {r.read()[:300]!r}",
                    }
                ]
            while True:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    try:
                        events.append(json.loads(line[5:]))
                    except ValueError:
                        events.append(
                            {
                                "type": "client_error",
                                "error": "event data is not JSON: " + line[:120],
                            }
                        )
                    if events[-1].get("type") == "message_stop":
                        break
        except (OSError, http.client.HTTPException) as exc:
            events.append(
                {"type": "client_error", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            c.close()
        return events

    def body(
        self,
        messages: list,
        max_tokens: int,
        *,
        stream: bool,
        extra: dict | None = None,
    ) -> dict:
        b = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if stream:
            b.update(stream=True, stream_options={"include_usage": True})
        if self.ignore_eos:
            b.update(ignore_eos=True, min_tokens=max_tokens)
        b.update(extra or {})
        return b

    def stream(
        self,
        messages: list,
        max_tokens: int,
        *,
        gate: threading.Barrier | None = None,
        extra: dict | None = None,
        holder: dict | None = None,
        on_event=None,
    ) -> dict:
        """One streamed chat request. Never raises: a failure is {"error": ...}. ``holder`` gets the connection
        (``cancel`` closes it from another thread); ``on_event(n, t)`` runs at each token event."""
        holder = holder if holder is not None else {}
        payload = json.dumps(
            self.body(messages, max_tokens, stream=True, extra=extra)
        ).encode()
        c = self.conn()
        holder["conn"] = c
        if gate is not None:
            try:
                gate.wait(timeout=self.timeout)
            except threading.BrokenBarrierError:
                return {"error": "barrier broken"}
        out: dict = {
            "t_send": time.monotonic(),
            "times": [],
            "text": "",
            "reasoning_chars": 0,
        }
        text: list[str] = []
        r = None
        try:
            c.request(
                "POST",
                self.prefix + "/v1/chat/completions",
                body=payload,
                headers=self.headers,
            )
            # getresponse() drops c.sock for a Connection: close reply
            holder["sock"] = c.sock
            if holder.get("cancelled"):
                cancel(holder)
            r = c.getresponse()
            if r.status >= 400:
                out["error"] = (
                    f"HTTP {r.status}: {r.read().decode('utf-8', 'replace')[:300]}"
                )
                return out
            while True:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):
                    out["error"] = f"in-stream error: {json.dumps(ev['error'])[:300]}"
                    break
                if ev.get("usage"):
                    out["usage"] = ev["usage"]
                if isinstance(ev.get("tensorfold"), dict):
                    out["server"] = ev["tensorfold"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    thought = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if piece or thought:
                        t = time.monotonic()
                        out["times"].append(t)
                        text.append(piece)
                        out["reasoning_chars"] += len(thought)
                        if on_event is not None:
                            on_event(len(out["times"]), t)
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
        except (OSError, http.client.HTTPException) as exc:
            if not holder.get("cancelled"):
                out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            out["t_end"] = time.monotonic()
            out["text"] = "".join(text)
            out["cancelled"] = bool(holder.get("cancelled"))
            if r is not None:
                r.close()  # the response holds the socket open after the connection is closed
            c.close()
        if not out["times"] and "error" not in out and not out["cancelled"]:
            out["error"] = "no streamed tokens"
        return out


def cancel(holder: dict) -> None:
    """Close a streaming request's connection from another thread (the client leaves)."""
    holder["cancelled"] = True
    sock = holder.get("sock") or getattr(holder.get("conn"), "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def figures(r: dict) -> dict:
    """A streamed request's figures from its raw record (times are monotonic seconds)."""
    out = {
        k: r[k] for k in ("error", "finish_reason", "cancelled") if r.get(k) is not None
    }
    times = r.get("times") or []
    usage = r.get("usage") or {}
    srv = r.get("server") or {}
    pt = usage.get("prompt_tokens", srv.get("prompt"))
    ct = usage.get("completion_tokens", srv.get("completion"))
    if ct is None and times:
        ct = len(times)
    cached = (usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens", srv.get("cached")
    )
    out.update(
        prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached, events=len(times)
    )
    if times:
        out["ttft_s"] = rnd(times[0] - r["t_send"])
        if pt is not None and out["ttft_s"]:
            out["prefill_tok_s"] = rnd((pt - (cached or 0)) / out["ttft_s"], 1)
        if ct and ct > 1 and times[-1] > times[0]:
            out["decode_tok_s"] = rnd((ct - 1) / (times[-1] - times[0]), 2)
    if srv:
        out["server"] = {k: v for k, v in srv.items() if k != "token_ids"}
        if srv.get("prefill_s") and pt is not None:
            out["server_prefill_tok_s"] = rnd(
                (pt - (cached or 0)) / srv["prefill_s"], 1
            )
        if srv.get("decode_s") and ct:
            out["server_decode_tok_s"] = rnd(ct / srv["decode_s"], 2)
        rounds = srv.get("rounds")
        if rounds:
            out["tokens_per_round"] = rnd(
                (srv.get("completion") or ct or 0) / rounds, 3
            )
        if srv.get("drafted"):
            out["accept_rate"] = rnd((srv.get("accepted") or 0) / srv["drafted"], 3)
    return out


# ------------------------------------------------------------------------------------------------------------- math
def totals(streams: list[dict]) -> dict:
    """A multi-stream cell's totals from each stream's token-event times (``times``) and token count (``tokens``).

    wall_tok_s: sum(tokens - 1) / (latest last event - earliest first event), Mia's tools/bench.py definition.
    live_tok_s: tokens streamed in the span where every stream is between its first and last event, over that span;
    an event counts as the stream's (tokens - 1) / (events - 1) tokens (one per event when the server streams token by
    token, a decode round's accepted tokens when it streams a round). first_spread_s: latest first - earliest first.
    sum_per_stream_tok_s: the sum of the streams' own rates, each over its own window (overstates a staggered cell)."""
    ok = [s for s in streams if s.get("times") and s.get("tokens")]
    per = [
        (s["tokens"] - 1) / (s["times"][-1] - s["times"][0])
        if s["tokens"] > 1 and len(s["times"]) > 1 and s["times"][-1] > s["times"][0]
        else None
        for s in ok
    ]
    res = {
        "streams_ok": len(ok),
        "per_stream_tok_s": [rnd(p, 2) for p in per],
        "mean_per_stream_tok_s": rnd(statistics.mean([p for p in per if p]), 2)
        if any(per)
        else None,
        "sum_per_stream_tok_s": rnd(sum(p for p in per if p), 2) if any(per) else None,
        "wall_tok_s": None,
        "window_s": None,
        "live_tok_s": None,
        "live_span_s": None,
        "first_spread_s": None,
    }
    if not ok:
        return res
    first, last = min(s["times"][0] for s in ok), max(s["times"][-1] for s in ok)
    lo, hi = max(s["times"][0] for s in ok), min(s["times"][-1] for s in ok)
    res["first_spread_s"] = rnd(lo - first)
    if last > first:
        res["window_s"] = rnd(last - first)
        res["wall_tok_s"] = rnd(sum(s["tokens"] - 1 for s in ok) / (last - first), 2)
    if len(ok) > 1 and hi > lo:
        inside = 0.0
        for s in ok:
            t = s["times"]
            each = (s["tokens"] - 1) / (len(t) - 1) if len(t) > 1 else 0.0
            inside += sum(1 for x in t if lo < x <= hi) * each
        res["live_tok_s"], res["live_span_s"] = rnd(inside / (hi - lo), 2), rnd(hi - lo)
    elif len(ok) == 1:
        res["live_tok_s"], res["live_span_s"] = res["wall_tok_s"], res["window_s"]
    return res


def gaps_during(times: list[float], lo: float, hi: float) -> list[float]:
    """Gaps between consecutive events of a stream that overlap [lo, hi]; a stream still running past ``hi`` with no
    event since its last one before ``hi`` counts that open gap up to ``hi``."""
    out = [b - a for a, b in itertools.pairwise(times) if b > lo and a < hi]
    before = [t for t in times if t < hi]
    if before and before[-1] == times[-1] and hi > max(before[-1], lo):
        out.append(hi - max(before[-1], lo))
    return out


# ----------------------------------------------------------------------------------------------------------- suites
class Run:
    def __init__(self, a: argparse.Namespace, client: Client):
        self.a, self.c = a, client
        self.salt = (
            a.salt
            if a.salt is not None
            else f"{random.SystemRandom().randrange(16**8):08x}"
        )
        self.tpu: dict[str, float] = {}
        self.sizing: dict[str, str] = {}
        self.notes: list[str] = []

    def log(self, msg: str) -> None:
        if not self.a.quiet:
            print(msg, flush=True)

    def tokens_per_unit(self, family: str) -> float:
        """Filler tokens a unit, measured once with /tokenize (an estimate when the server has none)."""
        if family not in self.tpu:
            n = 400
            probe = filler(family, n, "calibrate")
            base = self.c.post_json(
                "/tokenize",
                {"model": self.c.model, "messages": [{"role": "user", "content": "x"}]},
            )
            full = self.c.post_json(
                "/tokenize",
                {
                    "model": self.c.model,
                    "messages": [{"role": "user", "content": "x\n" + probe}],
                },
            )
            try:
                got = int(full.get("count") or len(full["tokens"])) - int(
                    base.get("count") or len(base["tokens"])
                )
                if got <= 0:
                    raise ValueError(got)
                self.tpu[family], self.sizing[family] = got / n, "tokenize"
            except (KeyError, TypeError, ValueError):
                self.tpu[family], self.sizing[family] = (
                    EST_TOKENS_PER_UNIT[family],
                    "estimate",
                )
                self.notes.append(
                    f"/tokenize unavailable: {family} prompts sized by an estimate; see prompt_tokens"
                )
        return self.tpu[family]

    def document(self, family: str, tokens: int, seed: str) -> str:
        return filler(family, max(1, int(tokens / self.tokens_per_unit(family))), seed)

    def show(self, label: str, f: dict) -> None:
        if f.get("error"):
            self.log(f"  {label}: ERROR {f['error']}")
            return
        bits = [f"prompt {f.get('prompt_tokens')}"]
        if f.get("cached_tokens"):
            bits.append(f"cached {f['cached_tokens']}")
        bits += [
            f"ttft {f.get('ttft_s')} s",
            f"prefill {f.get('prefill_tok_s')} tok/s",
            f"out {f.get('completion_tokens')}",
            f"decode {f.get('decode_tok_s')} tok/s",
        ]
        if f.get("tokens_per_round"):
            bits.append(f"tpr {f['tokens_per_round']}")
        self.log(f"  {label}: " + ", ".join(bits))

    # -- prefill
    def prefill(self) -> list:
        rows = []
        for n in self.a.lengths:
            for rep in range(self.a.reps):
                seed = f"{self.a.seed}-prefill-{n}-{rep}-{self.salt}"
                prompt = f"[{self.salt} prefill {n} {rep}]\n" + self.document(
                    self.a.family, n, seed
                )
                prompt += "\n\nReply with the single word: ok"
                f = figures(self.c.stream([{"role": "user", "content": prompt}], 1))
                f.update(suite="prefill", target_tokens=n, rep=rep)
                if (f.get("cached_tokens") or 0) > max(
                    32, 0.01 * (f.get("prompt_tokens") or 0)
                ):
                    f["cache_hit"] = True
                    self.notes.append(
                        f"prefill {n} rep {rep}: {f['cached_tokens']} tokens came from the prefix cache"
                    )
                rows.append(f)
                self.show(f"prefill {n} rep {rep}", f)
        return rows

    # -- decode
    def decode(self) -> list:
        rows = []
        for name in self.a.prompts:
            for rep in range(self.a.reps):
                prompt = make_prompt(name, 0, f"{self.salt}-{rep}")
                f = figures(
                    self.c.stream([{"role": "user", "content": prompt}], self.a.tokens)
                )
                f.update(suite="decode", prompt=name, rep=rep)
                if self.a.lean:
                    f.pop("server", None)
                rows.append(f)
                self.show(f"decode {name} rep {rep}", f)
        return rows

    # -- multi
    def multi(self) -> list:
        rows = []
        for name in self.a.prompts:
            for n in self.a.streams:
                for rep in range(self.a.reps):
                    gate, raw = threading.Barrier(n), [None] * n

                    def work(
                        i: int, gate=gate, raw=raw, n=n, rep=rep, name=name
                    ) -> None:
                        prompt = make_prompt(name, i, f"{self.salt}-m{n}-{rep}-{i}")
                        raw[i] = self.c.stream(
                            [{"role": "user", "content": prompt}],
                            self.a.tokens,
                            gate=gate,
                        )

                    threads = [
                        threading.Thread(target=work, args=(i,)) for i in range(n)
                    ]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    figs = [figures(r) for r in raw]
                    streams = [
                        {"times": r["times"], "tokens": f.get("completion_tokens")}
                        for r, f in zip(raw, figs, strict=True)
                        if not f.get("error")
                    ]
                    row = {
                        "suite": "multi",
                        "prompt": name,
                        "streams": n,
                        "rep": rep,
                        "errors": [f["error"] for f in figs if f.get("error")],
                        **totals(streams),
                        "ttft_s": [f.get("ttft_s") for f in figs],
                        "completion_tokens": [f.get("completion_tokens") for f in figs],
                        "tokens_per_round": median(
                            [f.get("tokens_per_round") for f in figs]
                        ),
                    }
                    rows.append(row)
                    self.log(
                        f"  multi {name} x{n} rep {rep}: wall {row['wall_tok_s']} tok/s, all-live "
                        f"{row['live_tok_s']} tok/s, per stream {row['mean_per_stream_tok_s']}, first-token "
                        f"spread {row['first_spread_s']} s, errors {len(row['errors'])}"
                    )
        return rows

    # -- needle
    def needle(self) -> list:
        rows = []
        for n in self.a.needle_lengths:
            for depth in self.a.depths:
                for rep in range(self.a.needle_reps):
                    seed = f"{self.a.seed}-needle-{n}-{depth}-{rep}-{self.salt}"
                    vault, code, fact = needle_fact(seed)
                    hay = insert_at(
                        self.document(self.a.family, n, seed),
                        self.a.family,
                        depth,
                        fact,
                    )
                    prompt = (
                        f"[{self.salt} needle {n} {depth} {rep}]\n{hay}\n\nWhat is the access code for the "
                        f"{vault} vault in the text above? Answer with the code only."
                    )
                    r = self.c.stream(
                        [{"role": "user", "content": prompt}], self.a.needle_tokens
                    )
                    f = figures(r)
                    score = (
                        needle_score(r.get("text", ""), code)
                        if not f.get("error")
                        else {"found": False, "exact": False}
                    )
                    f.update(
                        suite="needle",
                        target_tokens=n,
                        depth=depth,
                        rep=rep,
                        code=code,
                        reply=r.get("text", "")[-200:],
                        passed=bool(score["found"]),
                        exact=bool(score["exact"]),
                    )
                    rows.append(f)
                    self.log(
                        f"  needle {n} depth {depth} rep {rep}: {'PASS' if f['passed'] else 'FAIL'} "
                        f"(prompt {f.get('prompt_tokens')}, ttft {f.get('ttft_s')} s, reply {f['reply'][-40:]!r})"
                    )
        return rows

    # -- interleave
    def interleave(self) -> list:
        rows = []
        for n in self.a.interleave_lengths:
            for rep in range(self.a.reps):
                rows.append(self.interleave_one(n, rep))
        return rows

    def health_count(self) -> int | None:
        h = self.c.get_json("/health", timeout=5)
        if not isinstance(h, dict):
            return None
        for k in ("inflight", "requests_running", "requests_in_progress"):
            if isinstance(h.get(k), int):
                return h[k]
        return None

    def interleave_one(self, n: int, rep: int) -> dict:
        row: dict = {"suite": "interleave", "target_tokens": n, "rep": rep}
        holder: dict = {}
        emitting = threading.Event()
        dec: dict = {}
        warm = self.a.interleave_warm

        def on_event(k: int, _t: float) -> None:
            if k >= warm:
                emitting.set()

        prompt = make_prompt(self.a.prompts[0], 0, f"{self.salt}-i{n}-{rep}")
        extra = {"ignore_eos": True, "min_tokens": self.a.interleave_tokens}
        th = threading.Thread(
            target=lambda: dec.update(
                self.c.stream(
                    [{"role": "user", "content": prompt}],
                    self.a.interleave_tokens,
                    extra=extra,
                    holder=holder,
                    on_event=on_event,
                )
            )
        )
        th.start()
        if not emitting.wait(timeout=self.a.timeout):
            cancel(holder)
            th.join()
            row["error"] = (
                f"the decoding stream never emitted {warm} tokens: {dec.get('error')}"
            )
            self.log(f"  interleave {n} rep {rep}: ERROR {row['error']}")
            return row
        if dec.get("t_end") is not None:
            row["error"] = "the decoding stream ended before the long prompt was sent"
            return row
        seed = f"{self.a.seed}-interleave-{n}-{rep}-{self.salt}"
        long_prompt = f"[{self.salt} interleave {n} {rep}]\n" + self.document(
            self.a.family, n, seed
        )
        long_prompt += "\n\nReply with the single word: ok"
        lng: dict = {}
        long_extra = (
            {"ignore_eos": False, "min_tokens": 0} if self.a.ignore_eos else None
        )
        lt = threading.Thread(
            target=lambda: lng.update(
                self.c.stream(
                    [{"role": "user", "content": long_prompt}], 1, extra=long_extra
                )
            )
        )
        t_submit = time.monotonic()
        lt.start()
        cancel_info = None
        if self.a.cancel_at is not None:
            cancel_info = self.cancel_probe(holder, lt, t_submit)
        lt.join()
        if not holder.get("cancelled"):
            cancel(holder)
        th.join()
        lf = figures(lng)
        times = dec.get("times") or []
        t_long_first = (
            lng["times"][0] if lng.get("times") else lng.get("t_end", time.monotonic())
        )
        cut = holder.get("t_cancel", t_long_first)
        hi = min(t_long_first, cut)
        g = gaps_during(times, t_submit, hi)
        pre = [b - a for a, b in itertools.pairwise(times) if b <= t_submit]
        dfig = figures(dec)
        each = (
            ((dfig.get("completion_tokens") or len(times)) - 1) / (len(times) - 1)
            if len(times) > 1
            else 1.0
        )
        during = sum(1 for t in times if t_submit < t <= hi)
        row.update(
            long=lf,
            long_ttft_s=lf.get("ttft_s"),
            long_prompt_tokens=lf.get("prompt_tokens"),
            long_cached_tokens=lf.get("cached_tokens"),
            window_s=rnd(hi - t_submit),
            longest_gap_s=rnd(max(g)) if g else None,
            p99_gap_s=rnd(percentile(g, 99)),
            median_gap_s=rnd(median(g)),
            base_gap_s=rnd(median(pre)),
            tokens_during=round(during * each),
            decode_tokens_total=dfig.get("completion_tokens"),
        )
        if lf.get("error"):
            row["error"] = "long prompt: " + lf["error"]
        if dec.get("error"):
            row["error"] = "decoding stream: " + dec["error"]
        last = times[-1] if times else 0.0
        if (
            not holder.get("t_cancel")
            and dec.get("finish_reason")
            and last < t_long_first
        ):
            row["warning"] = (
                "the decoding stream finished before the long prompt's first token: raise "
                "--interleave-tokens; the gaps cover only part of the pass"
            )
            self.notes.append(f"interleave {n} rep {rep}: {row['warning']}")
        if cancel_info is not None:
            row["cancel"] = cancel_info
        self.log(
            f"  interleave {n} rep {rep}: long prompt {row['long_prompt_tokens']} tokens, ttft "
            f"{row['long_ttft_s']} s; decoding stream longest gap {row['longest_gap_s']} s, p99 "
            f"{row['p99_gap_s']} s, {row['tokens_during']} tokens during the pass (gap before "
            f"{row['base_gap_s']} s)"
            + (f"; cancel: {cancel_info}" if cancel_info else "")
        )
        return row

    def cancel_probe(
        self, holder: dict, long_thread: threading.Thread, t_submit: float
    ) -> dict:
        """Close the decoding client --cancel-at seconds into the long prompt; poll /health for the drop."""
        wait = t_submit + self.a.cancel_at - time.monotonic()
        if wait > 0:
            long_thread.join(timeout=wait)
        if not long_thread.is_alive():
            return {
                "measured": False,
                "why": "the long prompt finished before --cancel-at",
            }
        now = self.health_count()
        holder["t_cancel"] = time.monotonic()
        cancel(holder)
        if now is None:
            return {
                "measured": False,
                "at_s": self.a.cancel_at,
                "why": "/health has no inflight / requests_running count",
            }
        deadline = holder["t_cancel"] + self.a.cancel_wait
        while time.monotonic() < deadline:
            k = self.health_count()
            if not long_thread.is_alive():
                return {
                    "measured": False,
                    "at_s": self.a.cancel_at,
                    "why": "the long prompt finished first, "
                    "so a drop in the count cannot be told apart from it",
                }
            if k is not None and k < now:
                return {
                    "measured": True,
                    "at_s": self.a.cancel_at,
                    "inflight_before": now,
                    "inflight_after": k,
                    "gone_s": rnd(time.monotonic() - holder["t_cancel"], 3),
                    "poll_s": 0.1,
                }
            time.sleep(0.1)
        return {
            "measured": False,
            "at_s": self.a.cancel_at,
            "why": f"the count did not drop in {self.a.cancel_wait} s"
            " (the server may only notice at its next write)",
        }

    # -- exact
    def exact_cases(self) -> list[tuple[str, str, int]]:
        doc = filler("prose", 1500, "exact-doc")
        cases = [
            ("code", f"[exact] {CODE}", 96),
            ("prose", f"[exact] {PROSE}", 96),
            ("mia", MIA_DECODE + "exact", 96),
            (
                "doc2k",
                f"[exact]\n{doc}\n\nSummarize the text above in one sentence.",
                64,
            ),
        ]
        group = [CODE_MORE[0], CODE_MORE[1], PROSE_MORE[0], PROSE_MORE[1]]
        cases += [(f"conc_{i + 1}", f"[exact] {p}", 64) for i, p in enumerate(group)]
        cases += [(f"solo_{i + 1}", f"[exact] {p}", 64) for i, p in enumerate(group)]
        cases.append(("repeat_code", f"[exact] {CODE}", 96))
        return cases

    def exact_one(self, name: str, prompt: str, max_tokens: int) -> dict:
        body = self.c.body(
            [{"role": "user", "content": prompt}],
            max_tokens,
            stream=False,
            extra={"return_token_ids": True},
        )
        r = self.c.post_json("/v1/chat/completions", body)
        row = {"case": name, "prompt_sha256": sha256(prompt), "max_tokens": max_tokens}
        if r.get("error"):
            row["error"] = r["error"]
            return row
        try:
            msg = r["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            row["error"] = "no choices[0].message in the reply"
            return row
        content = msg.get("content") or ""
        srv = r.get("tensorfold") if isinstance(r.get("tensorfold"), dict) else {}
        ids = srv.get("token_ids")
        row.update(
            text_sha256=sha256(content + "\0" + (msg.get("reasoning_content") or "")),
            completion_tokens=(r.get("usage") or {}).get("completion_tokens"),
            finish_reason=r["choices"][0].get("finish_reason"),
            text_head=content[:80],
        )
        if isinstance(ids, list):
            row["tokens_sha256"] = sha256(json.dumps(ids, separators=(",", ":")))
        if srv.get("token_sha"):
            row["server_token_sha"] = srv["token_sha"]
        return row

    def exact(self) -> list:
        cases = self.exact_cases()
        rows: dict[str, dict] = {}
        conc = [c for c in cases if c[0].startswith("conc_")]
        for name, prompt, mt in cases:
            if name.startswith("conc_"):
                if name == "conc_1":
                    res: list = [None] * len(conc)

                    def one(i: int, case: tuple, res: list = res) -> None:
                        res[i] = self.exact_one(*case)

                    ts = [
                        threading.Thread(target=one, args=(i, c))
                        for i, c in enumerate(conc)
                    ]
                    for t in ts:
                        t.start()
                    for t in ts:
                        t.join()
                    rows.update({r["case"]: r for r in res})
                continue
            rows[name] = self.exact_one(name, prompt, mt)
        out = [rows[c[0]] for c in cases]
        intra = [
            ("conc_" + str(i), "solo_" + str(i)) for i in range(1, len(conc) + 1)
        ] + [("repeat_code", "code")]
        for a_name, b_name in intra:
            a, b = rows[a_name], rows[b_name]
            a["same_as"] = b_name
            a["intra_equal"] = (
                not a.get("error")
                and not b.get("error")
                and exact_key(a) == exact_key(b)
            )
        for r in out:
            r["suite"] = "exact"
            self.log(
                f"  exact {r['case']}: "
                + (
                    f"ERROR {r['error']}"
                    if r.get("error")
                    else f"{exact_key(r)[1][:16]} ({r.get('completion_tokens')} tokens)"
                    + (
                        ""
                        if r.get("intra_equal", True)
                        else f" DIFFERS from {r['same_as']} in this run"
                    )
                )
            )
        return out

    # -- reuse
    def chat_once(self, messages: list, max_tokens: int = 64) -> dict:
        t0 = time.monotonic()
        r = self.c.post_json(
            "/v1/chat/completions", self.c.body(messages, max_tokens, stream=False)
        )
        out = {"seconds": rnd(time.monotonic() - t0, 3)}
        if r.get("error"):
            out["error"] = r["error"]
            return out
        usage = r.get("usage") or {}
        srv = r.get("tensorfold") if isinstance(r.get("tensorfold"), dict) else {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None and isinstance(srv.get("cached"), int):
            cached = srv["cached"]
        try:
            reply = r["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            reply = ""
        out.update(
            prompt_tokens=usage.get("prompt_tokens"), cached_tokens=cached, reply=reply
        )
        return out

    def reuse(self) -> list:
        """An agent's conversation keeps its prompt state while other conversations with the same system prompt run:
        each of its next turns must report >= --reuse-min of its prompt as cached."""
        rows = []
        for rep in range(self.a.reuse_reps):
            seed = f"{self.a.seed}-reuse-{rep}-{self.salt}"
            system = {
                "role": "system",
                "content": f"[{self.salt} reuse {rep}] You are a coding assistant working in a "
                "repository. Read the notes below and keep answers short.\n"
                + self.document("prose", self.a.reuse_system, seed + "-sys"),
            }
            agent = [
                system,
                {
                    "role": "user",
                    "content": "Give a one-line summary of the log below.\n"
                    + self.document(
                        self.a.family, self.a.reuse_history, seed + "-hist"
                    ),
                },
            ]
            first = self.chat_once(agent)
            turns = [{"turn": 0, "kind": "agent (cold)", **first}]
            self.log(
                f"  reuse rep {rep} agent turn 0: {first.get('prompt_tokens')} prompt tokens, cached "
                f"{first.get('cached_tokens')}, {first.get('seconds')} s"
                + (f" ERROR {first['error']}" if first.get("error") else "")
            )
            reply = first.get("reply", "")
            for k in range(1, self.a.reuse_rounds + 1):
                other = self.chat_once(
                    [
                        system,
                        {
                            "role": "user",
                            "content": f"Side question {k}: list two ways to "
                            "reverse a string in Python.",
                        },
                    ]
                )
                turns.append({"turn": k, "kind": "other conversation", **other})
                agent += [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": f"Update {k}: "
                        + self.document("prose", 200, f"{seed}-note{k}"),
                    },
                ]
                t = self.chat_once(agent)
                reply = t.get("reply", "")
                pt, ct = t.get("prompt_tokens"), t.get("cached_tokens")
                t["cached_frac"] = (
                    rnd(ct / pt, 4) if isinstance(ct, int) and pt else None
                )
                turns.append({"turn": k, "kind": "agent", **t})
                frac = "" if t["cached_frac"] is None else f" ({t['cached_frac']:.0%})"
                self.log(
                    f"  reuse rep {rep} agent turn {k}: {pt} prompt tokens, cached {ct}{frac}, "
                    f"{t.get('seconds')} s"
                    + (f" ERROR {t['error']}" if t.get("error") else "")
                )
            agent_turns = [t for t in turns if t["kind"] == "agent"]
            fracs = [
                t["cached_frac"]
                for t in agent_turns
                if t.get("cached_frac") is not None
            ]
            row = {
                "suite": "reuse",
                "rep": rep,
                "turns": [{k: v for k, v in t.items() if k != "reply"} for t in turns],
                "cold_s": first.get("seconds"),
                "resumed_s": median([t.get("seconds") for t in agent_turns]),
            }
            errs = [t["error"] for t in turns if t.get("error")]
            if errs:
                row["error"] = errs[0]
            if not fracs:
                row["status"] = "unmeasured"
                row["why"] = (
                    "the server reported no cached-token count (usage.prompt_tokens_details.cached_tokens)"
                )
            else:
                row["worst_cached_frac"] = min(fracs)
                row["status"] = (
                    "ok"
                    if min(fracs) >= self.a.reuse_min and len(fracs) == len(agent_turns)
                    else "FAIL"
                )
            rows.append(row)
            self.log(
                f"  reuse rep {rep}: {row['status']}"
                + (
                    f" (worst {row['worst_cached_frac']:.0%} cached, need "
                    f"{self.a.reuse_min:.0%})"
                    if fracs
                    else f" ({row.get('why')})"
                )
            )
        return rows

    # -- anthropic
    def anthropic(self) -> list:
        """The Anthropic Messages surface: shape and usage, count_tokens, streamed tool_use, a tool_result turn,
        stop_sequences, and the 400 error shape."""
        checks: list[dict] = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append(
                {
                    "suite": "anthropic",
                    "check": name,
                    "ok": bool(ok),
                    "detail": detail[:400],
                }
            )
            self.log(f"  anthropic {name}: {'ok' if ok else 'FAIL'} ({detail[:160]})")

        think = {"type": "adaptive"} if self.a.thinking else {"type": "disabled"}
        base = {"model": self.c.model, "temperature": 0, "thinking": think}
        text_body = {
            **base,
            "max_tokens": 64,
            "system": "You are terse.",
            "messages": [
                {"role": "user", "content": "Reply with the words: the server is up"}
            ],
        }
        st, msg = self.c.anthropic("/v1/messages", text_body)
        usage = msg.get("usage") or {}
        texts = " ".join(
            b.get("text", "")
            for b in msg.get("content") or []
            if b.get("type") == "text"
        )
        check(
            "message",
            st == 200
            and msg.get("type") == "message"
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), list)
            and bool(texts.strip())
            and usage.get("input_tokens", 0) > 0
            and usage.get("output_tokens", 0) > 0
            and msg.get("stop_reason") in ("end_turn", "max_tokens", "stop_sequence"),
            f"HTTP {st}, stop_reason {msg.get('stop_reason')}, usage {usage}, text {texts.strip()[:40]!r}",
        )
        cst, count = self.c.anthropic(
            "/v1/messages/count_tokens",
            {
                k: v
                for k, v in text_body.items()
                if k not in ("max_tokens", "temperature")
            },
        )
        got, want = count.get("input_tokens"), usage.get("input_tokens")
        check(
            "count_tokens",
            cst == 200
            and isinstance(got, int)
            and isinstance(want, int)
            and abs(got - want) <= self.a.count_tolerance,
            f"HTTP {cst}, count_tokens {got} vs usage.input_tokens {want} (tolerance {self.a.count_tolerance})",
        )
        ask = "Use the tool to label ticket 7 with 'billing', 'refund' and 'priority'."
        events = self.c.anthropic_stream(
            "/v1/messages",
            {
                **base,
                "max_tokens": 512,
                "tools": ANTHROPIC_TOOLS,
                "stream": True,
                "messages": [{"role": "user", "content": ask}],
            },
        )
        call = assemble_tool_use(events)
        args = call.get("input") if isinstance(call.get("input"), dict) else {}
        check(
            "streamed tool_use",
            call.get("name") == "set_labels"
            and call.get("json_ok")
            and isinstance(args.get("labels"), list)
            and len(args["labels"]) >= 1
            and call.get("stop_reason") == "tool_use"
            and "message_start" in call["types"]
            and "message_stop" in call["types"],
            f"{len(events)} events, {call.get('name')}({json.dumps(args)[:120]}), json_ok {call.get('json_ok')}, "
            f"stop {call.get('stop_reason')}"
            + (f", error {call['error']}" if call.get("error") else ""),
        )
        loop = {
            **base,
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": ask},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "set_labels",
                            "input": {
                                "ticket_id": 7,
                                "labels": ["billing", "refund", "priority"],
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "labels saved: billing, refund, priority",
                        }
                    ],
                },
            ],
        }
        st2, msg2 = self.c.anthropic("/v1/messages", loop)
        texts2 = " ".join(
            b.get("text", "")
            for b in msg2.get("content") or []
            if b.get("type") == "text"
        )
        check(
            "tool_result turn",
            st2 == 200 and msg2.get("type") == "message" and bool(texts2.strip()),
            f"HTTP {st2}, stop {msg2.get('stop_reason')}, reply {texts2.strip()[:60]!r}",
        )
        stop = {
            **base,
            "max_tokens": 64,
            "stop_sequences": ["STOP"],
            "messages": [
                {
                    "role": "user",
                    "content": "Repeat this line exactly and add nothing: one two STOP "
                    "three",
                }
            ],
        }
        st3, msg3 = self.c.anthropic("/v1/messages", stop)
        texts3 = " ".join(
            b.get("text", "")
            for b in msg3.get("content") or []
            if b.get("type") == "text"
        )
        check(
            "stop_sequences",
            st3 == 200
            and msg3.get("stop_reason") == "stop_sequence"
            and msg3.get("stop_sequence") == "STOP"
            and "three" not in texts3,
            f"HTTP {st3}, stop {msg3.get('stop_reason')} {msg3.get('stop_sequence')!r}, text {texts3.strip()[:40]!r}",
        )
        st4, err = self.c.anthropic(
            "/v1/messages",
            {
                "model": self.c.model,
                "messages": [
                    {"role": "user", "content": "this request has no max_tokens"}
                ],
            },
        )
        check(
            "400 error shape",
            st4 == 400
            and err.get("type") == "error"
            and (err.get("error") or {}).get("type") == "invalid_request_error",
            f"HTTP {st4}, type {err.get('type')}, error.type {(err.get('error') or {}).get('type')}",
        )
        return checks


ANTHROPIC_TOOLS = [
    {
        "name": "set_labels",
        "description": "Set the labels of a support ticket.",
        "input_schema": {
            "type": "object",
            "required": ["ticket_id", "labels"],
            "properties": {
                "ticket_id": {"type": "integer"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
        },
    }
]


def assemble_tool_use(events: list[dict]) -> dict:
    """The first tool_use block of a streamed Anthropic reply: its name, its input_json_delta fragments joined and
    parsed (json_ok), the stop_reason and the event types seen."""
    out: dict = {
        "types": [e.get("type") for e in events],
        "name": None,
        "json_ok": False,
        "stop_reason": None,
    }
    parts: dict[int, list[str]] = {}
    names: dict[int, str] = {}
    for e in events:
        t = e.get("type")
        if t == "client_error":
            out["error"] = e.get("error")
        elif (
            t == "content_block_start"
            and (e.get("content_block") or {}).get("type") == "tool_use"
        ):
            names[e.get("index", 0)] = e["content_block"].get("name")
            parts[e.get("index", 0)] = []
        elif (
            t == "content_block_delta"
            and (e.get("delta") or {}).get("type") == "input_json_delta"
        ):
            parts.setdefault(e.get("index", 0), []).append(
                e["delta"].get("partial_json") or ""
            )
        elif t == "message_delta":
            out["stop_reason"] = (e.get("delta") or {}).get("stop_reason") or out[
                "stop_reason"
            ]
    if names:
        idx = min(names)
        out["name"], out["raw_input"] = names[idx], "".join(parts.get(idx, []))
        try:
            out["input"] = json.loads(out["raw_input"] or "{}")
            out["json_ok"] = isinstance(out["input"], dict)
        except ValueError:
            out["input"] = None
    return out


def exact_key(r: dict) -> tuple[str, str]:
    """What an exact case is compared by: its token ids' hash when the server returned them, else its text's."""
    if r.get("tokens_sha256"):
        return "tokens", r["tokens_sha256"]
    if r.get("server_token_sha"):
        return "server_token_sha", r["server_token_sha"]
    return "text", r.get("text_sha256", "")


def exact_check(rows: list[dict], ref_doc: dict) -> dict:
    """Each case against the reference run's: equal / different / missing, by token-id hash when both have one."""
    ref = {r["case"]: r for r in (ref_doc.get("runs") or {}).get("exact", [])}
    if not ref:
        raise SystemExit(
            "--exact-ref: the file has no runs.exact (record one with --suite exact --out FILE)"
        )
    res = {}
    for r in rows:
        x = ref.get(r["case"])
        if x is None:
            res[r["case"]] = "missing in the reference"
        elif r.get("error"):
            res[r["case"]] = "error: " + r["error"]
        elif x.get("prompt_sha256") != r.get("prompt_sha256"):
            res[r["case"]] = "different prompt (another prompt-set version)"
        else:
            if r.get("tokens_sha256") and x.get("tokens_sha256"):
                same = r["tokens_sha256"] == x["tokens_sha256"]
                by = "token ids"
            else:
                same = r.get("text_sha256") == x.get("text_sha256")
                by = "text"
            res[r["case"]] = ("equal" if same else "DIFFERENT") + f" ({by})"
    return res


# ---------------------------------------------------------------------------------------------------------- summary
def summarize(runs: dict) -> dict:
    s: dict = {}
    pre = [r for r in runs.get("prefill", []) if not r.get("error")]
    for n in sorted({r["target_tokens"] for r in pre}):
        rs = [r for r in pre if r["target_tokens"] == n]
        for k, src in (
            ("ttft_s", "ttft_s"),
            ("tok_s", "prefill_tok_s"),
            ("server_tok_s", "server_prefill_tok_s"),
            ("prompt_tokens", "prompt_tokens"),
            ("cached_tokens", "cached_tokens"),
        ):
            s[f"prefill.{n}.{k}"] = median([r.get(src) for r in rs])
    dec = [r for r in runs.get("decode", []) if not r.get("error")]
    for p in sorted({r["prompt"] for r in dec}):
        rs = [r for r in dec if r["prompt"] == p]
        for k, src in (
            ("tok_s", "decode_tok_s"),
            ("ttft_s", "ttft_s"),
            ("server_tok_s", "server_decode_tok_s"),
            ("tokens_per_round", "tokens_per_round"),
            ("accept_rate", "accept_rate"),
            ("completion_tokens", "completion_tokens"),
        ):
            s[f"decode.{p}.{k}"] = median([r.get(src) for r in rs])
    mul = [r for r in runs.get("multi", []) if not r.get("errors")]
    for p, n in sorted({(r["prompt"], r["streams"]) for r in mul}):
        rs = [r for r in mul if r["prompt"] == p and r["streams"] == n]
        for k, src in (
            ("wall_tok_s", "wall_tok_s"),
            ("live_tok_s", "live_tok_s"),
            ("per_stream_tok_s", "mean_per_stream_tok_s"),
            ("sum_per_stream_tok_s", "sum_per_stream_tok_s"),
            ("first_spread_s", "first_spread_s"),
            ("tokens_per_round", "tokens_per_round"),
        ):
            s[f"multi.{p}.{n}.{k}"] = median([r.get(src) for r in rs])
    ned = runs.get("needle", [])
    for n in sorted({r["target_tokens"] for r in ned}):
        rs = [r for r in ned if r["target_tokens"] == n]
        for d in sorted({r["depth"] for r in rs}):
            cell = [r for r in rs if r["depth"] == d]
            s[f"needle.{n}.{d:g}"] = (
                f"{sum(1 for r in cell if r.get('passed'))}/{len(cell)}"
            )
        s[f"needle.{n}.prompt_tokens"] = median([r.get("prompt_tokens") for r in rs])
        s[f"needle.{n}.ttft_s"] = median(
            [r.get("ttft_s") for r in rs if not r.get("error")]
        )
    itl = [r for r in runs.get("interleave", []) if not r.get("error")]
    for n in sorted({r["target_tokens"] for r in itl}):
        rs = [r for r in itl if r["target_tokens"] == n]
        for k in (
            "longest_gap_s",
            "p99_gap_s",
            "tokens_during",
            "long_ttft_s",
            "base_gap_s",
            "long_prompt_tokens",
        ):
            s[f"interleave.{n}.{k}"] = median([r.get(k) for r in rs])
        gone = [
            r["cancel"]["gone_s"] for r in rs if (r.get("cancel") or {}).get("measured")
        ]
        if any(r.get("cancel") for r in rs):
            s[f"interleave.{n}.cancel_gone_s"] = median(gone)
    ex = runs.get("exact", [])
    if ex:
        s["exact.intra_equal"] = (
            f"{sum(1 for r in ex if r.get('intra_equal'))}/{sum(1 for r in ex if 'same_as' in r)}"
        )
        for r in ex:
            s[f"exact.{r['case']}"] = (
                exact_key(r)[1][:16] if not r.get("error") else "error"
            )
    reu = runs.get("reuse", [])
    if reu:
        statuses = [r.get("status") for r in reu]
        s["reuse.status"] = (
            "FAIL"
            if "FAIL" in statuses
            else ("unmeasured" if "unmeasured" in statuses else "ok")
        )
        s["reuse.worst_cached_frac"] = min(
            [r["worst_cached_frac"] for r in reu if "worst_cached_frac" in r],
            default=None,
        )
        s["reuse.cold_s"] = median([r.get("cold_s") for r in reu])
        s["reuse.resumed_s"] = median([r.get("resumed_s") for r in reu])
    for r in runs.get("anthropic", []):
        s[f"anthropic.{r['check'].replace(' ', '_')}"] = "ok" if r["ok"] else "FAIL"
    errors = sum(
        1 for rows in runs.values() for r in rows if r.get("error") or r.get("errors")
    )
    s["errors"] = errors
    return s


def needle_grid(rows: list) -> list[str]:
    if not rows:
        return []
    depths = sorted({r["depth"] for r in rows})
    lines = [
        "needle grid (passed / runs):",
        f"  {'tokens':>9} " + " ".join(f"{d:>7g}" for d in depths),
    ]
    for n in sorted({r["target_tokens"] for r in rows}):
        cells = []
        for d in depths:
            c = [r for r in rows if r["target_tokens"] == n and r["depth"] == d]
            cells.append(
                f"{sum(1 for r in c if r.get('passed'))}/{len(c)}" if c else "-"
            )
        lines.append(f"  {n:>9} " + " ".join(f"{c:>7}" for c in cells))
    return lines


def health_delta(before, after) -> dict:
    """Numeric /health counters after - before (top level and graphs.<cache>.<field>)."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    out = {}
    for k, v in after.items():
        b = before.get(k)
        if (
            isinstance(v, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(v, bool)
            and k.endswith("_total")
        ):
            out[k] = rnd(v - b, 4)
    for name, g in (after.get("graphs") or {}).items():
        gb = (before.get("graphs") or {}).get(name) or {}
        for k in ("captured", "replayed", "eager", "floor_hits"):
            if isinstance(g.get(k), int) and isinstance(gb.get(k), int):
                out[f"graphs.{name}.{k}"] = g[k] - gb[k]
    if out.get("rounds_total"):
        out["tokens_per_round"] = rnd(
            out.get("completion_tokens_total", 0) / out["rounds_total"], 3
        )
    return out


# ---------------------------------------------------------------------------------------------------------- compare
COMPARED_SETTINGS = (
    "model",
    "thinking",
    "ignore_eos",
    "tokens",
    "family",
    "filler_version",
    "prompt_sha256",
    "needle_tokens",
    "exact_set",
    "temperature",
)


def load_run(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    if doc.get("tool") == TOOL or (doc.get("identity") or {}).get("tool") == TOOL:
        return {
            "path": path,
            "summary": doc.get("summary") or {},
            "settings": doc.get("settings") or {},
            "label": doc.get("label") or "",
            "kind": "speed.py",
        }
    if "summary" in doc and "runs" in doc and isinstance(doc.get("runs"), list):
        return adapt_mia(path, doc)
    raise SystemExit(f"{path}: neither a {TOOL} run nor a tools/bench.py run")


def adapt_mia(path: str, doc: dict) -> dict:
    """Mia's tools/bench.py JSON as our cells: its decode and multi prompt is our 'mia' prompt."""
    s, out = doc.get("summary") or {}, {}
    for k, v in s.items():
        m = re.fullmatch(r"multi\.(\d+)\.(aggregate_tok_s|per_stream_tok_s)", k)
        if m:
            out[
                f"multi.mia.{m[1]}.{'wall_tok_s' if m[2] == 'aggregate_tok_s' else 'per_stream_tok_s'}"
            ] = v
        elif k in ("decode.tok_s", "decode.ttft_s"):
            out["decode.mia." + k.split(".")[1]] = v
        elif k == "decode.accepted_over_drafted":
            out["decode.mia.accept_rate"] = v
        elif re.fullmatch(r"prefill\.\d+\.(ttft_s|tok_s)", k) or k == "errors":
            out[k] = v
    argv = doc.get("argv") or []
    tokens = 256
    if "--decode-tokens" in argv and argv.index("--decode-tokens") + 1 < len(argv):
        tokens = int(argv[argv.index("--decode-tokens") + 1])
    settings = {
        "model": doc.get("model"),
        "thinking": bool(doc.get("thinking")),
        "ignore_eos": False,
        "tokens": tokens,
        "family": "mia-bench.py prose",
        "filler_version": "mia-bench.py",
        "prompt_sha256": {"mia": sha256(MIA_DECODE)},
        "temperature": 0,
    }
    return {
        "path": path,
        "summary": out,
        "settings": settings,
        "label": doc.get("label") or "",
        "kind": "Mia tools/bench.py",
    }


def compare(paths: list[str], strict: bool = False, out=sys.stdout) -> int:
    runs = [load_run(p) for p in paths]
    names = [chr(ord("A") + i) for i in range(len(runs))]
    for n, r in zip(names, runs, strict=True):
        print(
            f"{n} = {r['path']} ({r['kind']}{', ' + r['label'] if r['label'] else ''})",
            file=out,
        )
    diffs = []
    for key in COMPARED_SETTINGS:
        vals = [r["settings"].get(key) for r in runs]
        if key == "prompt_sha256":
            common = (
                set.intersection(*[set((v or {}).keys()) for v in vals])
                if vals
                else set()
            )
            for p in sorted(common):
                pv = [(v or {}).get(p) for v in vals]
                if len(set(pv)) > 1:
                    diffs.append(f"prompt {p!r} text differs")
            continue
        if any(v != vals[0] for v in vals):
            diffs.append(
                f"{key}: "
                + ", ".join(f"{n}={v!r}" for n, v in zip(names, vals, strict=True))
            )
    if any(r["kind"] != "speed.py" for r in runs):
        diffs.append(
            "a file is Mia's tools/bench.py: decode / multi use the same prompt (cells *.mia.*), prefill uses "
            "a different filler, and her multi per-stream rate is the mean of the streams' own rates"
        )
    if diffs:
        print("\nSETTINGS DIFFER (cells may not be comparable):", file=out)
        for d in diffs:
            print("  - " + d, file=out)
    keys = sorted(set().union(*[r["summary"].keys() for r in runs]), key=cell_order)
    head = f"\n{'cell':40s} " + " ".join(f"{n:>12s}" for n in names[:1])
    head += "".join(f" {n:>12s} {'vs A':>8s}" for n in names[1:])
    print(head, file=out)
    flagged = 0
    for k in keys:
        vals = [r["summary"].get(k) for r in runs]
        if all(v is None for v in vals):
            continue
        line = f"{k:40s} {fmt(vals[0]):>12s}"
        for v in vals[1:]:
            line += f" {fmt(v):>12s} {delta(vals[0], v):>8s}"
        mark = prompt_token_mismatch(k, runs)
        if mark:
            line += "  ! " + mark
            flagged += 1
        print(line, file=out)
    if flagged:
        print(
            f"\n! {flagged} cell(s): the files' actual prompt sizes differ by more than 2%",
            file=out,
        )
    return 1 if strict and (diffs or flagged) else 0


def cell_order(k: str):
    parts = k.split(".")
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts]


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def delta(a, b) -> str:
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and a
    ):
        return f"{(b - a) / abs(a):+.1%}"
    if a is None or b is None:
        return ""
    return "" if a == b else "differs"


def prompt_token_mismatch(cell: str, runs: list[dict]) -> str:
    """For a prefill / needle / interleave cell: do the files' actual prompt sizes differ by more than 2%?"""
    m = re.match(r"(prefill|needle|interleave)\.(\d+)\.", cell)
    if not m:
        return ""
    key = f"{m[1]}.{m[2]}." + (
        "long_prompt_tokens" if m[1] == "interleave" else "prompt_tokens"
    )
    sizes = [r["summary"].get(key) for r in runs]
    sizes = [x for x in sizes if isinstance(x, (int, float))]
    if len(sizes) > 1 and max(sizes) > 1.02 * min(sizes):
        return f"prompt tokens {', '.join(fmt(x) for x in sizes)}"
    return ""


# ------------------------------------------------------------------------------------------------------------- main
SUITES = (
    "prefill",
    "decode",
    "multi",
    "needle",
    "interleave",
    "exact",
    "reuse",
    "anthropic",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("API_URL", DEFAULT_URL))
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", ""),
        help="Bearer key (default: $API_KEY)",
    )
    ap.add_argument(
        "--suite",
        default="prefill,decode,multi,needle",
        help=f"comma separated: {', '.join(SUITES)}",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="write the run's JSON here")
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="repetitions a prefill / decode / multi / interleave cell",
    )
    ap.add_argument(
        "--streams", default="1,2,4", help="multi: concurrent streams, comma separated"
    )
    ap.add_argument(
        "--tokens", type=int, default=256, help="decode / multi: max_tokens a stream"
    )
    ap.add_argument(
        "--prompts",
        default="mia,code,prose",
        help="decode / multi / interleave prompts: mia, code, prose",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="decode / multi: run every stream to --tokens",
    )
    ap.add_argument(
        "--lengths",
        default=None,
        help="prompt tokens for every length suite (prefill default "
        "1K,8K,32K,131K; K = 1024)",
    )
    ap.add_argument(
        "--needle-lengths",
        default=None,
        help="needle haystack tokens (default --lengths or 8K,32K,128K)",
    )
    ap.add_argument(
        "--interleave-lengths",
        default=None,
        help="interleave long-prompt tokens (default 131072)",
    )
    ap.add_argument(
        "--depths", default="0.1,0.5,0.9", help="needle depths as fractions"
    )
    ap.add_argument("--needle-reps", type=int, default=1)
    ap.add_argument(
        "--needle-tokens", type=int, default=32, help="needle: max_tokens of the answer"
    )
    ap.add_argument(
        "--family",
        choices=("prose", "code"),
        default="prose",
        help="filler of the long prompts",
    )
    ap.add_argument(
        "--interleave-tokens",
        type=int,
        default=16384,
        help="interleave: the decoding stream's max_tokens",
    )
    ap.add_argument(
        "--interleave-warm",
        type=int,
        default=16,
        help="interleave: tokens before the long prompt is sent",
    )
    ap.add_argument(
        "--cancel-at",
        type=float,
        default=None,
        metavar="S",
        help="interleave: close the decoding client S seconds into the long prompt",
    )
    ap.add_argument(
        "--cancel-wait",
        type=float,
        default=30.0,
        help="interleave: seconds to wait for the drop",
    )
    ap.add_argument(
        "--exact-ref",
        metavar="FILE",
        help="exact: compare with this reference run (exit 1 on a change)",
    )
    ap.add_argument(
        "--reuse-system",
        type=int,
        default=2048,
        help="reuse: the shared system prompt's tokens",
    )
    ap.add_argument(
        "--reuse-history",
        type=int,
        default=30000,
        help="reuse: the agent's history tokens",
    )
    ap.add_argument(
        "--reuse-rounds",
        type=int,
        default=3,
        help="reuse: other-conversation + agent-turn rounds",
    )
    ap.add_argument(
        "--reuse-min",
        type=float,
        default=0.9,
        help="reuse: the cached share an agent turn needs",
    )
    ap.add_argument("--reuse-reps", type=int, default=1)
    ap.add_argument(
        "--require-counter",
        action="store_true",
        help="reuse: exit 1 when the server reports no cached-token count (default: unmeasured, exit 0)",
    )
    ap.add_argument(
        "--count-tolerance",
        type=int,
        default=8,
        help="anthropic: count_tokens may differ from usage.input_tokens by this many tokens",
    )
    ap.add_argument("--thinking", action="store_true", help="thinking on (default off)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--salt",
        default=None,
        help="fixed salt (default random per run, so prompts are cold)",
    )
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument(
        "--health",
        action="store_true",
        help="store GET /health before and after, with counter deltas",
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="store GET /metrics before and after (raw)",
    )
    ap.add_argument(
        "--lean",
        action="store_true",
        help="drop each decode reply's raw server block from the JSON",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN.json",
        help="compare saved runs (first = base) and exit",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="--compare: exit 1 when settings or prompt sizes differ",
    )
    return ap


def settings_of(a: argparse.Namespace, run: Run, suites: list[str]) -> dict:
    prompts = {name: sha256("\n".join(PROMPTS[name])) for name in PROMPTS}
    return {
        "url": redact_text(a.url),
        "model": a.model,
        "suites": suites,
        "reps": a.reps,
        "streams": a.streams,
        "tokens": a.tokens,
        "prompts": a.prompts,
        "prompt_sha256": prompts,
        "ignore_eos": a.ignore_eos,
        "lengths": a.lengths,
        "needle_lengths": a.needle_lengths,
        "interleave_lengths": a.interleave_lengths,
        "depths": a.depths,
        "needle_reps": a.needle_reps,
        "needle_tokens": a.needle_tokens,
        "family": a.family,
        "filler_version": FILLER_VERSION,
        "exact_set": EXACT_SET,
        "interleave_tokens": a.interleave_tokens,
        "cancel_at": a.cancel_at,
        "reuse": [a.reuse_system, a.reuse_history, a.reuse_rounds, a.reuse_min],
        "thinking": a.thinking,
        "temperature": 0,
        "seed": a.seed,
        "salt": run.salt,
        "tokens_per_unit": run.tpu,
        "sizing": run.sizing,
    }


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.compare:
        return compare(a.compare, a.strict)
    suites = [s.strip() for s in a.suite.split(",") if s.strip()]
    bad = [s for s in suites if s not in SUITES]
    if bad:
        raise SystemExit(f"unknown suite {bad}; choose from {', '.join(SUITES)}")
    a.prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    if not a.prompts or any(p not in PROMPTS for p in a.prompts):
        raise SystemExit(f"--prompts: choose from {', '.join(PROMPTS)}")
    a.streams = parse_ints(a.streams)
    a.depths = parse_floats(a.depths)
    a.needle_lengths = parse_ints(a.needle_lengths or a.lengths or "8K,32K,128K")
    a.interleave_lengths = parse_ints(a.interleave_lengths or a.lengths or "131072")
    a.lengths = parse_ints(a.lengths or "1K,8K,32K,131K")
    ref = None
    if a.exact_ref:
        with open(a.exact_ref) as f:
            ref = json.load(f)
    client = Client(
        a.url,
        a.model,
        api_key=a.api_key,
        timeout=a.timeout,
        thinking=a.thinking,
        seed=a.seed,
        ignore_eos=a.ignore_eos,
    )
    run = Run(a, client)
    doc: dict = {
        "tool": TOOL,
        "schema": SCHEMA,
        "identity": tool_identity(),
        "label": a.label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "argv": redact_argv(sys.argv[1:] if argv is None else argv),
        "snapshots": {},
    }
    if a.health:
        doc["snapshots"]["health_before"] = client.get_json("/health")
    if a.metrics:
        doc["snapshots"]["metrics_before"] = metrics_text(client)
    runs: dict[str, list] = {}
    for s in suites:
        run.log(f"[{s}]")
        if s == "exact":
            runs[s] = run.exact()
        else:
            # the exact suite must be repeatable: only the timed suites use the run's salt
            runs[s] = getattr(run, s)()
    if a.health:
        doc["snapshots"]["health_after"] = client.get_json("/health")
        doc["health_delta"] = health_delta(
            doc["snapshots"]["health_before"], doc["snapshots"]["health_after"]
        )
    if a.metrics:
        doc["snapshots"]["metrics_after"] = metrics_text(client)
    doc["settings"] = settings_of(a, run, suites)
    doc["runs"], doc["summary"], doc["notes"] = runs, summarize(runs), run.notes
    rc = 1 if doc["summary"]["errors"] else 0
    if any(not r.get("passed") for r in runs.get("needle", [])):
        rc = 1
    for r in runs.get("reuse", []):
        if r.get("status") == "FAIL" or (
            r.get("status") == "unmeasured" and a.require_counter
        ):
            rc = 1
    if any(not r["ok"] for r in runs.get("anthropic", [])):
        rc = 1
    if "exact" in runs:
        if any(r.get("intra_equal") is False for r in runs["exact"]):
            rc = 1
        if ref is not None:
            doc["exact_check"] = exact_check(runs["exact"], ref)
            doc["exact_ref"] = a.exact_ref
            if any(not v.startswith("equal") for v in doc["exact_check"].values()):
                rc = 1
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report(doc, run)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        # argv and the URL are masked above; this catches the key echoed anywhere else (a server error, a snapshot)
        secrets = known_secrets() + ([a.api_key] if len(a.api_key) >= 4 else [])
        with open(a.out, "w") as f:
            f.write(
                mask_values(
                    json.dumps(doc, indent=1), [json.dumps(s)[1:-1] for s in secrets]
                )
            )
        run.log(f"wrote {a.out}")
    return rc


def metrics_text(client: Client) -> str:
    try:
        return client.get("/metrics")[1][:200_000]
    except (OSError, http.client.HTTPException) as exc:
        return f"error: {type(exc).__name__}: {exc}"


def report(doc: dict, run: Run) -> None:
    if run.a.quiet:
        return
    print("\nsummary (medians):")
    for k, v in doc["summary"].items():
        print(f"  {k:44s} {fmt(v)}")
    for line in needle_grid(doc["runs"].get("needle", [])):
        print(line)
    if doc.get("health_delta"):
        print(
            "health deltas: "
            + ", ".join(f"{k} {v}" for k, v in doc["health_delta"].items())
        )
    if doc.get("exact_check"):
        print(f"exact vs {doc['exact_ref']}:")
        for k, v in doc["exact_check"].items():
            print(f"  {k:16s} {v}")
    for n in dict.fromkeys(doc.get("notes") or []):
        print("note: " + n)


if __name__ == "__main__":
    sys.exit(main())
