"""bench/speed.py against a small in-process fake server (http.server in a thread, no network beyond localhost).

The fake streams one token an event with a fixed delay, reports usage and a ``tensorfold`` block like the Zig server,
answers the needle question from the haystack, keeps a word-level prefix cache for the reuse check, serves the
Anthropic Messages routes, and can hold every stream while a long prompt "prefills" (the interleave suite).
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("speed", ROOT / "bench" / "speed.py")
speed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speed)


class Fake:
    """Server state and knobs."""

    def __init__(self):
        self.token_delay = 0.002
        self.reply_tokens = 24
        # prompts with more words "prefill" for long_pause s, holding other streams
        self.long_words = 10**9
        self.long_pause = 0.0
        self.long_tail = 0.0
        self.cache = True
        self.cached_counter = True
        self.perturb = ""
        self.free = threading.Event()
        self.free.set()
        self.lock = threading.Lock()
        self.inflight = 0
        self.seen: list[list[str]] = []
        self.auth: list[str | None] = []  # every request's Authorization header
        self.totals = {
            "requests_total": 0,
            "completion_tokens_total": 0,
            "rounds_total": 0,
        }


def words_of(messages) -> list[str]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(
                b.get("text", "")
                or b.get("content", "")
                or json.dumps(b.get("input", ""))
                for b in c
            )
        out += str(c or "").split()
    return out


def reply_text(fake: Fake, messages, n: int) -> str:
    last = str(messages[-1].get("content"))
    m = re.search(r"access code for the (\w+) vault in the text", last)
    if m:
        code = re.search(rf"access code for the {m[1]} vault is (\d+)", last)
        return code[1] if code else "unknown"
    h = hashlib.sha256((fake.perturb + json.dumps(messages)).encode()).hexdigest()
    return " ".join(f"w{h[i % 60 : i % 60 + 4]}" for i in range(n))


def make_handler(fake: Fake):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send_json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            fake.auth.append(self.headers.get("Authorization"))
            if self.path == "/health":
                with fake.lock:
                    # a careless server that echoes the caller's header: the saved snapshot must not carry it
                    body = {
                        "ok": True,
                        "inflight": fake.inflight,
                        **fake.totals,
                        "auth": fake.auth[-1],
                        "graphs": {
                            "decode": {
                                "held": 1,
                                "captured": 1,
                                "replayed": 5,
                                "eager": 0,
                                "floor_hits": 0,
                                "under_floor": False,
                            }
                        },
                    }
                return self.send_json(200, body)
            if self.path == "/metrics":
                data = b"tensorfold_requests_total 1\n"
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            return self.send_json(404, {"error": "not found"})

        def do_POST(self):
            fake.auth.append(self.headers.get("Authorization"))
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            )
            with fake.lock:
                fake.inflight += 1
                fake.totals["requests_total"] += 1
            try:
                if self.path == "/tokenize":
                    return self.send_json(
                        200, {"count": len(words_of(body["messages"])) + 2}
                    )
                if self.path == "/v1/chat/completions":
                    return self.chat(body)
                if self.path == "/v1/messages/count_tokens":
                    return self.send_json(
                        200, {"input_tokens": len(words_of(body["messages"])) + 5}
                    )
                if self.path == "/v1/messages":
                    return self.messages(body)
                return self.send_json(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with fake.lock:
                    fake.inflight -= 1

        def chat(self, body):
            msgs = body["messages"]
            words = words_of(msgs)
            cached = 0
            if fake.cache:
                with fake.lock:
                    for prev in fake.seen:
                        k = 0
                        for a, b in zip(prev, words, strict=False):
                            if a != b:
                                break
                            k += 1
                        cached = max(cached, k)
                    fake.seen.append(words)
            n = (
                body["max_tokens"]
                if body.get("ignore_eos")
                else min(body["max_tokens"], fake.reply_tokens)
            )
            text = reply_text(fake, msgs, n)
            toks = text.split(" ")
            n = len(toks)
            usage = {"prompt_tokens": len(words), "completion_tokens": n}
            if fake.cached_counter:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            tf = {
                "prompt": len(words),
                "cached": cached if fake.cached_counter else None,
                "completion": n,
                "prefill_s": 0.001,
                "decode_s": 0.001 * n,
                "rounds": (n + 1) // 2,
                "drafted": n,
                "accepted": n // 2,
            }
            if not fake.cached_counter:
                tf.pop("cached")
            if body.get("return_token_ids"):
                tf["token_ids"] = [
                    int(hashlib.sha256(t.encode()).hexdigest()[:6], 16) for t in toks
                ]
            with fake.lock:
                fake.totals["completion_tokens_total"] += n
                fake.totals["rounds_total"] += tf["rounds"]
            long = len(words) > fake.long_words
            if long:
                fake.free.clear()
                time.sleep(fake.long_pause)
                fake.free.set()
                time.sleep(fake.long_tail)
            if not body.get("stream"):
                return self.send_json(
                    200,
                    {
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": usage,
                        "tensorfold": tf,
                    },
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i, t in enumerate(toks):
                fake.free.wait()
                piece = t if i == 0 else " " + t
                self.event(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                time.sleep(fake.token_delay)
            self.event(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
            )
            self.event(
                {
                    "choices": [],
                    "usage": usage,
                    "tensorfold": {k: v for k, v in tf.items() if k != "token_ids"},
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def event(self, obj) -> None:
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        def messages(self, body):
            if "max_tokens" not in body:
                return self.send_json(
                    400,
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "max_tokens is required",
                        },
                    },
                )
            usage = {
                "input_tokens": len(words_of(body["messages"])) + 3,
                "output_tokens": 4,
            }
            last = body["messages"][-1]["content"]
            if body.get("stream"):
                args = json.dumps(
                    {"ticket_id": 7, "labels": ["billing", "refund", "priority"]}
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.event(
                    {
                        "type": "message_start",
                        "message": {
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "usage": usage,
                        },
                    }
                )
                self.event(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_x",
                            "name": "set_labels",
                            "input": {},
                        },
                    }
                )
                for i in range(0, len(args), 7):
                    self.event(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args[i : i + 7],
                            },
                        }
                    )
                self.event({"type": "content_block_stop", "index": 0})
                self.event(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                        "usage": {"output_tokens": 20},
                    }
                )
                self.event({"type": "message_stop"})
                self.close_connection = True
                return None
            msg = {
                "type": "message",
                "role": "assistant",
                "usage": usage,
                "stop_sequence": None,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "the server is up"}],
            }
            if body.get("stop_sequences") and isinstance(last, str) and "STOP" in last:
                msg.update(
                    stop_reason="stop_sequence",
                    stop_sequence="STOP",
                    content=[{"type": "text", "text": "one two "}],
                )
            elif isinstance(last, list):
                msg["content"] = [{"type": "text", "text": "Labels are set."}]
            return self.send_json(200, msg)

    return H


@pytest.fixture
def server():
    fake = Fake()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    fake.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield fake
    httpd.shutdown()
    httpd.server_close()


def run(server, tmp_path, *args, name="run.json"):
    out = tmp_path / name
    rc = speed.main(["--url", server.url, "--quiet", "--out", str(out), *args])
    return rc, json.loads(out.read_text())


# ------------------------------------------------------------------------------------------------------- pure math
def test_totals_wall_live_spread():
    a = {"times": [float(t) for t in range(11)], "tokens": 11}
    b = {"times": [float(t) for t in range(5, 16)], "tokens": 11}
    t = speed.totals([a, b])
    assert t["wall_tok_s"] == pytest.approx(20 / 15, abs=1e-2)  # (22 - 2) / (15 - 0)
    assert t["live_tok_s"] == pytest.approx(2.0)  # 5 + 5 tokens in (5, 10]
    assert t["live_span_s"] == pytest.approx(5.0)
    assert t["first_spread_s"] == pytest.approx(5.0)
    # the overstated legacy figure
    assert t["sum_per_stream_tok_s"] == pytest.approx(2.0)
    assert t["mean_per_stream_tok_s"] == pytest.approx(1.0)


def test_totals_weights_multi_token_events():
    # one event a round carrying 3 tokens: 4 events, 10 tokens -> 3 tokens an event after the first
    a = {"times": [0.0, 1.0, 2.0, 3.0], "tokens": 10}
    b = {"times": [0.0, 1.0, 2.0, 3.0], "tokens": 10}
    t = speed.totals([a, b])
    assert t["wall_tok_s"] == pytest.approx(6.0)
    assert t["live_tok_s"] == pytest.approx(6.0)


def test_gaps_and_percentile():
    times = [0.0, 0.1, 0.2, 2.2, 2.3]
    g = speed.gaps_during(times, 0.15, 2.25)
    assert max(g) == pytest.approx(2.0)
    assert speed.percentile([1, 2, 3, 4], 99) == 4
    assert speed.percentile([], 50) is None
    # a stream silent since 0.2 up to the window's end: the open gap counts
    assert max(speed.gaps_during([0.0, 0.1, 0.2], 0.0, 5.0)) == pytest.approx(4.8)


def test_parse_ints_and_needle_score():
    assert speed.parse_ints("1K,8k,32768,1M") == [1024, 8192, 32768, 1048576]
    assert speed.needle_score("The code is 1234567.", "1234567") == {
        "found": True,
        "exact": False,
        "numbers": ["1234567"],
    }
    assert speed.needle_score("1234567", "1234567")["exact"]
    assert not speed.needle_score("12345678", "1234567")["found"]


def test_filler_is_deterministic_and_salted():
    assert speed.filler("prose", 50, "a") == speed.filler("prose", 50, "a")
    assert speed.filler("prose", 50, "a") != speed.filler("prose", 50, "b")
    assert len(speed.filler("code", 7, "a").splitlines()) == 7
    _, code, fact = speed.needle_fact("x")
    assert code in fact
    text = speed.insert_at(speed.filler("prose", 200, "s"), "prose", 0.5, fact)
    assert code in text


# ---------------------------------------------------------------------------------------------- through the server
def test_suites_json_shape(server, tmp_path):
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "prefill,decode,multi,needle",
        "--lengths",
        "200,400",
        "--needle-lengths",
        "300",
        "--depths",
        "0.1,0.9",
        "--reps",
        "2",
        "--streams",
        "1,2",
        "--prompts",
        "mia,code",
        "--tokens",
        "16",
        "--health",
        "--metrics",
        "--label",
        "t",
    )
    assert rc == 0, doc["summary"]
    assert doc["tool"] == "bench/speed.py" and doc["schema"] == speed.SCHEMA
    assert len(doc["identity"]["sha256"]) == 64
    st = doc["settings"]
    assert (
        st["lengths"] == [200, 400]
        and st["streams"] == [1, 2]
        and st["sizing"] == {"prose": "tokenize"}
    )
    assert set(doc["runs"]) == {"prefill", "decode", "multi", "needle"}
    assert (
        len(doc["runs"]["prefill"]) == 4
        and len(doc["runs"]["decode"]) == 4
        and len(doc["runs"]["multi"]) == 8
    )
    s = doc["summary"]
    for k in (
        "prefill.200.ttft_s",
        "prefill.400.tok_s",
        "decode.mia.tok_s",
        "decode.code.tokens_per_round",
        "multi.mia.2.wall_tok_s",
        "multi.mia.2.live_tok_s",
        "multi.code.2.first_spread_s",
        "multi.mia.2.per_stream_tok_s",
        "needle.300.prompt_tokens",
    ):
        assert s.get(k) is not None, k
    assert s["needle.300.0.1"] == "1/1" and s["needle.300.0.9"] == "1/1"
    # prompts sized with /tokenize land near the target
    assert abs(s["prefill.400.prompt_tokens"] - 400) < 40
    # the salt keeps every prefill cold
    assert all(not r.get("cache_hit") for r in doc["runs"]["prefill"])
    assert doc["runs"]["decode"][0]["tokens_per_round"] == pytest.approx(16 / 8)
    assert (
        doc["health_delta"]["requests_total"] > 0
        and "graphs.decode.eager" in doc["health_delta"]
    )
    assert doc["snapshots"]["metrics_before"].startswith("tensorfold_requests_total")
    multi = [r for r in doc["runs"]["multi"] if r["streams"] == 2]
    for r in multi:
        assert r["wall_tok_s"] <= r["sum_per_stream_tok_s"] + 1e-6


def test_needle_failure_exits_1(server, tmp_path, monkeypatch):
    monkeypatch.setattr(
        speed,
        "needle_score",
        lambda reply, code: {"found": False, "exact": False, "numbers": []},
    )
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "needle",
        "--needle-lengths",
        "200",
        "--depths",
        "0.5",
    )
    assert rc == 1 and doc["summary"]["needle.200.0.5"] == "0/1"


def test_interleave_gap_and_cancel(server, tmp_path):
    server.token_delay = 0.01
    server.long_words = 1000
    server.long_pause = 0.6
    server.long_tail = 0.4
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "interleave",
        "--interleave-lengths",
        "2000",
        "--reps",
        "1",
        "--interleave-tokens",
        "400",
        "--interleave-warm",
        "5",
        "--prompts",
        "prose",
    )
    assert rc == 0, doc
    row = doc["runs"]["interleave"][0]
    assert row["longest_gap_s"] >= 0.5  # the held stream's pause shows
    assert row["base_gap_s"] < 0.1
    assert row["long_ttft_s"] >= 1.0
    assert doc["summary"]["interleave.2000.longest_gap_s"] == row["longest_gap_s"]
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "interleave",
        "--interleave-lengths",
        "2000",
        "--reps",
        "1",
        "--interleave-tokens",
        "400",
        "--interleave-warm",
        "5",
        "--cancel-at",
        "0.2",
        name="c.json",
    )
    cancel = doc["runs"]["interleave"][0]["cancel"]
    assert cancel["measured"], cancel
    assert 0.2 < cancel["gone_s"] < 1.0  # dropped at the server's next write


def test_interleave_cancel_unmeasurable(server, tmp_path):
    _, doc = run(
        server,
        tmp_path,
        "--suite",
        "interleave",
        "--interleave-lengths",
        "100",
        "--reps",
        "1",
        "--interleave-tokens",
        "200",
        "--cancel-at",
        "5",
    )
    cancel = doc["runs"]["interleave"][0]["cancel"]
    assert cancel["measured"] is False and "finished before" in cancel["why"]


def test_exact_record_and_check(server, tmp_path):
    rc, ref = run(server, tmp_path, "--suite", "exact", name="ref.json")
    assert rc == 0
    rows = {r["case"]: r for r in ref["runs"]["exact"]}
    assert rows["conc_1"]["intra_equal"] and rows["repeat_code"]["intra_equal"]
    assert len(rows["code"]["tokens_sha256"]) == 64
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "exact",
        "--exact-ref",
        str(tmp_path / "ref.json"),
        name="b.json",
    )
    assert rc == 0 and all(
        v.startswith("equal (token ids)") for v in doc["exact_check"].values()
    )
    server.perturb = "changed"
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "exact",
        "--exact-ref",
        str(tmp_path / "ref.json"),
        name="c.json",
    )
    assert rc == 1 and doc["exact_check"]["code"].startswith("DIFFERENT")


def test_reuse_ok_fail_unmeasured(server, tmp_path):
    args = ["--suite", "reuse", "--reuse-system", "100", "--reuse-history", "4000"]
    rc, doc = run(server, tmp_path, *args)
    assert rc == 0 and doc["summary"]["reuse.status"] == "ok"
    assert doc["summary"]["reuse.worst_cached_frac"] >= 0.9
    server.cache = False
    rc, doc = run(server, tmp_path, *args, name="f.json")
    assert rc == 1 and doc["summary"]["reuse.status"] == "FAIL"
    server.cached_counter = False
    rc, doc = run(server, tmp_path, *args, name="u.json")
    assert rc == 0 and doc["summary"]["reuse.status"] == "unmeasured"
    rc, doc = run(server, tmp_path, *args, "--require-counter", name="r.json")
    assert rc == 1


def test_anthropic_checks(server, tmp_path):
    rc, doc = run(server, tmp_path, "--suite", "anthropic")
    assert rc == 0, doc["runs"]["anthropic"]
    assert {r["check"] for r in doc["runs"]["anthropic"]} == {
        "message",
        "count_tokens",
        "streamed tool_use",
        "tool_result turn",
        "stop_sequences",
        "400 error shape",
    }
    rc, doc = run(
        server,
        tmp_path,
        "--suite",
        "anthropic",
        "--count-tolerance",
        "1",
        name="t.json",
    )
    assert rc == 1 and doc["summary"]["anthropic.count_tokens"] == "FAIL"


def test_assemble_tool_use_bad_json():
    ev = [
        {"type": "message_start"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "name": "f"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"a": ['},
        },
        {"type": "message_stop"},
    ]
    out = speed.assemble_tool_use(ev)
    assert out["name"] == "f" and out["json_ok"] is False


def test_request_error_exits_1(tmp_path):
    rc = speed.main(
        [
            "--url",
            "http://127.0.0.1:9",
            "--quiet",
            "--suite",
            "decode",
            "--reps",
            "1",
            "--prompts",
            "code",
            "--out",
            str(tmp_path / "e.json"),
        ]
    )
    assert rc == 1


# ------------------------------------------------------------------------------------------------------- compare
def _doc(summary: dict, **settings) -> dict:
    base = {
        "model": "m",
        "thinking": False,
        "ignore_eos": False,
        "tokens": 256,
        "family": "prose",
        "filler_version": 1,
        "prompt_sha256": {"mia": "x"},
        "temperature": 0,
    }
    base.update(settings)
    return {"tool": "bench/speed.py", "summary": summary, "settings": base}


def test_compare_deltas_and_mismatch(tmp_path):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(
        json.dumps(
            _doc(
                {
                    "decode.mia.tok_s": 50.0,
                    "prefill.8192.tok_s": 2000.0,
                    "prefill.8192.prompt_tokens": 8192,
                    "needle.8192.0.5": "1/1",
                }
            )
        )
    )
    b.write_text(
        json.dumps(
            _doc(
                {
                    "decode.mia.tok_s": 55.0,
                    "prefill.8192.tok_s": 1900.0,
                    "prefill.8192.prompt_tokens": 8190,
                    "needle.8192.0.5": "0/1",
                }
            )
        )
    )
    c.write_text(
        json.dumps(
            _doc(
                {
                    "decode.mia.tok_s": 60.0,
                    "prefill.8192.tok_s": 2100.0,
                    "prefill.8192.prompt_tokens": 9000,
                },
                tokens=512,
            )
        )
    )
    out = io.StringIO()
    assert speed.compare([str(a), str(b)], out=out) == 0
    text = out.getvalue()
    assert (
        "+10.0%" in text
        and "-5.0%" in text
        and "differs" in text
        and "SETTINGS DIFFER" not in text
    )
    out = io.StringIO()
    assert speed.compare([str(a), str(b), str(c)], strict=True, out=out) == 1
    text = out.getvalue()
    assert "SETTINGS DIFFER" in text and "tokens: A=256, B=256, C=512" in text
    assert "! prompt tokens" in text  # 9000 vs 8192 is flagged


def test_compare_reads_mia_bench(tmp_path):
    mia = {
        "schema": 1,
        "label": "hers",
        "model": "DeepSeek-V4.1-Flash-1017",
        "thinking": False,
        "argv": [],
        "runs": [],
        "summary": {
            "decode.tok_s": 58.0,
            "multi.4.aggregate_tok_s": 115.0,
            "multi.4.per_stream_tok_s": 30.0,
            "prefill.8192.tok_s": 2300.0,
            "errors": 0,
        },
    }
    ours = _doc(
        {
            "decode.mia.tok_s": 45.0,
            "multi.mia.4.wall_tok_s": 100.0,
            "prefill.8192.tok_s": 2200.0,
        },
        model="DeepSeek-V4.1-Flash-TF",
        prompt_sha256={"mia": speed.sha256(speed.MIA_DECODE)},
    )
    (tmp_path / "m.json").write_text(json.dumps(mia))
    (tmp_path / "o.json").write_text(json.dumps(ours))
    out = io.StringIO()
    speed.compare([str(tmp_path / "o.json"), str(tmp_path / "m.json")], out=out)
    text = out.getvalue()
    assert "Mia tools/bench.py" in text
    assert re.search(r"multi\.mia\.4\.wall_tok_s\s+100\.0\s+115\.0\s+\+15\.0%", text)
    assert re.search(r"decode\.mia\.tok_s\s+45\.0\s+58\.0", text)
    assert "prompt 'mia' text differs" not in text  # the same printing-press prompt


def test_api_key_never_saved(server, tmp_path, monkeypatch, capsys):
    """--api-key KEY, --api-key=KEY and $API_KEY reach the server as a header but never the result file or stdout."""

    # built at runtime: no credential literal in the tree
    key = "DUMMY" + "-speed-" + "8e41c07b2d95"
    base = [
        "--suite",
        "decode",
        "--reps",
        "1",
        "--prompts",
        "code",
        "--tokens",
        "4",
        "--health",
    ]
    runs = {"separate": ["--api-key", key], "equals": ["--api-key=" + key], "env": []}
    for name, flag in runs.items():
        if name == "env":
            monkeypatch.setenv("API_KEY", key)
        out = tmp_path / f"{name}.json"
        rc = speed.main(["--url", server.url, "--out", str(out), *base, *flag])
        text = out.read_text()
        assert rc == 0 and key not in text, name
        doc = json.loads(text)
        assert doc["snapshots"]["health_before"]["auth"] == "Bearer <redacted>"
        masked = {
            "separate": ["--api-key", "<redacted>"],
            "equals": ["--api-key=<redacted>"],
            "env": [],
        }[name]
        assert doc["argv"][-len(base) - len(masked) :] == base + masked
    assert server.auth and all(h == "Bearer " + key for h in server.auth)
    printed = capsys.readouterr()
    assert key not in printed.out + printed.err
