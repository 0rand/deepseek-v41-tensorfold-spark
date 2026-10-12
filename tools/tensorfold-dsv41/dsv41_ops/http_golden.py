#!/usr/bin/env python3
"""Wire goldens: the reference HTTP server with a scripted engine answering fixed requests, one exchange per line."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tempfile
import threading
import types
from pathlib import Path

EOS = 1
WINDOW = 4096
VOCAB = 129280
D = "｜DSML｜"
SCRIPTS = [
    ("@think", "Let me think about the greeting.</think>Hello there!"),
    ("@tool", "I should look up the weather.</think>Checking.\n\n<" + D + "calls>\n<" + D + "invoke name=\"get_weather\">\n<"
     + D + "parameter name=\"city\" string=\"true\">Paris</" + D + "parameter>\n<" + D
     + "parameter name=\"days\" string=\"false\">3</" + D + "parameter>\n</" + D + "invoke>\n<" + D
     + "invoke name=\"get_weather\">\n<" + D + "parameter name=\"city\" string=\"true\">東京</" + D + "parameter>\n</"
     + D + "invoke>\n</" + D + "calls>"),
    ("@stop", "ok</think>abc END def"),
    ("@long", "Plenty to say here, more than the limit lets through."),
    ("@plain", "Plain answer."),
]
# the batch engine's ``job.stats`` keys (the Zig server's ``Outcome``); ttft_s is a clock reading (masked)
STATS = {"cached": 3, "ttft_s": 0.125, "finish": "done", "prefill_s": 0.25, "decode_s": 0.5, "rounds": 4, "drafted": 6,
         "accepted": 5}


def stub_torch() -> None:
    """vision.py imports torch at module level; the placeholder image mode never calls it."""

    class Any_:
        def __getattr__(self, name):
            return Any_()

        def __call__(self, *a, **k):
            return a[0] if len(a) == 1 and callable(a[0]) and not k else Any_()

    class Module(types.ModuleType):
        def __getattr__(self, name):
            return Any_()

    torch, nn, fn = Module("torch"), Module("torch.nn"), Module("torch.nn.functional")
    torch.nn, nn.functional = nn, fn
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": fn})


class Stub:
    """The engine attributes the app reads, and a scripted generate()."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self.eos = (EOS,)
        self.limit = WINDOW
        self.batch = None
        self.request = types.SimpleNamespace()
        self.seen = None

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        text = self.tok.decode(list(prompt), skip_special_tokens=False)
        last = text[text.rfind("<｜User｜>"):] if "<｜User｜>" in text else text
        reply = "?"
        for marker, script in SCRIPTS:
            if marker in last:
                reply = script
        r = self.request
        self.seen = {"prompt": list(prompt), "max_tokens": int(max_tokens),
                     "sampling": None if sampling is None else [str(sampling.seed), sampling.temperature,
                                                                sampling.top_k, sampling.top_p],
                     "draft": bool(draft), "stop_eos": bool(getattr(r, "stop_eos", True)),
                     "background": bool(getattr(r, "background", False))}
        ids = list(self.tok.encode(reply, add_special_tokens=False).ids) + [EOS]
        i = n = 0
        while i < len(ids) and n < max_tokens:
            step = min(1 + i % 3, len(ids) - i, max_tokens - n)
            stop = on_tokens(ids[i:i + step])
            i += step
            n += step
            if stop:
                break
        s = STATS
        return {"prompt": len(prompt), "cached": s["cached"], "ttft_s": s["ttft_s"], "finish": s["finish"],
                "completion": n, **{k: s[k] for k in ("prefill_s", "decode_s", "rounds", "drafted", "accepted")}}


def requests() -> list[tuple[str, str, str, str]]:
    """(name, method, path, raw body)."""

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}, "days": {"type": "integer"}},
                                                   "required": ["city"]}}}]

    def chat(marker, **kw):
        return json.dumps(dict({"model": "deepseek-v41", "messages": [{"role": "user", "content": f"hi {marker}"}],
                                "max_tokens": 200, "temperature": 0}, **kw), ensure_ascii=False)

    out = [
        ("models", "GET", "/v1/models", ""),
        ("chat-think", "POST", "/v1/chat/completions", chat("@think")),
        ("chat-think-stream", "POST", "/v1/chat/completions", chat("@think", stream=True)),
        ("chat-think-stream-usage", "POST", "/v1/chat/completions",
         chat("@think", stream=True, stream_options={"include_usage": True})),
        ("chat-alias-model", "POST", "/v1/chat/completions", chat("@plain", model="alias-a")),
        ("chat-effort-none", "POST", "/v1/chat/completions", chat("@plain", reasoning_effort="none")),
        ("chat-effort-low", "POST", "/v1/chat/completions", chat("@think", reasoning_effort="low")),
        ("chat-effort-37", "POST", "/v1/chat/completions", chat("@think", reasoning_effort=37)),
        ("chat-effort-bad", "POST", "/v1/chat/completions", chat("@think", reasoning_effort="huge")),
        ("chat-kwargs-off", "POST", "/v1/chat/completions",
         chat("@plain", chat_template_kwargs={"enable_thinking": False})),
        ("chat-kwargs-bad", "POST", "/v1/chat/completions", chat("@plain", chat_template_kwargs=[])),
        ("chat-tool", "POST", "/v1/chat/completions", chat("@tool", tools=tools)),
        ("chat-tool-stream", "POST", "/v1/chat/completions", chat("@tool", tools=tools, stream=True)),
        ("chat-tool-single", "POST", "/v1/chat/completions", chat("@tool", tools=tools, parallel_tool_calls=False)),
        ("chat-tool-none", "POST", "/v1/chat/completions", chat("@tool", tools=tools, tool_choice="none")),
        ("chat-stop", "POST", "/v1/chat/completions", chat("@stop", stop="END")),
        ("chat-stop-stream", "POST", "/v1/chat/completions", chat("@stop", stop=["END"], stream=True)),
        ("chat-length", "POST", "/v1/chat/completions", chat("@long", max_tokens=12)),
        ("chat-length-stream", "POST", "/v1/chat/completions", chat("@long", max_tokens=12, stream=True)),
        ("chat-token-ids", "POST", "/v1/chat/completions", chat("@plain", return_token_ids=True)),
        ("chat-ignore-eos", "POST", "/v1/chat/completions", chat("@plain", ignore_eos=True, max_tokens=40)),
        ("chat-sampled", "POST", "/v1/chat/completions", chat("@plain", temperature=0.7, top_p=0.9, seed=7)),
        ("chat-default-sampling", "POST", "/v1/chat/completions",
         json.dumps({"messages": [{"role": "user", "content": "hi @plain"}], "max_tokens": 50})),
        ("chat-no-draft", "POST", "/v1/chat/completions", chat("@plain", draft=False)),
        ("chat-background", "POST", "/v1/chat/completions", chat("@plain", priority="background")),
        ("chat-image", "POST", "/v1/chat/completions", json.dumps({"model": "deepseek-v41", "max_tokens": 50,
            "temperature": 0, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi @plain"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})),
        ("chat-max-completion", "POST", "/v1/chat/completions",
         json.dumps({"messages": [{"role": "user", "content": "hi @plain"}], "max_completion_tokens": 5,
                     "temperature": 0})),
        ("chat-n2", "POST", "/v1/chat/completions", chat("@plain", n=2)),
        ("chat-n1", "POST", "/v1/chat/completions", chat("@plain", n=1)),
        ("chat-temp-bad", "POST", "/v1/chat/completions", chat("@plain", temperature="hot")),
        ("chat-max-frac", "POST", "/v1/chat/completions", chat("@plain", max_tokens=1.5)),
        ("chat-stop-bad", "POST", "/v1/chat/completions", chat("@plain", stop=5)),
        ("chat-empty", "POST", "/v1/chat/completions", json.dumps({"messages": []})),
        ("chat-context", "POST", "/v1/chat/completions", chat("@plain", max_tokens=WINDOW)),
        ("chat-context-stream", "POST", "/v1/chat/completions", chat("@plain", max_tokens=WINDOW, stream=True)),
        ("bad-json", "POST", "/v1/chat/completions", "{not json"),
        ("not-object", "POST", "/v1/chat/completions", "[1, 2]"),
        ("completion", "POST", "/v1/completions",
         json.dumps({"prompt": "hi @plain", "max_tokens": 50, "temperature": 0})),
        ("completion-stream", "POST", "/v1/completions",
         json.dumps({"prompt": "hi @plain", "max_tokens": 50, "temperature": 0, "stream": True})),
        ("completion-ids", "POST", "/v1/completions",
         json.dumps({"prompt": [104, 105, 32, 64], "max_tokens": 5, "temperature": 0})),
        ("completion-ids-bad", "POST", "/v1/completions",
         json.dumps({"prompt": [104, 129280], "max_tokens": 5, "temperature": 0})),
        ("completion-context", "POST", "/v1/completions",
         json.dumps({"prompt": "hi @plain", "max_tokens": WINDOW, "temperature": 0})),
        ("tokenize-messages", "POST", "/tokenize", json.dumps({"messages": [{"role": "user", "content": "hi"}]})),
        ("tokenize-prompt", "POST", "/v1/tokenize", json.dumps({"prompt": "hi there"})),
        ("tokenize-bad", "POST", "/tokenize", json.dumps({"prompt": 5})),
        ("unknown-get", "GET", "/v1/nothing", ""),
        ("unknown-post", "POST", "/v1/nothing", "{}"),
        ("health", "GET", "/health", ""),
        ("metrics", "GET", "/metrics", ""),
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the Python engine's src/ (8474f31)")
    ap.add_argument("--tokenizer", required=True, help="the release tokenizer.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stub_torch()
    sys.path.insert(0, a.src)
    work = Path(tempfile.mkdtemp(prefix="dsv41-wire-"))
    model = work / "model"
    model.mkdir()
    (model / "tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
    (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "vocab_size": VOCAB}))
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    reqlog = work / "requests.jsonl"
    for k in [k for k in os.environ if k.startswith(("TF_DSV41_", "GLM53_TF_"))]:
        del os.environ[k]
    os.environ.update({"TF_DSV41_REQUEST_LOG": str(reqlog), "TF_DSV41_IMAGES": "placeholder"})

    from tokenizers import Tokenizer
    from tensorfold.families.deepseek_v41.cuda.app import Dsv41App
    from tensorfold.families.glm5_next.spark.server import make_handler
    from http.server import ThreadingHTTPServer

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    engine = Stub(tok)
    app = Dsv41App(engine, model, "deepseek-v41", aliases=("alias-a",), max_tokens=4096)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    lines = []
    for name, method, path, body in requests():
        engine.seen = None
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        data = body.encode() if body else None
        c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        r = c.getresponse()
        raw = r.read().decode()
        c.close()
        lines.append({"name": name, "method": method, "path": path, "body": body, "status": r.status,
                      "type": r.getheader("Content-Type"), "response": raw, "engine": engine.seen})
    srv.shutdown()
    app.reqlog.close()
    lines.append({"name": "_reqlog", "lines": [json.loads(x) for x in reqlog.read_text().splitlines()]})
    with open(a.out, "w") as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for p in sorted(work.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    work.rmdir()
    print(f"{len(lines) - 1} exchanges, {len(lines[-1]['lines'])} request-log lines -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
