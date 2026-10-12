#!/usr/bin/env python3
"""Print each answer's wire shape (status, type, keys, SSE event shapes, metric names) for diffing two servers."""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from urllib.parse import urlparse


def shape(v):
    if isinstance(v, dict):
        return {k: shape(x) for k, x in v.items()}
    if isinstance(v, list):
        return [shape(v[0])] if v else []
    if isinstance(v, bool) or v is None:
        return v if v is None else "bool"
    return {int: "int", float: "num", str: "str"}.get(type(v), type(v).__name__)


def answer_shape(status: int, kind: str, raw: str):
    if kind.startswith("text/event-stream"):
        events, last = [], None
        for ev in raw.split("\n\n"):
            if not ev.startswith("data: "):
                continue
            data = ev[6:]
            s = "[DONE]" if data == "[DONE]" else json.dumps(shape(json.loads(data)), sort_keys=False)
            if s != last:
                events.append(s)
            last = s
        return {"status": status, "type": kind, "events": events}
    if kind.startswith("text/plain"):
        return {"status": status, "type": kind, "metrics": sorted({ln.split("{")[0] for ln in raw.splitlines()
                                                                   if ln and not ln.startswith("#")})}
    try:
        return {"status": status, "type": kind, "body": shape(json.loads(raw))}
    except json.JSONDecodeError:
        return {"status": status, "type": kind, "body": raw[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
    a = ap.parse_args()
    u = urlparse(a.base)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]
    user = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    short = {"model": a.model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 32,
             "temperature": 0, "reasoning_effort": "low"}
    probes = [
        ("GET", "/v1/models", None), ("GET", "/health", None), ("GET", "/metrics", None),
        ("POST", "/v1/chat/completions", short),
        ("POST", "/v1/chat/completions", dict(short, stream=True)),
        ("POST", "/v1/chat/completions", dict(short, stream=True, stream_options={"include_usage": True})),
        ("POST", "/v1/chat/completions", dict(short, messages=user, tools=tools, max_tokens=400)),
        ("POST", "/v1/chat/completions", dict(short, messages=user, tools=tools, max_tokens=400, stream=True)),
        ("POST", "/v1/chat/completions", dict(short, stop=["i"], reasoning_effort="none")),
        ("POST", "/v1/chat/completions", dict(short, return_token_ids=True, reasoning_effort="none")),
        ("POST", "/v1/completions", {"model": a.model, "prompt": "The capital of France is", "max_tokens": 8,
                                     "temperature": 0}),
        ("POST", "/v1/completions", {"model": a.model, "prompt": "1, 2, 3,", "max_tokens": 8, "temperature": 0,
                                     "stream": True}),
        ("POST", "/v1/chat/completions", dict(short, n=2)),
        ("POST", "/v1/chat/completions", dict(short, max_tokens=50_000_000)),
        ("POST", "/v1/chat/completions", dict(short, temperature="hot")),
        ("POST", "/v1/chat/completions", "{not json"),
        ("POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}]}),
        ("POST", "/tokenize", {"prompt": "hi"}),
        ("GET", "/v1/nothing", None),
        ("GET", "/health", None),
    ]
    for method, path, body in probes:
        c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
        data = None if body is None else (body if isinstance(body, str) else json.dumps(body)).encode()
        c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        r = c.getresponse()
        out = answer_shape(r.status, r.getheader("Content-Type") or "", r.read().decode(errors="replace"))
        c.close()
        print(json.dumps({"probe": f"{method} {path} {json.dumps(body)[:80] if body else ''}", **out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
