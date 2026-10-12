"""DeepSeek-V4.1 reply goldens: replies parsed whole and streamed by the reference DSML parser, for the Zig parser."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

D = "｜DSML｜"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"enum": ["c", "f"]},
        "detail": {"type": ["boolean", "null"]}}}}},
    {"type": "function", "function": {"name": "search", "parameters": {"properties": {
        "query": {"type": "string"}, "top_k": {"type": "number"}, "tags": {"type": "array"},
        "filters": {"type": "object"}, "when": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}},
    {"type": "function", "function": {"name": "run_cmd", "parameters": {"properties": {
        "cmd": {"type": "string"}, "timeout": {"type": "integer"}, "env": {}}}}},
    {"name": "write_file", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                          "mode": {"type": "integer"}}}},
    {"type": "function", "function": {"name": "find_user", "parameters": {"properties": {"email": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_flights", "parameters": {"properties": {
        "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}}}},
]
VALUES = [("city", "true", "Paris"), ("city", "true", "東京"), ("days", "true", "3"), ("days", "false", "3"),
          ("days", "false", "3.0"), ("days", "true", "three"), ("unit", "true", "c"), ("detail", "false", "true"),
          ("detail", "true", "null"), ("detail", "true", "True"), ("query", "true", "zig <io> & \"quotes\""),
          ("top_k", "false", "2.5"), ("top_k", "true", "\"7\""), ("tags", "false", "[\"a\", \"b\""),
          ("tags", "false", "[\"a\", [1, 2]"), ("tags", "true", "[1,2]"), ("filters", "false", "{\"lang\": \"en\""),
          ("filters", "false", "{'a': 1}"), ("when", "true", "null"), ("cmd", "true", "ls -la\n  && echo </done>"),
          ("timeout", "false", "1e3"), ("env", "false", "{\"A\": 1}"), ("env", "true", "plain"),
          ("path", "true", "/tmp/x.py"), ("content", "true", "def f():\n    return 1\n\n\nprint(f())\n"),
          ("mode", "true", "0o644"), ("email", "true", "ana@example.com"), ("origin", "true", "Oslo"),
          ("unknown", "false", "not json"), ("days", "false", "  4  "), ("query", "false", "\"quoted\"")]
REASONS = ["Let me check the weather.", "The user wants two things.\n\nFirst the weather, then flights.", "",
           "思考：需要调用工具。", "Plan:\n1. search\n2. answer\n"]
CONTENTS = ["", "I'll look that up.", "Checking now…", "Sure!\n", "Here you go:\n\n", "Text with </think> inside."]


def invoke(rng: random.Random) -> str:
    t = rng.choice(TOOLS)
    name = (t.get("function") or t)["name"]
    name = rng.choice([name, name, name.upper(), "ns::" + name, "nonexistent_tool", "very_long_" * 12 + name])
    gap = rng.choice([" ", " ", "   ", "\n ", "\t\t"])
    sp = rng.choice([" ", " ", "", "\t"])
    params = []
    for _ in range(rng.randint(0, 4)):
        k, s, v = rng.choice(VALUES)
        ps = rng.choice([" ", " ", ""])
        g2 = rng.choice([" ", " ", "    ", "\n"])
        params.append(f"<{D}{ps}parameter{gap}name=\"{k}\"{g2}string=\"{s}\"{rng.choice(['', ' ', '  '])}>{v}</{D}{ps}parameter>")
    body = "\n".join(params)
    end = rng.choice([f"\n</{D}{sp}invoke>", f"\n</{D}{sp}invoke>", "", f"</{D}invoke>"])
    return f"<{D}{sp}invoke{gap}name=\"{name}\"{rng.choice(['', ' ', '   '])}>\n{body}{end}"


def reply(rng: random.Random, thinking: bool) -> str:
    parts = []
    if thinking:
        parts.append(rng.choice(REASONS))
        if rng.random() < 0.85:
            parts.append("</think>")
    parts.append(rng.choice(CONTENTS))
    if rng.random() < 0.85:
        name = rng.choice([" calls", " calls", "calls", " tool_calls", " function_calls"])
        sep = rng.choice(["\n\n", "\n\n", "\n", "", "\n\n\n"])
        invokes = "\n".join(invoke(rng) for _ in range(rng.randint(1, 3)))
        close = rng.choice([f"\n</{D}{name}>", f"\n</{D}{name}>", "", f"\n</{D}calls>"])
        parts.append(f"{sep}<{D}{name}>\n{invokes}{close}")
        if rng.random() < 0.1:
            parts.append("\ntrailing words")
    return "".join(parts)


def walk(v, out):
    if isinstance(v, str):
        if D in v:
            out.append(v)
    elif isinstance(v, dict):
        for x in v.values():
            walk(x, out)
    elif isinstance(v, list):
        for x in v:
            walk(x, out)


def norm(deltas: list) -> list:
    out = []
    for d in deltas:
        if "reasoning" in d:
            out.append(["r", d["reasoning"]])
        elif "content" in d:
            out.append(["c", d["content"]])
        else:
            tc = d["tool_calls"][0]
            if "id" in tc:
                out.append(["open", tc["index"], tc["function"]["name"]])
            else:
                out.append(["args", tc["index"], tc["function"]["arguments"]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--real", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    sys.path.insert(0, str(args.src))
    from tensorfold.families.deepseek_v41.cuda import dsml

    rng = random.Random(args.seed)
    real: list[str] = []
    for p in args.real:
        walk(json.loads(p.read_text()), real)
    texts = [(t, True) for t in real] + [(t, False) for t in real]
    while len(texts) < args.cases:
        thinking = rng.random() < 0.7
        t = reply(rng, thinking)
        texts.append((t, thinking))
        if rng.random() < 0.3:                      # cut by max_tokens
            texts.append((t[:rng.randint(0, len(t))], thinking))
    n = 0
    with open(args.out, "w") as f:
        for text, thinking in texts:
            tools = rng.choice([TOOLS, TOOLS, TOOLS[:2], None])
            cuts, i = [], 0
            while i < len(text):
                i = min(len(text), i + rng.choice([1, 1, 2, 3, 4, 7, 12]))
                cuts.append(i)
            s = dsml.Stream(thinking=thinking, tools=tools)
            deltas = [norm(s.feed(text[:c])) for c in cuts] + [norm(s.finish(text))]
            p = dsml.parse(text, thinking=thinking, tools=tools)
            f.write(json.dumps({
                "text": text, "thinking": thinking, "tools": tools,
                "cuts": [len(text[:c].encode()) for c in cuts], "deltas": deltas,
                "parse": {"reasoning": p.reasoning, "content": p.content,
                          "calls": [[c.name, c.arguments()] for c in p.calls]},
                "stream_calls": [[c.name, c.arguments()] for c in s.calls]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} replies ({len(real)} recorded ones, each with and without thinking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
