"""bench/tooleval/run.sh end to end with a dummy key: stand-in tool-eval-bench and spark-bench that echo their argv,
environment and request headers into every report, log and dump, and a localhost fake server that checks the key.
The key goes in through API_KEY (the separate --api-key form) and TEB_ARGS / SB_ARGS (--api-key=KEY); it must not
appear in any saved file or on the terminal."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# built at runtime: no credential literal in the tree
KEY = "DUMMY" + "-tooleval-" + "5f3c9a0e1b7d"
MODEL = "DeepSeek-V4.1-Flash-TF"

TEB = """#!/bin/sh
json=""; prev=""
for a in "$@"; do [ "$prev" = --json-file ] && json=$a; prev=$a; done
echo "tool-eval-bench argv: $*"
echo "tool-eval-bench using key $API_KEY" >&2
printf '{"argv": "%s", "headers": {"Authorization": "Bearer %s"}}\\n' "$*" "$API_KEY" > "$json"
"""
SB = """#!/bin/sh
echo "spark-bench argv: $*"
echo "spark-bench key=$SPARK_BENCH_API_KEY"
printf '{"request_headers": {"x-api-key": "%s"}, "argv": "%s"}\\n' "$SPARK_BENCH_API_KEY" "$*" \\
    > "$SPARK_BENCH_DUMP_DIR/raw_dump.jsonl"
"""


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            ok = self.headers.get("Authorization") == "Bearer " + KEY
            body = (
                {"status": "ok"}
                if self.path == "/health"
                else {"data": [{"id": MODEL}]}
            )
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(body if ok else {"error": "invalid_api_key"}).encode()
            )

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


def install_fakes(cache: Path) -> None:
    for path, text in (
        (cache / "tool-eval-bench/.venv/bin/tool-eval-bench", TEB),
        (cache / "spark-bench/.venv/bin/python", SB),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(text)
        path.chmod(0o755)


@pytest.mark.skipif(shutil.which("curl") is None, reason="run.sh needs curl")
def test_key_never_saved_or_printed(tmp_path, server):
    url, seen = server
    install_fakes(tmp_path / "cache")
    env = dict(
        os.environ,
        BASE_URL=url + "/v1",
        MODEL=MODEL,
        API_KEY=KEY,
        OUT=str(tmp_path / "out"),
        TOOLEVAL_CACHE=str(tmp_path / "cache"),
        TEB_ARGS="--api-key=" + KEY,
        SB_ARGS="--api-key=" + KEY,
    )
    r = subprocess.run(
        ["bash", str(ROOT / "bench/tooleval/run.sh"), "both", "off"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    # the key still reached the server, as a header
    assert seen and all(h == "Bearer " + KEY for h in seen)

    files = [p for p in (tmp_path / "out").rglob("*") if p.is_file()]
    names = {p.name for p in files}
    assert {
        "teb.cmd",
        "teb.log",
        "teb.json",
        "sb.cmd",
        "sb.log",
        "raw_dump.jsonl",
        "models-before.json",
    } <= names
    for p in files:
        assert KEY not in p.read_text(), p
    assert KEY not in r.stdout + r.stderr

    run = next(p.parent for p in files if p.name == "teb.cmd")
    teb_cmd = (run / "teb.cmd").read_text()
    assert (
        "--api-key '<redacted>'" in teb_cmd
        and "'--api-key=<redacted>'" in teb_cmd
        and "--no-think" in teb_cmd
    )
    assert "'--api-key=<redacted>'" in (run / "sb.cmd").read_text()
