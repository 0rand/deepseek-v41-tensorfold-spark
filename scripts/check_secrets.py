#!/usr/bin/env python3
"""Fail on credentials in files about to be published (stdlib only; scripts/check-public.sh runs it).

  check_secrets.py               every tracked and untracked (not ignored) file except the vendor/TensorFold submodule
  check_secrets.py PATH...       these files (directories recursively), git or not
  check_secrets.py --history     every line added by every commit of every ref, and every commit message

Finds API keys (sk-, sk-ant-), Hugging Face (hf_), GitHub (ghp_ gho_ ghu_ ghs_ ghr_ github_pat_), Slack (xox?-) and
AWS (AKIA) tokens, PEM private keys, long Bearer tokens and quoted credential assignments (api_key = "...") whose value
is not a placeholder. A finding prints its file, line and kind with the value masked to its first four characters;
the value itself is never printed. Exit 1 on a finding, 0 when clean.

Documented placeholders and test fixtures go in scripts/check-secrets-allow.txt: one Python regex a line, matched
against the whole finding, optionally followed by a TAB and a path glob that limits it to those files.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import PEM_BEGIN, TOKEN_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
ALLOW = Path(__file__).resolve().with_name("check-secrets-allow.txt")
EXCLUDED = re.compile(r"^vendor/TensorFold(/|$)")
NAMES = (
    r"api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|secret[_-]?key|client[_-]?secret"
    r"|(?:hf|github|gh|slack|openai|anthropic)[_-]?(?:token|key)|password|passwd|secret|token"
)
PATTERNS = (
    *TOKEN_PATTERNS,
    ("PEM private key", PEM_BEGIN),
    ("Bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "credential assignment",
        re.compile(
            r"""(?i)(?<![A-Za-z0-9_-])(?:"""
            + NAMES
            + r""")["']?\s*[:=]\s*(["'])([^"'\s]{6,})\1"""
        ),
    ),
)
# A quoted value that only stands in for a credential: <you>, $API_KEY, ${KEY}, {key}, %s, xxxx, ****, changeme, ...
PLACEHOLDER = re.compile(
    r"(?i)^(?:[<${%].*|x{4,}|\*{4,}|\.{3,}.*|changeme|change[-_]me|your[-_].*|.*example.*"
    r"|dummy.*|fake.*|placeholder.*|redacted|test[-_].*|none|null)$"
)


def allowlist(path: Path = ALLOW) -> list[tuple[re.Pattern, str]]:
    rules = []
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip() and not line.startswith("#"):
            regex, _, glob = line.partition("\t")
            rules.append((re.compile(regex), glob.strip()))
    return rules


def mask(value: str) -> str:
    return f"{value[:4]}…({len(value)} chars)"


def findings(text: str, where: str, rules: list[tuple[re.Pattern, str]]):
    """(where:line, kind, masked value) for every credential in ``text`` not allowed for ``where``."""

    if not any(pattern.search(text) for _, pattern in PATTERNS):
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS:
            for m in pattern.finditer(line):
                if kind == "credential assignment" and PLACEHOLDER.match(m[2]):
                    continue
                if any(
                    r.search(m[0]) and (not glob or fnmatch.fnmatch(where, glob))
                    for r, glob in rules
                ):
                    continue
                yield (
                    f"{where}:{lineno}",
                    kind,
                    mask(m[2] if kind == "credential assignment" else m[0]),
                )


def tree_files(paths: list[str]) -> list[tuple[Path, str]]:
    if paths:
        out = []
        for p in map(Path, paths):
            out += (
                [(f, str(f)) for f in sorted(p.rglob("*")) if f.is_file()]
                if p.is_dir()
                else [(p, str(p))]
            )
        return out
    names = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [
        (ROOT / n, n) for n in names if not EXCLUDED.match(n) and (ROOT / n).is_file()
    ]


def scan_tree(paths: list[str], rules) -> list[tuple[str, str, str]]:
    found = []
    for f, name in tree_files(paths):
        if f.resolve() == ALLOW:  # its lines are regexes, not values
            continue
        data = f.read_bytes()
        if b"\0" not in data[:8192]:
            found += findings(data.decode("utf-8", "replace"), name, rules)
    return found


def scan_history(rules) -> list[tuple[str, str, str]]:
    """Lines added by every commit reachable from any ref (vendor/TensorFold excluded) and every commit message."""

    log = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "-p",
            "--no-color",
            "--no-ext-diff",
            "--format=%x00commit %h%n%B",
            "--",
            ".",
            ":(exclude)vendor/TensorFold",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    found, commit, path, allow = [], "?", "?", ALLOW.relative_to(ROOT).as_posix()
    for raw in log.stdout.decode("utf-8", "replace").splitlines():
        if raw.startswith("\0commit "):
            commit, path = raw.split()[1], "(message)"
        elif raw.startswith("diff --git "):
            path = "?"
        elif raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else raw[4:]
        elif path == allow:  # its lines are regexes, not values
            continue
        elif path == "(message)" or (raw.startswith("+") and not raw.startswith("+++")):
            line = raw if path == "(message)" else raw[1:]
            found += [
                (f"{commit} {path}", kind, masked)
                for _, kind, masked in findings(line, path, rules)
            ]
    return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    rules = allowlist()
    found = scan_history(rules) if args == ["--history"] else scan_tree(args, rules)
    for where, kind, masked in found:
        print(f"{where}: {kind}: {masked}")
    if found:
        sys.stdout.flush()  # the findings before the summary when both streams go to one place
        print(
            f"check_secrets: {len(found)} credential finding(s); values masked above",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
