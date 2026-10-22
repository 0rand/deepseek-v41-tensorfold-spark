"""scripts/check_secrets.py fails on a planted credential, prints it masked, and passes documented placeholders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "check_secrets.py"

# Built at runtime so this file carries no credential-shaped literal.
FAKES = {
    "OpenAI / Anthropic key": "sk-ant-" + "api03-" + "Q" * 30,
    "Hugging Face token": "hf_" + "Ab3" * 12,
    "GitHub token": "ghp_" + "Zx9" * 12,
    "Slack token": "xoxb-" + "1234-5678-abcdefgh",
    "AWS access key id": "AKIA" + "ABCDEFGH23456789",
    "Bearer token": "Bearer " + "eyJ" + "x" * 30,
    "credential assignment": 'api_key = "' + "q7Rk" * 6 + '"',
    "PEM private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
}


def scan(*paths: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCAN), *map(str, paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_each_planted_credential_fails_masked(tmp_path):
    for kind, fake in FAKES.items():
        f = tmp_path / f"{kind.split()[0].lower()}.txt"
        f.write_text(f"config line\n{fake}\n")
        r = scan(f)
        assert r.returncode == 1, (kind, r.stdout)
        assert f"{f}:2: {kind}:" in r.stdout
        secret = fake.split('"')[1] if kind == "credential assignment" else fake
        # only the first four characters are shown
        assert secret not in r.stdout + r.stderr


def test_placeholders_pass(tmp_path):
    f = tmp_path / "README.md"
    f.write_text(
        'api_key = "<your key>"\npassword: "$PASSWORD"\ntoken = "${HF_TOKEN}"\nsecret = "changeme"\n'
        'API_KEY="xxxxxxxx"\nAuthorization: Bearer $API_KEY\nsk-cli\n--api-key sk-env\n'
    )
    r = scan(f)
    assert r.returncode == 0, r.stdout


def test_directory_scan_and_clean_exit(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ok.py").write_text("print('hello')\n")
    assert scan(tmp_path).returncode == 0
    (tmp_path / "a" / "leak.env").write_text(
        "HF_TOKEN=" + FAKES["Hugging Face token"] + "\n"
    )
    r = scan(tmp_path)
    assert r.returncode == 1 and "leak.env:1: Hugging Face token: hf_A…" in r.stdout
