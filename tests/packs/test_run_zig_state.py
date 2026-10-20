"""scripts/run-zig.sh: who may use a STATE_DIR (STATE_DIR/pack-id), run for real against a stub docker.

The launcher runs unchanged with CONFIG pointing at a throwaway profile; `docker` on PATH is a stub that records its
arguments, so nothing starts and no GPU, image or network is needed.
"""

from __future__ import annotations

import os
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import ROOT

LAUNCHER = ROOT / "scripts" / "run-zig.sh"
DOCKER = '#!/bin/sh\nprintf "%s\\n" "$@" > "$DOCKER_LOG.$$"\n'


@pytest.fixture
def node(tmp_path):
    """Folders a rank-1 start checks, a stub docker, and a config writer; returns run(pack, *args) -> process."""
    for name in ("model", "engram", "cache", "prepared", "bin"):
        (tmp_path / name).mkdir()
    assets = tmp_path / "assets"
    (assets / "aot").mkdir(parents=True)
    (assets / "aot" / "aot.json").write_text("{}")
    (assets / "sigs.json").write_text("{}")
    (assets / "engram-host.bin").write_bytes(b"")
    (assets / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<I", 4096 + 2048))
    docker = tmp_path / "bin" / "docker"
    docker.write_text(DOCKER)
    docker.chmod(0o755)
    state = tmp_path / "state"

    def run(pack: str | None, *args: str, select: str | None = None):
        config = tmp_path / f"prod-zig-{pack or 'default'}.env"
        lines = [] if pack is None else [f"PACK={pack}"]
        lines += [f'{key}="{tmp_path / name}"' for key, name in KEYS.items()]
        lines += ['IMAGE="stub-image"', 'TF_TP_MASTER="127.0.0.1"', 'NCCL_SOCKET_IFNAME="lo"', 'NCCL_IB_HCA="none"']
        tmp = config.with_suffix(f".{threading.get_ident()}")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(config)  # atomic, so racing launchers never source a half-written config
        env = {
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "CONFIG": str(config),
            "DOCKER_LOG": str(tmp_path / "docker"),
        }
        if select is not None:
            env["PACK"] = select
        return subprocess.run(["bash", str(LAUNCHER), "1", *args], check=False, env=env, capture_output=True, text=True)

    run.state = state
    run.ran = lambda: sorted(p.name for p in tmp_path.glob("docker.*"))
    return run


KEYS = {
    "MODEL_DIR": "model",
    "ENGRAM_DIR": "engram",
    "ASSETS_DIR": "assets",
    "PREPARED_DIR": "prepared",
    "STATE_DIR": "state",
    "CACHE_DIR": "cache",
}


def legacy(state):
    """A state folder that served a pack before pack-id existed: a draft calibration table and a session."""
    (state / "calib-zig").mkdir(parents=True)
    (state / "calib-zig" / "calib-zig-tp2.json").write_text("{}")
    (state / "sessions").mkdir()
    (state / "sessions" / "s1.bin").write_bytes(b"x")


def test_new_state_dir_is_claimed_and_the_start_proceeds(node):
    r = node("eaq")
    assert r.returncode == 0, r.stderr
    assert (node.state / "pack-id").read_text() == "eaq\n"
    assert sorted(p.name for p in node.state.iterdir()) == ["pack-id", "sessions"]
    assert len(node.ran()) == 1


def test_existing_empty_state_dir_is_claimed(node):
    node.state.mkdir()
    assert node("eaq").returncode == 0
    assert (node.state / "pack-id").read_text() == "eaq\n"


@pytest.mark.parametrize("pack", ["eaq", "flash17"])
def test_used_unmarked_state_dir_is_refused_and_left_untouched(node, pack):
    legacy(node.state)
    before = sorted(p.relative_to(node.state) for p in node.state.rglob("*"))
    r = node(pack)
    assert r.returncode == 2
    assert "marked for no pack" in r.stderr and "own new STATE_DIR" in r.stderr
    assert not (node.state / "pack-id").exists()
    assert sorted(p.relative_to(node.state) for p in node.state.rglob("*")) == before
    assert node.ran() == []


def test_aot_needs_file_alone_is_a_used_folder(node):
    node.state.mkdir()
    (node.state / "aot-needs-r1.json").write_text("[]")
    assert node("eaq").returncode == 2
    assert not (node.state / "pack-id").exists()


def test_check_aot_claims_before_it_writes_into_state(node):
    r = node("eaq", "--check-aot")
    assert r.returncode == 0, r.stderr
    assert (node.state / "pack-id").read_text() == "eaq\n"
    assert len(node.ran()) == 1
    assert node("eaq").returncode == 0  # the real start after it still finds its own folder


def test_check_aot_on_a_used_unmarked_folder_is_refused_before_docker(node):
    legacy(node.state)
    assert node("eaq", "--check-aot").returncode == 2
    assert node.ran() == []
    assert not (node.state / "aot-needs-r1.json").exists()


def test_dry_run_writes_nothing(node):
    r = node("eaq", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert not node.state.exists()
    assert node.ran() == []


def test_state_dir_of_another_pack_is_refused_and_never_overwritten(node):
    node.state.mkdir()
    (node.state / "pack-id").write_text("flash17\n")
    r = node("eaq")
    assert r.returncode == 2
    assert "belongs to pack flash17" in r.stderr
    assert (node.state / "pack-id").read_text() == "flash17\n"
    assert node.ran() == []


def test_state_dir_of_the_same_pack_proceeds_even_when_used(node):
    legacy(node.state)
    (node.state / "pack-id").write_text("eaq\n")
    r = node("eaq")
    assert r.returncode == 0, r.stderr
    assert (node.state / "pack-id").read_text() == "eaq\n"
    assert len(node.ran()) == 1


@pytest.mark.parametrize("content", ["", "\n", "EAQ\n", "eaq\nflash17\n", "../x\n", "e aq\n"])
def test_malformed_pack_id_is_refused_and_kept(node, content):
    node.state.mkdir()
    (node.state / "pack-id").write_text(content)
    r = node("eaq")
    assert r.returncode == 2
    assert "holds no pack id" in r.stderr
    assert (node.state / "pack-id").read_text() == content
    assert node.ran() == []


def test_pack_id_that_is_not_a_regular_file_is_refused(node):
    node.state.mkdir()
    (node.state / "pack-id").symlink_to(node.state / "elsewhere")
    assert node("eaq").returncode == 2
    assert node.ran() == []


def test_default_config_never_claims_and_keeps_a_legacy_folder_working(node):
    legacy(node.state)
    r = node(None)
    assert r.returncode == 0, r.stderr
    assert not (node.state / "pack-id").exists()
    assert len(node.ran()) == 1


@pytest.mark.parametrize(
    ("mark", "select", "ok"), [("flash17", None, True), ("eaq", None, False), ("eaq", "eaq", True)]
)
def test_default_config_is_refused_a_folder_marked_for_another_pack(node, mark, select, ok):
    node.state.mkdir()
    (node.state / "pack-id").write_text(f"{mark}\n")
    r = node(None, select=select)
    assert (r.returncode == 0) is ok, r.stderr
    assert (node.state / "pack-id").read_text() == f"{mark}\n"


def test_racing_launchers_of_two_packs_leave_exactly_one_owner(node):
    node.state.mkdir()
    packs = ["eaq", "flash17"] * 6
    with ThreadPoolExecutor(len(packs)) as pool:
        results = list(pool.map(node, packs))
    owner = (node.state / "pack-id").read_text()
    assert owner in ("eaq\n", "flash17\n")
    for pack, r in zip(packs, results):
        assert r.returncode == (0 if f"{pack}\n" == owner else 2), (pack, r.stderr)
    assert len(node.ran()) == packs.count(owner.strip())
    assert not [p for p in os.listdir(node.state) if p.startswith(".pack-id.")]
