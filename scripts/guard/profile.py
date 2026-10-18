"""Snapshot the running, qualified pair into an exact guard profile keyed by the serving env file's SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path

from backend import Backend
from common import RANKS, SCHEDULE_KEY, atomic, validate_profiles

ENV_KEYS = ("TF_DSV41_", "GLM53_TF_")
EXTRA_KEYS = (SCHEDULE_KEY, "TF_TP_ROCE_VAR_KB")


def inference_environment(item):
    env = dict(value.split("=", 1) for value in item["Config"]["Env"] if "=" in value)
    env = {key: value for key, value in env.items() if key.startswith(ENV_KEYS) or key in EXTRA_KEYS}
    env.pop("TF_DSV41_LAUNCH_T0", None)
    return env


def snapshot(config, backend):
    sha = hashlib.sha256(Path(config["production_config"]).read_bytes()).hexdigest()
    profile = {"image_ids": {}, "mounts": {}, "inference_environment": {}, "cpu_sets": {}}
    for rank in RANKS:
        item = json.loads(backend.run(rank, ["docker", "inspect", config["containers"][rank]]))[0]
        if item["State"]["Running"] is not True:
            raise ValueError("rank " + rank + " is not running; snapshot a qualified, running pair")
        profile["image_ids"][rank] = item["Image"]
        profile["mounts"][rank] = {m["Destination"]: m["Source"] for m in item["Mounts"]}
        profile["inference_environment"][rank] = inference_environment(item)
        profile["cpu_sets"][rank] = (item.get("HostConfig") or {}).get("CpusetCpus", "")
    if not any(profile["cpu_sets"].values()):
        del profile["cpu_sets"]
    validate_profiles({sha: profile})
    return sha, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--write", action="store_true", help="add the profile to the config file")
    args = parser.parse_args()
    # The config may not hold a profile yet, so it is read without load_config's full validation.
    config = json.loads(Path(args.config).read_bytes())
    sha, profile = snapshot(config, Backend(config))
    if args.write:
        config.setdefault("profiles", {})[sha] = profile
        atomic(args.config, config)
    print(json.dumps({"sha256": sha, "profile": profile}, indent=1))


if __name__ == "__main__":
    main()
