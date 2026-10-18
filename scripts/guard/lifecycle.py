"""Mandatory native start/watch checks: exit0 permit,75 hold,78 durable STOP or invalid config."""

import argparse
import json
import time
from pathlib import Path

from backend import Backend
from common import load_config


def decision(config, state, operation, backend, now=None):
    now = time.time() if now is None else now
    owned, mode = backend.ownership()
    if not owned:
        return 75, "foreign/unowned lifecycle: " + mode
    if state.get("version") != 1 or state.get("hard_stop"):
        return 78, "durable hard STOP/invalid state"
    if state.get("clock_verified") is False:
        return 75, "clock service proof unavailable; no restart/heal"
    current = backend.profile()[0]
    target = state.get("bootstrap_profile") or state.get("profile")
    if current != target:
        return 75, "profile mismatch; explicit operator prepare required"
    if not 0 <= now - state.get("last_update_epoch", 0) <= 6 or state.get("fresh") is not True:
        return 75, "guard/sampler stale; no automatic native restart/heal"
    if operation == "start" and state.get("bootstrap_profile") == current:
        return 0, "explicit different-profile native bootstrap only; admission remains blocked"
    if state.get("terminal_stop"):
        return 78, "durable terminal STOP; no reboot/watchdog reset"
    if operation == "watch" and state.get("bootstrap_profile") is not None:
        return 75, "pending explicit profile bootstrap; no watchdog probes/heals"
    if state.get("paused") or state.get("transitioning") or state.get("operator_hold"):
        return 75, "thermal/operator hold"
    return 0, "permitted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["check", "status"])
    parser.add_argument("--operation", choices=["start", "watch"], default="watch")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        state = json.loads(Path(config["state_path"]).read_bytes())
        code, reason = decision(config, state, args.operation, Backend(config))
        print(json.dumps({"code": code, "reason": reason, "operation": args.operation, "state": state}))
    except Exception as exc:
        print(json.dumps({"code": 78, "reason": "fail-closed:" + str(exc)}))
        raise SystemExit(78) from exc
    raise SystemExit(code if args.command == "check" else 0)


if __name__ == "__main__":
    main()
