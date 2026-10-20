"""Bounded atomic two-node telemetry; missing worker remains explicitly missing/fail-closed."""

import argparse
import concurrent.futures
import glob
import json
import re
import subprocess
import time
from pathlib import Path

from common import atomic, load_config

NODE_SAMPLE = r"""
import glob,json,re,time
from pathlib import Path
mem=Path('/proc/meminfo').read_text();available=int(re.search(r'^MemAvailable:\s+(\d+) kB$',mem,re.M)[1])/1048576
values=[]
for f in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
 try:
  n=float(Path(f).read_text())/1000
  if 0<n<150:values.append(n)
 except (OSError,ValueError):pass
assert values
print(json.dumps({'epoch':time.time(),'available_gib':available,'max_zone_c':max(values)}))
"""


def local_sample():
    memory = Path("/proc/meminfo").read_text()
    available = int(re.search(r"^MemAvailable:\s+(\d+) kB$", memory, re.MULTILINE)[1]) / 1048576
    values = []
    for name in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            value = float(Path(name).read_text()) / 1000
            if 0 < value < 150:
                values.append(value)
        except (OSError, ValueError):
            pass
    if not values:
        raise ValueError("no valid thermal zones")
    return {"epoch": time.time(), "available_gib": available, "max_zone_c": max(values)}


def worker_sample(config):
    import shlex

    result = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=2",
            config["worker_ssh"],
            "python3 -c " + shlex.quote(NODE_SAMPLE),
        ],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    period = config.get("sampler_period_s", 2.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            started = time.monotonic()
            result = {"samples": {}, "errors": {}}
            jobs = {"0": executor.submit(local_sample), "1": executor.submit(worker_sample, config)}
            for rank, job in jobs.items():
                try:
                    # Fixed per-rank bound (ssh 4 s, job 5 s) stays under the 6 s freshness window for every
                    # period; a slow rank stretches this loop rather than overlapping samples.
                    result["samples"][rank] = job.result(timeout=5)
                except Exception as exc:
                    result["errors"][rank] = str(exc)
            atomic(config["samples_path"], result)
            time.sleep(max(0.1, period - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
