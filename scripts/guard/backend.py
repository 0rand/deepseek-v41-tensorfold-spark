"""Ownership-qualified native identity, clock and hard-stop actions; no broad process killing."""

import hashlib
import http.client
import json
import os
import re
import shlex
import stat
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from common import RANKS

CLOCK_PROGRAM = r"""
import json,os,subprocess,sys,time
from pathlib import Path
def run(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20,check=True);return r.stdout+r.stderr
def heat():
 values=[]
 for f in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
  try: values.append(float(f.read_text())/1000)
  except (OSError,ValueError): pass
 assert values and 0<max(values)<150
 return {'epoch':time.time(),'max_zone_c':max(values)}
unit,dropin,ladder=sys.argv[1],Path(sys.argv[2]),[int(x) for x in sys.argv[3].split(',')]
source,target=map(int,sys.argv[4:6])
assert source in ladder[:-1] and target==ladder[ladder.index(source)+1]
props=['-p','ExecStart','-p','ActiveState','-p','UnitFileState','-p','Result','-p','ExecMainStatus']
before=run(['systemctl','show',unit]+props)
assert f'300,{source} ' in before and 'ActiveState=active' in before and 'UnitFileState=enabled' in before
raw=dropin.read_text();assert not dropin.is_symlink()
expected=(f'[Unit]\nDescription=GPU SM clock cap (300-{source} MHz)\n'
          f'[Service]\nExecStart=\nExecStart=/usr/bin/nvidia-smi -lgc 300,{source}\n')
assert raw==expected
thermal_before=heat();assert thermal_before['max_zone_c']<=88
updated=raw.replace(f'300-{source}',f'300-{target}').replace(f'300,{source}',f'300,{target}')
tmp=dropin.with_name(dropin.name+'.thermal-tmp')
fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,dropin.stat().st_mode&0o777)
try:
 with os.fdopen(fd,'w') as out:out.write(updated);out.flush();os.fsync(out.fileno())
 assert dropin.read_text()==raw
 os.replace(tmp,dropin)
 directory=os.open(dropin.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
finally:
 if tmp.exists():tmp.unlink()
run(['systemctl','daemon-reload']);applied=run(['systemctl','restart',unit])
after=run(['systemctl','show',unit]+props)
required=(f'300,{target} ','ActiveState=active','UnitFileState=enabled','Result=success','ExecMainStatus=0')
assert all(x in after for x in required)
print(json.dumps({'before':before,'after':after,'before_file':raw,'after_file':dropin.read_text(),
                  'source_mhz':source,'target_mhz':target,'thermal_before':thermal_before,'thermal_after':heat(),
                  'apply_stdout':applied}))
"""


def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("nonregular ownership evidence")
    return info


class Backend:
    def __init__(self, config):
        self.config = config
        self.upstream = urlsplit(config["upstream"])
        self.ladder = tuple(config["clock_ladder_mhz"])

    def run(self, rank, args, timeout=6):
        command = (
            args
            if rank == "0"
            else [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=3",
                self.config["worker_ssh"],
                shlex.join(args),
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
        return result.stdout + result.stderr if args[:2] == ["docker", "logs"] else result.stdout

    def profile(self):
        path = Path(self.config["production_config"])
        regular(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in self.config["profiles"]:
            raise ValueError("unapproved prod config SHA")
        return sha, self.config["profiles"][sha]

    def ownership(self):
        """A fresh operator maintenance lease makes the guard stand aside: no STOP actions, no admissions."""
        lease = Path(self.config["lease_path"])
        if os.path.lexists(lease):
            info = regular(lease)
            if time.time() - info.st_mtime < self.config.get("lease_minutes", 20) * 60:
                return False, "operator_lease"
        return True, "owned"

    def identity(self, rank, expected_sha, require_running=True):
        owned, mode = self.ownership()
        sha, profile = self.profile()
        if not owned or sha != expected_sha:
            raise ValueError("ownership/profile changed: " + mode)
        item = json.loads(self.run(rank, ["docker", "inspect", self.config["containers"][rank]]))[0]
        if item["Image"] != profile["image_ids"][rank] or item["Name"] != "/" + self.config["containers"][rank]:
            raise ValueError("container image/name mismatch")
        if require_running and item["State"]["Running"] is not True:
            raise ValueError("owned native container not running")
        host = item.get("HostConfig")
        if not isinstance(host, dict) or host.get("CpusetCpus") != profile.get("cpu_sets", {}).get(rank, ""):
            raise ValueError("container host CPU set changed or unavailable")
        mounts = {m["Destination"]: m["Source"] for m in item["Mounts"]}
        if mounts != profile["mounts"][rank]:
            raise ValueError("container mounts changed")
        model = [m for m in item["Mounts"] if m["Destination"] == "/model"]
        if len(model) != 1 or model[0]["RW"] is not False:
            raise ValueError("model mount must be read only")
        env = dict(
            value.split("=", 1)
            for value in item["Config"]["Env"]
            if (value.split("=", 1)[0].startswith(("TF_DSV41_", "GLM53_TF_"))
                or value.split("=", 1)[0] in ("TF_CUDA_SCHED", "TF_TP_ROCE_VAR_KB"))
        )
        env.pop("TF_DSV41_LAUNCH_T0", None)
        if env != profile["inference_environment"][rank]:
            raise ValueError("inference environment changed")
        return {"id": item["Id"], "image": item["Image"], "profile": sha, "lease_mode": mode}

    def request_token_count(self, body, path):
        if path.split("?", 1)[0] not in ("/v1/chat/completions", "/v1/completions"):
            return {"count_source": "unsupported_route"}
        document = json.loads(body)
        # Avoid running image preparation twice merely for STOP metadata.
        if any(isinstance(m.get("content"), list) for m in document.get("messages", [])):
            return {"count_source": "multimodal_not_retokenized"}
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=3)
        try:
            connection.request("POST", "/tokenize", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("native tokenize failed")
            value = json.loads(response.read(16 * 1024 * 1024))
            count = value["count"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid native token count")
            return {"prompt_tokens": count, "count_source": "native_tokenize"}
        finally:
            connection.close()

    def stop_request_context(self, evidence):
        requests = evidence.get("request_metadata", {})
        if not requests:
            return {"requests": {}, "resume_observations": []}
        since = min(row["epoch"] for row in requests.values())
        logs = self.run("0", ["docker", "logs", "--timestamps", "--since", str(since),
                              "--tail", "400", self.config["containers"]["0"]], timeout=4)
        observations = []
        pattern = r"sessions: slot (\d+) resumed (\d+) of (\d+) prompt tokens from RAM"
        for line in logs.splitlines():
            match = re.search(pattern, line)
            if match:
                observations.append({"slot": int(match[1]), "resumed_tokens": int(match[2]),
                                     "prompt_tokens": int(match[3]), "native_log": line})
        enriched = json.loads(json.dumps(requests))
        for observation in observations:
            matches = [key for key, row in enriched.items()
                       if row.get("prompt_tokens") == observation["prompt_tokens"]]
            same_count = [row for row in observations
                          if row["prompt_tokens"] == observation["prompt_tokens"]]
            if len(matches) == 1 and len(same_count) == 1:
                enriched[matches[0]].update(resumed_tokens=observation["resumed_tokens"],
                                           resumed_source="native_session_log")
        return {"requests": enriched, "resume_observations": observations}

    def health(self):
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("native health not200")
            return json.loads(response.read(1048576))
        finally:
            connection.close()

    def idle(self):
        health = self.health()
        if health.get("ok") is not True or any(health.get(key, 0) != 0 for key in ("requests_running", "queued")):
            raise ValueError("native not healthy/idle")
        if health.get("inflight") != 0:
            raise ValueError("explicit native inflight0 required")
        return health

    def boot_proof(self, expected_sha):
        identities = {rank: self.identity(rank, expected_sha) for rank in RANKS}
        health = self.idle()
        if health.get("streams", {}).get("max") != self.config["limits"]["max_inflight"]:
            raise ValueError("native slots must equal the guard's max_inflight")
        records = {}
        for rank in RANKS:
            logs = self.run(rank, ["docker", "logs", "--tail", "400", self.config["containers"][rank]])
            lines = [line for line in logs.splitlines() if "boot growth" in line and "worst serving point" in line]
            line = lines[-1] if lines else ""
            match = re.search(r"worst serving point ([0-9.]+) GiB", line)
            if match is None or float(match[1]) < 4 or "pool does not fit" in logs:
                raise ValueError("current-container native boot floor proof missing/failed")
            records[rank] = {"identity": identities[rank], "native_boot_line": line, "worst_gib": float(match[1])}
        return {"native_health": health, "ranks": records, "epoch": time.time()}

    def actual_clock(self):
        caps = []
        props = ["-p", "ExecStart", "-p", "ActiveState", "-p", "UnitFileState", "-p", "Result", "-p", "ExecMainStatus"]
        for rank in RANKS:
            output = self.run(rank, ["systemctl", "show", self.config["clock_service"], *props])
            matches = [cap for cap in self.ladder if "300," + str(cap) + " " in output]
            if len(matches) != 1 or not all(
                word in output
                for word in ("ActiveState=active", "UnitFileState=enabled", "Result=success", "ExecMainStatus=0")
            ):
                raise ValueError("actual persistent clock cap service not active on a ladder cap")
            caps.append(matches[0])
        if caps[0] != caps[1]:
            raise ValueError("node clocks disagree")
        return caps[0]

    def lower(self, expected_sha, permit):
        source = self.actual_clock()
        if source not in self.ladder[:-1]:
            raise ValueError("no ladder step below the floor cap")
        target = self.ladder[self.ladder.index(source) + 1]
        receipts = []
        for rank in RANKS:
            before = permit()
            if before["clock_mhz"] != source:
                raise ValueError("durable/current cap mismatch before step")
            first = {r: self.identity(r, expected_sha) for r in RANKS}
            native = self.idle()
            # Both identities, ownership and real native idle rechecked before EACH node operation.
            if any(self.identity(r, expected_sha)["id"] != first[r]["id"] for r in RANKS):
                raise ValueError("container identity changed before cap step")
            output = self.run(
                rank,
                ["sudo", "-n", "python3", "-c", CLOCK_PROGRAM,
                 self.config["clock_service"], self.config["clock_dropin"],
                 ",".join(map(str, self.ladder)), str(source), str(target)],
                timeout=60,
            )
            receipts.append(
                {
                    "rank": rank,
                    "epoch": time.time(),
                    "gate_before": before,
                    "backend_before": native,
                    "identities": first,
                    "clock": json.loads(output),
                }
            )
        if self.actual_clock() != target:
            raise ValueError("both clocks disagree with target after step")
        return receipts

    def hold_heals(self, expected_sha):
        owned, _ = self.ownership()
        if not owned or self.profile()[0] != expected_sha:
            raise ValueError("ownership changed before watchdog hold")
        units = self.config.get("watchdog_units", [])
        if not units:
            return
        subprocess.run(
            ["systemctl", "--user", "stop", *units],
            capture_output=True,
            timeout=5,
            check=True,
        )

    def stop_owned(self, expected_sha):
        records = []
        self.hold_heals(expected_sha)
        for rank in RANKS:
            try:
                before = self.identity(rank, expected_sha, require_running=False)
                if self.identity(rank, expected_sha, require_running=False)["id"] != before["id"]:
                    raise ValueError("container identity changed before hard stop")
                output = self.run(rank, ["docker", "stop", "--time", "0", before["id"]], timeout=8)
                records.append({"rank": rank, "id": before["id"], "stopped": True, "output": output})
            except Exception as exc:
                records.append({"rank": rank, "stopped": False, "error": str(exc)})
        return records
