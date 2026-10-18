"""Golden for draft/calib_env.zig: the reference cached-table read, rank-0 share and depth pricing (usage: receipts)."""

import importlib, json, random, sys, types
from pathlib import Path


def load(ref, glm):
    """The pure-Python policy modules of REF (golden.py's loader; golden.py runs on import, so it is not imported)."""
    for n in ("tensorfold", "tensorfold.families", "tensorfold.families.deepseek_v41"):
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    sys.modules["tensorfold"].__path__ = [glm + "/tensorfold"]
    sys.modules["tensorfold.families"].__path__ = [glm + "/tensorfold/families"]
    pkg = types.ModuleType("tensorfold.families.deepseek_v41.cuda"); pkg.__path__ = [ref]; sys.modules[pkg.__name__] = pkg
    for stub, attrs in (("forward", {"Seg": object}), ("phases", {"PH": None}), ("protocol", {"Candidates": object, "Rows": object})):
        m = types.ModuleType(pkg.__name__ + "." + stub); m.__dict__.update(attrs); sys.modules[m.__name__] = m
    get = lambda n: importlib.import_module(pkg.__name__ + "." + n)
    return get("depth"), get("joint"), get("tree"), get("calib")


TIMES = [22.31749, 26.80051, 31.5004999, 36.0715, 40.66250, 44.9996, 47.80149, 51.3335, 57.0024, 59.8817, 60.79,
         62.12345, 68.0, 70.0005, 70.5, 72.25]
WIDE = {24: 101.0405, 32: 116.0715, 48: 148.66, 64: 198.1235}


def main():
    D, _, _, C = load(sys.argv[1], sys.argv[2])
    rnd = random.Random(20261007)
    table = C.extend(C.table_of(TIMES, "raw"), WIDE)
    want = {"slots": 1, "world": 2, "dspark": True, "context": 300000}
    # (name, shape, time, table): the newest same-shape entry with our context is the one Zig must pick
    files = [
        ("calib-0aa.json", {**want, "gpu": "NVIDIA GB10", "context": 196608}, "2026-10-07T09:00:00", [t * 0.97 for t in table]),
        ("calib-1bb.json", {**want, "gpu": "NVIDIA GB10"}, "2026-10-07T08:00:00", table),
        ("calib-2cc.json", {**want, "gpu": "NVIDIA GB10", "world": 1}, "2026-10-07T10:00:00", [t * 1.5 for t in table]),
        ("calib-3dd.json", {**want, "gpu": "NVIDIA GB10", "slots": 4}, "2026-10-07T11:00:00", [t * 1.2 for t in table]),
    ]
    out = {"want": want, "pick": "calib-1bb.json", "files": [], "asks": []}
    for name, shape, when, verify in files:
        out["files"].append({"name": name, "content": {"verify": verify, "draft": 3.61449, "slot": 0.8305,
                                                       "meta": {"shape": shape, "time": when, "image": "py"}}})
    root = Path(sys.argv[3]).parent / "_calib_tmp"
    root.mkdir(exist_ok=True)
    for f in out["files"]:
        (root / f["name"]).write_text(json.dumps(f["content"]))
    got = C.load("1bb", root)                       # calib.load: calib-<key>.json
    for f in root.iterdir():
        f.unlink()
    root.rmdir()
    costs = D.Costs.decode(got.encode(), "cached")  # rank 0's share
    out["costs"] = {"verify": costs.verify, "draft": costs.draft, "slot": costs.slot}
    for i in range(400):
        conf = [round(rnd.random() * rnd.choice((0.4, 0.8, 1.0)), 6) for _ in range(5)]
        most = rnd.randint(0, 5)
        first = rnd.choice((None, rnd.randint(0, 99)))
        dep = D.Depth(costs, block=5, mode="cost", cap=5, skip=True, joint=1)
        if i % 3 == 2:
            sibs = [round(rnd.random() * 0.5, 6) for _ in range(rnd.randint(1, 3))]
            k, lens = dep.choose_tree(0, conf, most, sibs, alone=True, first=first, dup_ms=0.4, sib_rows=4)
            out["asks"].append({"conf": conf, "most": most, "first": first, "sibs": sibs, "k": k, "lens": lens})
        else:
            k = dep.choose(0, conf, most, alone=True, first=first)
            out["asks"].append({"conf": conf, "most": most, "first": first, "k": k})
    Path(sys.argv[3]).write_text(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
