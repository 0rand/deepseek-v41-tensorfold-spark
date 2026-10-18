"""Golden for draft/calib_measure.zig: the reference boot calibration over a fake two-rank forward, and its helpers."""

import importlib, json, random, sys, types


def load(ref, glm):
    """calib_golden.py's loader, with a protocol stub that has Piece and Rows (measure imports them)."""
    for n in ("tensorfold", "tensorfold.families", "tensorfold.families.deepseek_v41"):
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    sys.modules["tensorfold"].__path__ = [glm + "/tensorfold"]
    sys.modules["tensorfold.families"].__path__ = [glm + "/tensorfold/families"]
    pkg = types.ModuleType("tensorfold.families.deepseek_v41.cuda"); pkg.__path__ = [ref]; sys.modules[pkg.__name__] = pkg
    seg = lambda name: type(name, (tuple,), {"__new__": lambda c, slot, start, ids: tuple.__new__(c, (slot, start, ids))})
    for stub, attrs in (("forward", {"Seg": object}), ("phases", {"PH": None}),
                        ("protocol", {"Candidates": object, "Rows": seg("Rows"), "Piece": seg("Piece")})):
        m = types.ModuleType(pkg.__name__ + "." + stub); m.__dict__.update(attrs); sys.modules[m.__name__] = m
    get = lambda n: importlib.import_module(pkg.__name__ + "." + n)
    return get("calib"), importlib.import_module("tensorfold.families.glm5_next.spark.calib")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def perf_counter(self):
        return self.t


class Ids:
    def __init__(self, t):
        self.t = t

    def __getitem__(self, k):
        return self.t


class FakeFw:
    """Windows and the DSpark pass take a pseudo-random time (rank-dependent); the timed section is recorded."""

    def __init__(self, clock, rank, seed, draft):
        self.clock, self.rank, self.rnd = clock, rank, random.Random(seed * 7 + rank)
        self.calls, self.samples, self.timing, self.prefills = [], [], False, 0
        self.drafter = self if draft else None

    def reset(self, slot):
        pass

    def prefill(self, pieces, mode):
        self.prefills += 1
        self.timing = self.prefills >= 2 * len(self.slots_seen)

    def cost(self, rows, segs):
        base = 21.0 + 4.37 * rows + 0.613 * (segs - 1) + 0.171 * self.rank
        x = self.rnd.random()
        noise = 0.83 + 0.3 * self.rnd.random() if x < 0.2 else 0.05 * self.rnd.random()
        return base + noise

    def window(self, rows, count):
        t0 = self.clock.t
        self.clock.t += self.cost(sum(len(r[2]) for r in rows), len(rows)) / 1e3
        if self.timing:             # measure's (perf_counter() - t0) * 1e3 around the call: these floats
            self.calls.append([[int(r[0]), len(r[2])] for r in rows])
            self.samples.append((self.clock.t - t0) * 1e3)
        return types.SimpleNamespace(ids=Ids(1000 + len(self.calls)))

    def commit(self, slot, k):
        pass

    def propose(self, slots, anchors, starts):
        t0 = self.clock.t
        self.clock.t += (3.6 + 0.2 * self.rnd.random() + 0.05 * self.rank) / 1e3
        self.calls.append("draft")
        self.samples.append((self.clock.t - t0) * 1e3)


def run_case(C, case, seed):
    how = C.method(case["env"])
    prompt = list(range(5, 5 + 120))
    gathered = {}
    out = {}
    for rank in (1, 0):
        clock = Clock()
        C.time = types.SimpleNamespace(perf_counter=clock.perf_counter)
        fw = FakeFw(clock, rank, seed, case["draft"])
        wide = C.wide_buckets(case["slots"], case["cap"])
        n_slots = min(max(1, case["slots"]), max(2, -(-max(wide) // C.SLOT_ROWS)) if wide else 2)
        fw.slots_seen = list(range(n_slots))

        def gather_max(vals, rank=rank):
            gathered[rank] = list(vals)
            if rank == 1:
                return list(vals)
            return [max(a, b) for a, b in zip(vals, gathered[1])]

        costs = C.measure(fw, prompt, ensure=lambda s, n: None, gather_max=gather_max, sync=lambda: None,
                          slots=case["slots"], max_rows=case["cap"], how=dict(how))
        out[rank] = {"calls": fw.calls, "samples": fw.samples, "ints": gathered[rank]}
        if rank == 0:
            out["costs"] = {"verify": costs.verify, "draft": costs.draft, "slot": costs.slot}
    assert out[0]["calls"] == out[1]["calls"]
    return {"slots": case["slots"], "cap": case["cap"], "env": case["env"], "draft": case["draft"], "method": how,
            "n": n_slots, "wide": C.wide_buckets(case["slots"], case["cap"]), "calls": out[0]["calls"],
            "samples": [out[0]["samples"], out[1]["samples"]], "ints": [out[0]["ints"], out[1]["ints"]],
            "costs": out["costs"]}


def main():
    C, G = load(sys.argv[1], sys.argv[2])
    cases = [
        {"slots": 4, "cap": 64, "env": {}, "draft": True},
        {"slots": 1, "cap": 64, "env": {}, "draft": True},
        {"slots": 2, "cap": 64, "env": {"TF_DSV41_CALIB_SHAPE": "fit", "TF_DSV41_CALIB_STAT": "min"}, "draft": False},
        {"slots": 4, "cap": 32, "env": {"TF_DSV41_CALIB_WARM": "1", "TF_DSV41_CALIB_REPS": "4", "TF_DSV41_CALIB_RECHECK": "2",
                                        "TF_DSV41_CALIB_CYCLE": "0", "TF_DSV41_CALIB_DEEP_REPS": "2"}, "draft": True},
        {"slots": 8, "cap": 48, "env": {"TF_DSV41_CALIB_CYCLE": "8", "TF_DSV41_CALIB_RECHECK": "8"}, "draft": True},
    ]
    rnd = random.Random(20261010)
    pure = {"statistic": [], "table": [], "extend": [], "split": [], "wide": [], "method": [], "fallback": []}
    for k in range(12):
        xs = [round(rnd.uniform(20, 80), rnd.choice([1, 3, 6])) for _ in range(rnd.choice([1, 2, 3, 4, 5, 6, 7]))]
        pure["statistic"].append({"xs": xs, "median": C.statistic(xs, "median"), "min": C.statistic(xs, "min")})
    for k in range(10):
        n = rnd.choice([1, 2, 3, 5, 8, 9, 12, 16])
        times = [20 + 4.4 * r + rnd.uniform(-3, 3) for r in range(1, n + 1)]
        pure["table"].append({"times": times, "raw": C.table_of(times, "raw"), "fit": C.table_of(times, "fit")})
    for k in range(6):
        table = C.table_of([20 + 4.4 * r + rnd.uniform(-2, 2) for r in range(1, 17)], "raw")
        pts = {b: 60 + 1.9 * b + rnd.uniform(-15, 5) for b in rnd.sample([24, 32, 48, 64], rnd.randint(0, 4))}
        pure["extend"].append({"table": table, "points": sorted([b, v] for b, v in pts.items()), "out": C.extend(table, pts)})
    for total, n in ((24, 2), (32, 2), (48, 3), (64, 4), (7, 3), (8, 2), (5, 4)):
        pure["split"].append({"total": total, "n": n, "out": C.split(total, n)})
    for slots in (1, 2, 3, 4, 8):
        for cap in (16, 24, 40, 64, 100):
            pure["wide"].append({"slots": slots, "cap": cap, "out": C.wide_buckets(slots, cap)})
    for env in ({}, {"TF_DSV41_CALIB_SHAPE": "fit"}, {"TF_DSV41_CALIB_SHAPE": "FIT", "TF_DSV41_CALIB_WARM": "0"},
                {"TF_DSV41_CALIB_STAT": "min", "TF_DSV41_CALIB_REPS": " 7 "}, {"TF_DSV41_CALIB_REPS": "0"},
                {"TF_DSV41_CALIB_RECHECK": "9"}, {"TF_DSV41_CALIB_SHAPE": "line"}, {"TF_DSV41_CALIB_STAT": "mean"},
                {"TF_DSV41_CALIB_CYCLE": "8", "TF_DSV41_CALIB_DEEP_REPS": "50"}, {"TF_DSV41_CALIB_DEEP_REPS": "51"}):
        try:
            pure["method"].append({"env": env, "out": C.method(env)})
        except ValueError:
            pure["method"].append({"env": env, "out": None})
    for vocab in (129280, 1000, 3, 2):
        pure["fallback"].append({"vocab": vocab, "out": G.fallback_ids(vocab)})
    pure["text"] = G.TEXT
    gold = {"cases": [run_case(C, c, i + 1) for i, c in enumerate(cases)], "pure": pure}
    with open(sys.argv[3], "w") as f:
        json.dump(gold, f, indent=None, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
