"""Golden traces of the reference DSpark depth, joint depth, tree plan and calibration for draft/golden_test.zig."""

import importlib, json, math, random, sys, types


def load(ref, glm):
    for n in ("tensorfold", "tensorfold.families", "tensorfold.families.deepseek_v41"):
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    sys.modules["tensorfold"].__path__ = [glm + "/tensorfold"]
    sys.modules["tensorfold.families"].__path__ = [glm + "/tensorfold/families"]
    pkg = types.ModuleType("tensorfold.families.deepseek_v41.cuda"); pkg.__path__ = [ref]; sys.modules[pkg.__name__] = pkg
    for stub, attrs in (("forward", {"Seg": object}), ("phases", {"PH": None}), ("protocol", {"Candidates": object, "Rows": object})):
        m = types.ModuleType(pkg.__name__ + "." + stub); m.__dict__.update(attrs); sys.modules[m.__name__] = m
    get = lambda n: importlib.import_module(pkg.__name__ + "." + n)
    return get("depth"), get("joint"), get("tree"), get("calib")


ROWS = [21.954, 26.573, 31.977, 36.158, 40.852, 44.776, 47.237, 51.089, 57.951, 59.663, 60.793, 62.458, 69.962, 70.85,
        70.971, 72.362]
WIDE = {24: 100.2 + 0.83, 32: 115.2 + 0.83, 48: 147.1 + 2 * 0.83, 64: 196.3 + 3 * 0.83}
DRAFT, SLOT = 3.61, 0.83


def main():
    D, J, T, C = load(sys.argv[1], sys.argv[2])
    table = C.table_of(ROWS, "raw")
    points = {b: t - SLOT * (-(-b // 16) - 1) for b, t in WIDE.items()}
    costs = D.Costs(C.extend(table, points), DRAFT, "measured", SLOT)
    rnd = random.Random(20261006)
    out = {"costs": {"verify": costs.verify, "draft": costs.draft, "slot": costs.slot}, "traces": [], "plans": [],
           "allocs": [], "weights": []}
    for mode in (1, 2):
        dep = D.Depth(costs, block=5, mode="cost", cap=5, skip=True, joint=mode)
        ops = []

        def conf():
            hi = rnd.choice((0.3, 0.7, 1.0))
            return [round(rnd.random() * hi, 6) for _ in range(5)]

        for step in range(240):
            n = rnd.choice((1, 1, 2, 3, 4))
            slots = rnd.sample(range(6), n)
            others = [rnd.randint(1, 8) for _ in range(rnd.choice((0, 0, 1)))]
            asks = [{"slot": s, "conf": conf(), "most": rnd.randint(0, 5), "first": rnd.choice((None, rnd.randint(0, 99)))}
                    for s in slots]
            if n == 1 and not others and rnd.random() < 0.3:
                a = asks[0]
                sibs = [round(rnd.random() * 0.5, 6) for _ in range(rnd.randint(1, 3))]
                k, lens = dep.choose_tree(a["slot"], a["conf"], a["most"], sibs, alone=True, first=a["first"],
                                          dup_ms=0.4, sib_rows=4)
                ops.append({"op": "tree", **a, "sibs": sibs, "k": k, "lens": lens})
                ks = [k]
            else:
                ks = dep.choose_joint([a["slot"] for a in asks], [a["conf"] for a in asks], [a["most"] for a in asks],
                                      others=others, firsts=[a["first"] for a in asks])
                ops.append({"op": "joint", "asks": asks, "others": others, "ks": ks})
            for a, k in zip(asks, ks):
                rows = 1 + k
                keep = rnd.randint(1, rows)
                tokens = keep + rnd.randint(0, 2) if rnd.random() < 0.1 else None
                dep.record(a["slot"], rows, keep, tokens=tokens)
                bonus = a["first"] if (a["first"] is not None and rnd.random() < 0.5) else rnd.randint(0, 99)
                dep.bonus(a["slot"], bonus)
                ops.append({"op": "record", "slot": a["slot"], "rows": rows, "keep": keep, "tokens": tokens, "bonus": bonus})
            if rnd.random() < 0.05:
                s = rnd.randrange(6)
                dep.reset(s)
                ops.append({"op": "reset", "slot": s})
            if step % 4 == 3:
                ops.append({"op": "check", "kept": list(dep.cal.kept), "prob": list(dep.cal.prob),
                            "rates": [dep.rate(s) for s in range(6)], "joint": [dep.joint_rate(w) for w in range(1, 6)],
                            "zero": dep.stats.zero, "reached": list(dep.stats.reached), "statkept": list(dep.stats.kept)})
        out["traces"].append({"joint": mode, "ops": ops})
    for _ in range(150):
        qm = [rnd.random() for _ in range(rnd.randint(0, 5))]
        sibs = [rnd.random() * 0.6 for _ in range(rnd.randint(0, 3))]
        cont = [rnd.random() for _ in range(rnd.randint(0, 4))]
        rate, least, dup, rows = rnd.uniform(0.005, 0.08), rnd.randint(0, 1), rnd.choice((0.0, 0.4)), rnd.randint(1, 5)
        k, lens, ms, e = T.plan(qm, sibs, cont, costs.verify, rate, least=least, dup_ms=dup, sib_rows=rows, max_rows=16)
        out["plans"].append({"qm": qm, "sibs": sibs, "cont": cont, "rate": rate, "least": least, "dup": dup,
                             "rows": rows, "k": k, "lens": lens, "ms": ms, "e": e})
    for _ in range(150):
        n = rnd.randint(1, 5)
        qs = [[rnd.random() for _ in range(5)] for _ in range(n)]
        lo = [rnd.randint(0, 1) for _ in range(n)]
        top = [rnd.randint(0, 5) for _ in range(n)]
        w = [rnd.uniform(0.3, 2.0) for _ in range(n)] if rnd.random() < 0.5 else None
        shared, rate, mx = rnd.randint(0, 10), rnd.uniform(0.005, 0.1), rnd.choice((16, 32, 64))
        ks, best = J.allocate(qs, lo, top, shared=shared, rows_ms=costs.rows_ms, rate=rate, max_rows=mx, weight=w)
        out["allocs"].append({"qs": qs, "lo": lo, "top": top, "w": w, "shared": shared, "rate": rate, "max": mx,
                              "ks": ks, "best": best})
    for _ in range(60):
        tpr = [rnd.choice((None, rnd.uniform(0.5, 6.0))) for _ in range(rnd.randint(1, 5))]
        alpha = rnd.choice((0.0, 0.5, 1.0, 2.0))
        out["weights"].append({"tpr": tpr, "alpha": alpha, "w": J.weights(tpr, alpha)})
    json.dump(out, open(sys.argv[3], "w"))


main()
