"""Golden trace of the reference speculative pass's decisions over scripted rounds for draft/spec_test.zig."""

import importlib.util, json, random, sys, types


def load(ref):
    torch = types.ModuleType("torch")
    torch.Tensor = object
    sys.modules["torch"] = torch
    pkg = types.ModuleType("dsv41"); pkg.__path__ = [ref]; sys.modules["dsv41"] = pkg
    sys.modules["dsv41.vsample"] = types.ModuleType("dsv41.vsample")
    spec = importlib.util.spec_from_file_location("dsv41.spec", ref + "/spec.py")
    m = importlib.util.module_from_spec(spec); sys.modules["dsv41.spec"] = m
    spec.loader.exec_module(m)
    return m


class Host:
    """The entry's [3, Wn] (accepted, bonus, next start): indexed as the reference indexes its pinned tensor."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, ij):
        return self.rows[ij[0]][ij[1]]


class Head:
    def variant(self, slots):
        return 0


class Inp:
    n = 5

    def result(self, conf):
        return [[1] * 5 for _ in range(4)], None


def make(SP):
    dr = types.SimpleNamespace(head=Head(), valid={s: 0 for s in range(4)}, cw=None, last_conf=None,
                               tree_of=lambda inp, back: None)
    fw = types.SimpleNamespace(drafter=dr, device="cpu")
    return SP.Spec(fw)


def launch(SP, sp, cands):
    """spec.Spec.launch's bookkeeping; cands: [slot, start, accepted, bonus, noise]."""
    if sp.entry is not None:
        sp._drop(sp.entry, "unused")
    elig = [c for c in cands if sp.wanted(c[0])]
    if not elig:
        return False
    slots = tuple(sorted(int(c[0]) for c in elig))
    host = Host([[c[2] for c in elig], [c[3] for c in elig], [c[1] + c[2] + 1 for c in elig]])
    sp.entry = SP._Entry((slots, 5, 0), slots, Inp(), {c[0]: SP._Win(i, c[1], list(c[4])) for i, c in enumerate(elig)},
                         {s: sp.gen.get(s, 0) for s in slots}, {s: 0 for s in slots}, host)
    sp.stats["launched"] += 1
    return True


def state(sp):
    return {"rates": [sp.rate.get(s, 1.0) for s in range(4)], "skipped": [sp.skipped.get(s, 0) for s in range(4)],
            "launched": sp.stats["launched"], "hit": sp.stats["hit"], "differs": sp.stats["commit.differs"],
            "miss": {k[5:]: v for k, v in sp.stats.items() if k.startswith("miss.")}}


def main():
    SP = load(sys.argv[1])
    sp = make(SP)
    rnd = random.Random(7)
    pos = [100, 200, 300, 400]
    noise = [[], [11], [], [5]]
    ops = []
    for _ in range(200):
        r = rnd.random()
        if r < 0.03:
            s = rnd.randrange(4)
            sp.forget(s)
            ops.append({"op": "forget", "slot": s, "state": state(sp)})
            continue
        if r < 0.08:
            sp.clear()
            ops.append({"op": "clear", "state": state(sp)})
            continue
        live = sorted(rnd.sample(range(4), rnd.randint(1, 3)))
        cands = []
        for s in live:
            acc = rnd.randint(0, 4)
            cands.append([s, pos[s], acc, rnd.randrange(1000), noise[s]])
        got = launch(SP, sp, cands)
        ops.append({"op": "launch", "cands": cands, "got": got, "state": state(sp)})
        # the round's commits: mostly the speculated count, sometimes cut
        commits = []
        for c in cands:
            acc = c[2] if rnd.random() < 0.85 else rnd.randint(0, c[2])
            ok = sp.ingested(c[0], c[1], acc)
            commits.append([c[0], c[1], acc, ok])
            pos[c[0]] = c[1] + acc + 1
        ops.append({"op": "ingested", "commits": commits, "state": state(sp)})
        # the next round drafts: mostly the same slots at the committed pending tokens
        asks = []
        for c, (s, start, acc, _) in zip(cands, commits):
            if rnd.random() < 0.08:
                continue
            bonus = c[3] if acc == c[2] and rnd.random() < 0.95 else rnd.randrange(1000)
            nz = noise[s] if rnd.random() < 0.97 else [99]
            asks.append([s, bonus, start + acc + 1, nz])
        if not asks:
            continue
        t = sp.take([a[0] for a in asks], [a[1] for a in asks], [a[2] for a in asks], [a[3] for a in asks])
        ops.append({"op": "take", "asks": asks, "got": t is not None, "state": state(sp)})
    with open(sys.argv[2], "w") as f:
        json.dump({"decay": SP.DECAY, "min_rate": SP.MIN_RATE, "probe": SP.PROBE, "ops": ops}, f)


if __name__ == "__main__":
    main()
