"""Golden trace of the reference lookup plan's decisions over scripted requests for draft/lookup.zig's test."""

import json, random, sys


def main(src, out):
    sys.path.insert(0, src)
    from tensorfold.families.deepseek_v41.cuda import lookup as L
    from tensorfold.families.deepseek_v41.cuda.depth import Costs

    tables = {
        "default": Costs([42.0 + 5.0 * i for i in range(16)], 5.6),
        "measured": Costs([21.291 + 3.6445 * i + 0.013 * i * i for i in range(64)], 3.677),
    }
    scenarios = []
    for seed in range(24):
        rnd = random.Random(1000 + seed)
        vocab = 40 + seed * 7
        chunks = [[rnd.randrange(3, vocab) for _ in range(rnd.randrange(5, 40))] for _ in range(6)]
        eos = [2]
        def text(n):
            t = []
            while len(t) < n:
                if rnd.random() < 0.55:
                    c = rnd.choice(chunks)
                    a = rnd.randrange(0, len(c))
                    t += c[a:a + rnd.randrange(3, len(c) + 1)]
                else:
                    t += [rnd.randrange(3, vocab) for _ in range(rnd.randrange(1, 9))]
                if rnd.random() < 0.03:
                    t.append(2)
            return t[:n]
        prompt = text(rnd.randrange(20, 400))
        truth = text(600)
        costs = tables["measured" if seed % 2 else "default"]
        min_match = [4, 4, 3, 6][seed % 4]
        max_rows = [16, 16, 8, 12][seed % 4]
        p = L.Planner(costs, dspark=True, cap=5, lookup=True, min_match=min_match, max_rows=max_rows, eos=eos, skip=False)
        r = p.start(prompt)
        max_tokens = rnd.randrange(150, 520)
        pos = 0
        rounds = []
        while pos < max_tokens and pos < len(truth) - 20:
            left = max_tokens - pos
            depth, drafts = p.window(r, left, True, True)
            kind = r.kind
            if kind == "l":
                kept = 0
                while kept < len(drafts) and drafts[kept] == truth[pos + kept]:
                    kept += 1
                rows = 1 + len(drafts)
            elif kind == "d":
                d = rnd.randrange(0, depth + 1)
                kept = rnd.randrange(0, d + 1)
                rows = 1 + d
            else:
                kept, rows = 0, 1
            emitted = truth[pos:pos + kept + 1]
            p.observe(r, rows, kept, emitted)
            rounds.append({"left": left, "kind": kind, "drafts": [int(x) for x in drafts], "rows": rows, "kept": kept,
                           "emitted": [int(x) for x in emitted]})
            pos += kept + 1
        scenarios.append({"costs": {"verify": costs.verify, "draft": costs.draft}, "min_match": min_match,
                          "max_rows": max_rows, "eos": eos, "prompt": prompt, "rounds": rounds})
    lookups = sum(x["kind"] == "l" for s in scenarios for x in s["rounds"])
    total = sum(len(s["rounds"]) for s in scenarios)
    with open(out, "w") as f:
        json.dump({"generator": "tools/lookup_golden.py on the reference 8474f31 lookup.py", "scenarios": scenarios}, f)
    print(f"{len(scenarios)} scenarios, {total} rounds, {lookups} lookup rounds")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
