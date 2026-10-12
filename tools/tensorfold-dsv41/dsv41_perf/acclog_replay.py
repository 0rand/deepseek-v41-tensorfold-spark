#!/usr/bin/env python3
"""One-round policy replay on baseline traces; suffix outcomes and future rounds remain counterfactual."""
import argparse
import collections
import json
from pathlib import Path
import statistics

BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64)


def read_rounds(path):
    rounds, current, orphan = [], None, 0
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for event in sorted(events, key=lambda e: e['t']):
        if event['ev'] == 'choose':
            current = dict(event, commits={})
            rounds.append(current)
        elif event['ev'] == 'commit':
            if current is None or event['slot'] not in {a['slot'] for a in current['asks']}:
                orphan += 1
                continue
            ask = next(a for a in current['asks'] if a['slot'] == event['slot'])
            if event['rows'] != 1 + ask['k'] or not event['drafted'] or event['slot'] in current['commits']:
                orphan += 1 # COPY/zero-depth rounds can have no choose and inherit the previous logger id
                continue
            current['commits'][event['slot']] = event
    for i, row in enumerate(rounds):
        row['real_rows'] = row['others'] + sum(1 + a['k'] for a in row['asks'])
        row['dt_ms'] = (rounds[i + 1]['t'] - row['t']) / 1e6 if i + 1 < len(rounds) else None
        row['valid'] = bool(row['asks']) and all(
            a['slot'] in row['commits'] and row['commits'][a['slot']]['rows'] == 1 + a['k']
            for a in row['asks'])
        row['bench'] = row['valid'] and all(c['max_new'] == 384 for c in row['commits'].values())
        row['stable'] = row['bench'] and i + 1 < len(rounds) and all(
            any(b['slot'] == a['slot'] and b['start'] == a['start'] + row['commits'][a['slot']]['keep']
                for b in rounds[i + 1]['asks']) for a in row['asks'])
    return rounds, orphan


def empirical(rounds):
    pos = [[0, 0] for _ in range(5)]
    bins = [[[0, 0, 0.] for _ in range(10)] for _ in range(5)]
    for row in rounds:
        if not row['bench']:
            continue
        for ask in row['asks']:
            keep = row['commits'][ask['slot']]['keep']
            for j in range(min(ask['k'], keep)):
                b = min(9, int(ask['q'][j] * 10))
                accepted = int(keep > j + 1)
                for count in (pos[j], bins[j][b]):
                    count[0] += 1
                    count[1] += accepted
                bins[j][b][2] += ask['q'][j]
    def chance(j, q):
        n, yes, sum_q = bins[j][min(9, int(q * 10))]
        # Preserve within-bin confidence; sparse bins cannot identify a correction.
        return min(.995, q * (yes + 2) / (sum_q + 2)) if n >= 10 else q
    return pos, bins, chance


def expectation(qs, k):
    survival, tokens = 1., 1.
    for q in qs[:k]:
        survival *= q
        tokens += survival
    return tokens


def suffix_gain(ask, k, keep, chance):
    """Observed prefix rejection identifies zero gain; accepted prefixes leave suffixes censored."""
    if keep < 1 + ask['k']:
        return 0., 0
    survival, gain = 1., 0.
    for j in range(ask['k'], k):
        survival *= chance(j, ask['q'][j])
        gain += survival
    return gain, k - ask['k']


def policy(row, history, forward_cap=64, context_bucket=2048, single=True, warmup=4):
    asks = row['asks']
    old = [a['k'] for a in asks]
    rows = row['real_rows']
    if row['others'] or row['windows'] != len(asks) or any(a['start'] > 4096 for a in asks):
        return old
    shared = len(asks) > 1
    bucket = next((r for r in BUCKETS if rows <= r <= forward_cap), None) if shared else None
    if shared and bucket is None or not shared and (not single or old[0] == 0):
        return old
    room = (bucket if shared else forward_cap) - rows
    candidates = []
    for i, ask in enumerate(asks):
        k, top, start = ask['k'], min(5, ask['most']), ask['start']
        h = history.get(ask['slot'], ())
        if not (len(h) >= warmup and statistics.mean(h) >= 3 and top > k and
                start + top < 16 * context_bucket and (start + k) // context_bucket == (start + top) // context_bucket):
            continue
        survival = 1.
        for j, q in enumerate(ask['q'][:top]):
            survival *= q
            if j >= k:
                candidates.append((-survival, i, j))
    candidates.sort()
    tokens = sum(expectation(a['q'], k) for a, k in zip(asks, old))
    # Shared expansions keep C fixed. Single stored table isn't in this capture: use its published secant.
    def cost(n):
        return 1. if shared else 21.175 + 3.68 * (n - 1) + 3.643
    baseline, best, count = tokens / cost(rows), tokens / cost(rows), 0
    for n, (neg, _, _) in enumerate(candidates[:max(0, room)], 1):
        tokens -= neg
        ratio = tokens / cost(rows + n)
        if ratio > best + 1e-12:
            best, count = ratio, n
    if best < baseline * 1.01:
        return old
    new = old.copy()
    for _, i, _ in candidates[:count]:
        new[i] += 1
    return new


def analyze(root, cell, cost_penalty=0., warmup=4, single=True):
    rounds, orphans = read_rounds(root / f'acclog-{cell}.jsonl')
    pos, bins, chance = empirical(rounds)
    history, ends = {}, {}
    stats = collections.Counter()
    buckets = collections.Counter()
    costs = collections.defaultdict(list)
    for row in rounds:
        if row['stable'] and row['dt_ms'] is not None:
            costs[row['real_rows']].append(row['dt_ms'])
    single_slope = 3.68
    # Difference of medians at adjacent measured single-row widths is an acceptance-independent time estimate.
    slopes = [statistics.median(costs[r + 1]) - statistics.median(costs[r])
              for r in range(2, 6) if len(costs[r]) >= 5 and len(costs[r + 1]) >= 5]
    if cell.endswith('s1') and slopes:
        single_slope = max(0., statistics.median(slopes))
    for row in rounds:
        for ask in row['asks']:
            slot = ask['slot']
            if slot in ends and ask['start'] != ends[slot]:
                history.pop(slot, None)
            history.setdefault(slot, collections.deque(maxlen=16))
        new = policy(row, history, single=single, warmup=warmup) if row['valid'] else [a['k'] for a in row['asks']]
        if row['bench']:
            stats['rounds'] += 1
            buckets[next((r for r in BUCKETS if r >= row['real_rows']), 65)] += 1
            stats['changed'] += new != [a['k'] for a in row['asks']]
            for ask, k in zip(row['asks'], new):
                commit = row['commits'][ask['slot']]
                extra = k - ask['k']
                stats['slots'] += 1
                stats['keep'] += commit['keep']
                stats['added_rows'] += extra
                if extra and commit['keep'] < 1 + ask['k']:
                    stats['known_zero'] += 1
                elif extra:
                    stats['unknown_suffix'] += 1
                    gain, upper = suffix_gain(ask, k, commit['keep'], chance)
                    stats['gain'] += gain
                    stats['upper_gain'] += upper
            if row['stable']:
                stats['time_ms'] += row['dt_ms']
                extra = sum(k - a['k'] for a, k in zip(row['asks'], new))
                stats['extra_ms'] += extra * single_slope if len(new) == 1 else (row['dt_ms'] * cost_penalty if extra else 0)
        for ask in row['asks']:
            slot = ask['slot']
            commit = row['commits'].get(slot)
            if commit:
                history[slot].append(commit['keep'])
                ends[slot] = ask['start'] + commit['keep']
    cells = json.loads((root / f'bench-acc-{cell}' / 'cells.json').read_text())['cells']
    base_speed = max(c['aggregate_tok_s'] or c['mean_tok_s'] * c['streams'] for c in cells)
    emitted = sum(sum(c['tokens']) for c in cells)
    gain_ratio = stats['gain'] / emitted
    cost_ratio = stats['extra_ms'] / stats['time_ms'] if stats['time_ms'] else 0
    return dict(cell=cell, baseline_tok_s=base_speed, predicted_tok_s=base_speed * (1 + gain_ratio) / (1 + cost_ratio),
                baseline_tpr=stats['keep'] / stats['slots'], candidate_tpr=(stats['keep'] + stats['gain']) / stats['slots'],
                fitted_position_acceptance=[dict(reached=n, accepted=y, rate=y/n if n else None) for n, y in pos],
                q_bins=bins, bucket_rounds=dict(buckets), single_extra_row_ms=single_slope, counters=dict(stats),
                orphan_commits=orphans, cost_ratio=cost_ratio,
                emitted_tokens=emitted,
                caveats=['One-round replay, observed old-policy history; future calibration/round boundaries unknown.',
                         'Suffix estimate uses measured position/q-bin acceptance conditioned on full old-prefix acceptance.',
                         'Selected/censored suffix samples may not transfer; keep is pre-emission; epochs inferred by positions.',
                         'Single decision uses published stored-cost secant; actual table absent; cost uses observed median slope.',
                         'Choose gaps include logging/host/idle/admission; only continuous-position gaps priced.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--shared-cost-penalty', type=float, default=0.)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--shared-only', action='store_true', help='Replay the follow-up policy without single-window expansion')
    args = parser.parse_args()
    print(json.dumps([analyze(args.root, cell, args.shared_cost_penalty, args.warmup, not args.shared_only)
                      for cell in ('code-t0-s1', 'code-t0-s4', 'prose-t0-s4')], indent=2))


if __name__ == '__main__':
    main()
