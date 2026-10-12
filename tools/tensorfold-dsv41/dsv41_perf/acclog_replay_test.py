"""Regression checks for trace joins and censored acceptance, not runtime token exactness."""
import json
from pathlib import Path
import tempfile
import unittest

from acclog_replay import empirical, read_rounds, suffix_gain


class TraceTests(unittest.TestCase):
    def test_timestamp_join_quarantines_copy_and_reused_round_ids(self):
        choose = dict(ev='choose', r=7, t=10, windows=1, rows=6, others=0,
                      asks=[dict(slot=0, start=100, k=2, most=5, q=[.9]*5)])
        commit = dict(ev='commit', r=999, t=11, slot=0, rows=3, keep=3,
                      drafted=True, max_new=384)
        copy = dict(commit, t=12, rows=12, keep=4, drafted=False)
        next_choose = dict(choose, t=20, asks=[dict(choose['asks'][0], start=107)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in (copy, next_choose, commit, choose)))
            rounds, orphan = read_rounds(path)
        self.assertEqual(rounds[0]['commits'][0]['keep'], 3)
        self.assertEqual(rounds[0]['real_rows'], 3) # choose.rows was pre-trim
        self.assertEqual(orphan, 1)
        self.assertFalse(rounds[0]['stable']) # COPY gap is not a pure round timing sample
        self.assertFalse(rounds[1]['valid'])

    def test_rejection_identifies_zero_extra_gain(self):
        ask = dict(k=2, q=[.9]*5)
        self.assertEqual(suffix_gain(ask, 5, 2, lambda j, q: q), (0., 0))
        gain, upper = suffix_gain(ask, 5, 3, lambda j, q: q)
        self.assertAlmostEqual(gain, .9 + .9**2 + .9**3)
        self.assertEqual(upper, 3)

    def test_unreached_positions_are_censored(self):
        row = dict(bench=True, asks=[dict(slot=0, k=5, q=[.9]*5)], commits={0: dict(keep=2)})
        pos, _, chance = empirical([row])
        self.assertEqual(pos, [[1, 1], [1, 0], [0, 0], [0, 0], [0, 0]])
        self.assertEqual(chance(4, .9), .9) # no fabricated failures in an unseen suffix

    def test_dense_bin_preserves_within_bin_confidence(self):
        row = dict(bench=True, asks=[dict(slot=0, k=1, q=[.9]*5)], commits={0: dict(keep=2)})
        _, _, chance = empirical([row]*10)
        self.assertAlmostEqual(chance(0, .91), .91 * 12 / 11)


if __name__ == '__main__':
    unittest.main()
