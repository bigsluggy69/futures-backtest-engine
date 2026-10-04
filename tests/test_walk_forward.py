"""Walk-forward layer: split geometry, no lookahead, selection logic, and
end-to-end determinism.

Most orchestration tests replace grid_search.run_single with a stub that
returns hand-written metrics per (strategy, split). That makes the selection
outcome known in advance and lets us record exactly which bars each stage was
handed. Separate tests use the REAL engine to prove that mutating future bars
cannot change earlier results.
"""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import grid_search as gs
import walk_forward as wf
from tests.synth import make_synthetic_bars, write_csv

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2024-01-01", tz="UTC")


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------

class TestFolds(unittest.TestCase):
    def folds(self, n_days=200, train=60, val=20, test=20, step=None,
              anchored=False):
        return wf.make_folds(T0, T0 + n_days * DAY, train_days=train,
                             validate_days=val, test_days=test,
                             step_days=step if step is not None else test,
                             anchored=anchored)

    def test_splits_never_overlap_and_are_time_ordered(self):
        for anchored in (False, True):
            for train, val, test, step in [(60, 20, 20, 20), (60, 20, 20, 7),
                                           (30, 10, 10, 10), (45, 15, 30, 30),
                                           (60, 20, 20, 25), (10, 5, 5, 1)]:
                folds = self.folds(200, train, val, test, step, anchored)
                self.assertGreater(len(folds), 0)
                for f in folds:
                    # Strictly ordered, back to back, nothing shared.
                    self.assertLess(f.train_start, f.train_end)
                    self.assertEqual(f.train_end, f.validate_start)
                    self.assertEqual(f.validate_end, f.test_start)
                    self.assertLess(f.validate_start, f.validate_end)
                    self.assertLess(f.test_start, f.test_end)
                    spans = [(f.train_start, f.train_end),
                             (f.validate_start, f.validate_end),
                             (f.test_start, f.test_end)]
                    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                        self.assertLessEqual(a1, b0)  # half-open: disjoint
                    # Every bar timestamp belongs to at most one window.
                    probe = pd.date_range(f.train_start, f.test_end, freq="6h",
                                          inclusive="left")
                    for t in probe:
                        hits = sum(lo <= t < hi for lo, hi in spans)
                        self.assertEqual(hits, 1)
                    self.assertLessEqual(f.test_end, T0 + 200 * DAY)

    def test_test_windows_disjoint_across_folds_when_step_ge_test_days(self):
        for anchored in (False, True):
            folds = self.folds(200, 60, 20, 20, 20, anchored)
            self.assertFalse(wf.oos_windows_overlap(folds))
            for a, b in zip(folds, folds[1:]):
                self.assertLessEqual(a.test_end, b.test_start)

    def test_overlap_is_flagged_when_step_smaller_than_test_days(self):
        folds = self.folds(200, 60, 20, 20, 7)
        self.assertTrue(wf.oos_windows_overlap(folds))

    def test_rolling_keeps_train_length_anchored_keeps_start(self):
        rolling = self.folds(200, 60, 20, 20, 20, anchored=False)
        anchored = self.folds(200, 60, 20, 20, 20, anchored=True)
        self.assertEqual(len(rolling), len(anchored))
        for f in rolling:
            self.assertEqual(f.train_end - f.train_start, 60 * DAY)
        self.assertEqual({f.train_start for f in anchored}, {T0})
        lengths = [(f.train_end - f.train_start).days for f in anchored]
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[-1], lengths[0])
        # Same train_end / validate / test windows in both modes.
        for r, a in zip(rolling, anchored):
            self.assertEqual((r.train_end, r.validate_end, r.test_end),
                             (a.train_end, a.validate_end, a.test_end))

    def test_fold_count_and_partial_window_dropped(self):
        # 100 days needed for fold 0; every step of 20 adds 20 -> 6 folds in 200.
        self.assertEqual(len(self.folds(200, 60, 20, 20, 20)), 6)
        # 199 days: the last fold's test window would be partial -> dropped.
        self.assertEqual(len(self.folds(199, 60, 20, 20, 20)), 5)
        self.assertEqual(self.folds(99, 60, 20, 20, 20), [])

    def test_invalid_geometry_rejected(self):
        for kw in ({"train": 0}, {"val": 0}, {"test": -1}, {"step": 0}):
            with self.assertRaises(ValueError):
                self.folds(**kw)

    def test_naive_start_rejected(self):
        with self.assertRaises(ValueError):
            wf.make_folds(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"),
                          train_days=10, validate_days=5, test_days=5,
                          step_days=5, anchored=False)

    def test_bars_for_window_warmup_is_past_only_and_end_exclusive(self):
        bars = make_synthetic_bars(10)
        start, end = T0 + 3 * DAY, T0 + 6 * DAY
        w = wf.bars_for_window(bars, start, end, warmup_days=1)
        self.assertGreaterEqual(w["timestamp_utc"].min(), start - DAY)
        self.assertLess(w["timestamp_utc"].min(), start)   # warmup is present
        self.assertLess(w["timestamp_utc"].max(), end)     # end is exclusive
        none = wf.bars_for_window(bars, start, end, warmup_days=0)
        self.assertGreaterEqual(none["timestamp_utc"].min(), start)

    def test_too_little_data_raises(self):
        bars = make_synthetic_bars(10)
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
            wf.run_walk_forward(
                bars, {"stop-ticks": [8]}, gs.resolve_fixed(), train_days=60,
                validate_days=20, test_days=20, step_days=20, anchored=False,
                criteria=wf.SelectionCriteria(), out_dir=td)


# --------------------------------------------------------------------------
# Selection logic (pure: metrics in, one candidate out)
# --------------------------------------------------------------------------

def _m(trades, net, sharpe=None, pf=1.0):
    return {"trades": trades, "net_pnl": float(net),
            "sharpe": float(net) / 100 if sharpe is None else float(sharpe),
            "max_drawdown": 0.0, "profit_factor": pf, "win_rate": 0.5,
            "avg_winner": 1.0, "avg_loser": -1.0}


def _cand(cid, train, validate, train_2x=None, validate_2x=None):
    c = {"combo_id": cid, "params": {"stop_ticks": float(cid)},
         "train": train, "validate": validate}
    if train_2x is not None:
        c["train_2x"], c["validate_2x"] = train_2x, validate_2x
    return c


class TestSelection(unittest.TestCase):
    crit = wf.SelectionCriteria(top_k=10, min_split_trades=10)

    def test_prefers_consistent_performer_over_max_return(self):
        """The proof fixture. Candidate 0 has by far the highest backtest
        return and the best train Sharpe, but its edge flips sign out of
        sample. Candidate 2 has a higher total return than the consistent
        candidate 1 but is lopsided (strong train, near-zero validate).
        Candidate 3 is consistent but dies under doubled slippage."""
        ok = _m(40, 400)  # any positive stress result
        cands = [
            # 0: max return, sign flips on validate -> fails (a)
            _cand(0, _m(50, 9000, 4.0), _m(40, -300, -0.5)),
            # 1: modest but consistent -> should win
            _cand(1, _m(50, 800, 1.2), _m(40, 700, 1.1), ok, ok),
            # 2: positive both, higher total return than 1, but lopsided
            _cand(2, _m(50, 5000, 3.0), _m(40, 200, 0.2), ok, ok),
            # 3: consistent, but net P&L turns negative with doubled slippage
            _cand(3, _m(50, 2000, 2.0), _m(40, 1900, 1.9),
                  _m(50, -25), _m(40, -10)),
        ]
        chosen, evaluated = wf.select_one(cands, self.crit)

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["combo_id"], 1)

        total_return = {c["combo_id"]: c["train"]["net_pnl"] + c["validate"]["net_pnl"]
                        for c in cands}
        self.assertEqual(max(total_return, key=total_return.get), 0)
        self.assertNotEqual(chosen["combo_id"], max(total_return, key=total_return.get))
        qualified = [e for e in evaluated if e["qualified"]]
        self.assertEqual({e["combo_id"] for e in qualified}, {1, 2})
        best_return_among_qualified = max(qualified, key=lambda e: total_return[e["combo_id"]])
        self.assertEqual(best_return_among_qualified["combo_id"], 2)
        self.assertNotEqual(chosen["combo_id"], best_return_among_qualified["combo_id"])

        flags = {e["combo_id"]: (e["a_consistent_sign"], e["b_survives_2x"])
                 for e in evaluated}
        self.assertEqual(flags[0], (False, None))
        self.assertEqual(flags[1], (True, True))
        self.assertEqual(flags[3], (True, False))

    def test_sign_flip_between_splits_is_rejected(self):
        c = _cand(0, _m(50, 500), _m(50, -1), _m(50, 400), _m(50, 400))
        chosen, ev = wf.select_one([c], self.crit)
        self.assertIsNone(chosen)
        self.assertFalse(ev[0]["a_consistent_sign"])

    def test_consistently_negative_is_not_a_consistent_edge(self):
        c = _cand(0, _m(50, -500), _m(50, -400), _m(50, -600), _m(50, -500))
        self.assertIsNone(wf.select_one([c], self.crit)[0])

    def test_few_trades_have_no_sign_of_edge(self):
        c = _cand(0, _m(2, 500), _m(2, 400), _m(2, 400), _m(2, 300))
        self.assertIsNone(wf.select_one([c], self.crit)[0])
        relaxed = wf.SelectionCriteria(min_split_trades=0)
        self.assertEqual(wf.select_one([c], relaxed)[0]["combo_id"], 0)

    def test_stress_not_run_means_not_qualified(self):
        c = _cand(0, _m(50, 500), _m(50, 400))  # no 2x metrics
        chosen, ev = wf.select_one([c], self.crit)
        self.assertIsNone(chosen)
        self.assertIsNone(ev[0]["b_survives_2x"])

    def test_stress_boundary_zero_is_not_surviving(self):
        c = _cand(0, _m(50, 500), _m(50, 400), _m(50, 0), _m(50, 10))
        self.assertIsNone(wf.select_one([c], self.crit)[0])

    def test_ties_break_deterministically_on_combo_id(self):
        ok = _m(40, 100)
        a = _cand(7, _m(50, 800, 1.0), _m(40, 700, 1.0), ok, ok)
        b = _cand(3, _m(50, 800, 1.0), _m(40, 700, 1.0), ok, ok)
        self.assertEqual(wf.select_one([a, b], self.crit)[0]["combo_id"], 3)
        self.assertEqual(wf.select_one([b, a], self.crit)[0]["combo_id"], 3)

    def test_ranking_uses_worst_split_not_best_or_mean(self):
        ok = _m(40, 100)
        lopsided = _cand(0, _m(50, 800, 5.0), _m(40, 700, 0.3), ok, ok)
        steady = _cand(1, _m(50, 800, 1.0), _m(40, 700, 0.9), ok, ok)
        self.assertEqual(wf.select_one([lopsided, steady], self.crit)[0]["combo_id"], 1)

    def test_backtest_return_is_not_an_allowed_selection_metric(self):
        with self.assertRaises(ValueError):
            wf.SelectionCriteria(selection_metric="net_pnl")

    def test_stress_must_be_a_real_stress(self):
        with self.assertRaises(ValueError):
            wf.SelectionCriteria(slippage_multiplier=1.0)

    def test_profit_factor_metric_with_inf(self):
        crit = wf.SelectionCriteria(selection_metric="profit_factor")
        ok = _m(40, 100)
        a = _cand(0, _m(50, 800, pf=float("inf")), _m(40, 700, pf=1.5), ok, ok)
        b = _cand(1, _m(50, 800, pf=2.0), _m(40, 700, pf=1.8), ok, ok)
        self.assertEqual(wf.select_one([a, b], crit)[0]["combo_id"], 1)

    def test_test_sample_gate_is_a_verdict_not_a_selector(self):
        self.assertEqual(wf.verdict_for_test(_m(29, 100), self.crit),
                         (False, "INCONCLUSIVE_LOW_TEST_SAMPLE"))
        self.assertEqual(wf.verdict_for_test(_m(30, 100), self.crit),
                         (True, "TEST_PROFITABLE"))
        self.assertEqual(wf.verdict_for_test(_m(30, -100), self.crit),
                         (True, "TEST_UNPROFITABLE"))
        # select_one has no test input at all.
        self.assertNotIn("test", wf.select_one.__code__.co_varnames[:2])


# --------------------------------------------------------------------------
# Orchestration with a stub runner (known outcomes, recorded bar windows)
# --------------------------------------------------------------------------

STUB_GRID = {"stop-ticks": [8, 10, 12, 14, 16]}

# stop_ticks -> {split: (trades, net_pnl, sharpe)}
STUB_TABLE = {
    8.0: {"train": (50, 9000, 4.0), "validate": (40, -300, -0.5), "test": (40, 5000, 3.0)},
    10.0: {"train": (50, 800, 1.2), "validate": (40, 700, 1.1), "test": (35, 420, 0.9)},
    12.0: {"train": (50, 5000, 3.0), "validate": (40, 200, 0.2), "test": (33, -50, -0.1)},
    14.0: {"train": (50, 100, 2.5), "validate": (40, 90, 2.0), "test": (31, 10, 0.1)},
    16.0: {"train": (3, 500, 6.0), "validate": (3, 400, 5.0), "test": (3, 9, 1.0)},
}


class StubRunner:
    """Stands in for grid_search.run_single. Doubled slippage costs $2.50 per
    trade. Records what every call was handed."""

    def __init__(self, fold, table=STUB_TABLE, out_dir=None):
        self.table = table
        self.out_dir = out_dir
        self.fold = fold
        self.split_of = {fold.train_start: "train",
                         fold.validate_start: "validate",
                         fold.test_start: "test"}
        self.calls = []

    def __call__(self, bars, params, fixed, slippage_mult=1.0, scored_from=None):
        split = self.split_of[scored_from]
        self.calls.append({
            "split": split, "stop": params["stop_ticks"], "slip": slippage_mult,
            "min_ts": bars["timestamp_utc"].min(),
            "max_ts": bars["timestamp_utc"].max(),
            "scored_from": scored_from,
            "criteria_file_existed": bool(
                self.out_dir and os.path.exists(
                    os.path.join(self.out_dir, "selection_criteria.json"))),
        })
        trades, net, sharpe = self.table[params["stop_ticks"]][split]
        if slippage_mult > 1.0:
            net = net - 2.5 * trades
        return {"trades": trades, "net_pnl": float(net), "sharpe": float(sharpe),
                "max_drawdown": 0.0, "profit_factor": 1.5, "win_rate": 0.5,
                "avg_winner": 1.0, "avg_loser": -1.0}


class TestOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_synthetic_bars(40)  # train 20 / validate 10 / test 10
        cls.geom = dict(train_days=20, validate_days=10, test_days=10,
                        step_days=10, anchored=False)
        cls.fold = wf.make_folds(T0, T0 + 40 * DAY, train_days=20, validate_days=10,
                                 test_days=10, step_days=10, anchored=False)[0]
        cls.crit = wf.SelectionCriteria(top_k=5, min_split_trades=10,
                                        min_test_trades=30, warmup_days=5)

    def _run(self, table=STUB_TABLE, crit=None):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        stub = StubRunner(self.fold, table, out_dir=td.name)
        with mock.patch.object(gs, "run_single", stub):
            report = wf.run_walk_forward(
                self.bars, STUB_GRID, gs.resolve_fixed(), criteria=crit or self.crit,
                out_dir=td.name, **self.geom)
        return report, stub, td.name

    def test_end_to_end_selects_consistent_candidate_and_reports_on_test(self):
        report, stub, out = self._run()
        self.assertEqual(len(report["folds"]), 1)
        fr = report["folds"][0]
        self.assertEqual(fr["selected"]["params"]["stop_ticks"], 10.0)
        self.assertEqual(fr["n_candidates"], 4)   # the 3-trade fluke is excluded
        self.assertEqual(fr["n_pass_a"], 3)
        self.assertEqual(fr["n_qualified"], 2)
        self.assertEqual(fr["verdict"], "TEST_PROFITABLE")
        self.assertEqual(fr["selected"]["test"]["trades"], 35)

        with open(os.path.join(out, "walkforward_summary.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stop_ticks"], "10.0")
        self.assertEqual(rows[0]["verdict"], "TEST_PROFITABLE")
        self.assertEqual(rows[0]["test_trades"], "35")

        # Train table is the FULL grid, unfiltered (incl. the fluke combination).
        with open(os.path.join(out, "fold_00_train_grid.csv"), newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 5)
        # Candidates: top-K by train Sharpe, best first.
        with open(os.path.join(out, "fold_00_candidates.csv"), newline="") as f:
            cand = list(csv.DictReader(f))
        self.assertEqual([c["stop_ticks"] for c in cand],
                         ["8.0", "12.0", "14.0", "10.0"])
        self.assertEqual([c["selected"] for c in cand],
                         ["False", "False", "False", "True"])

    def test_only_the_selected_candidate_is_ever_run_on_test(self):
        _, stub, _ = self._run()
        test_calls = [c for c in stub.calls if c["split"] == "test"]
        self.assertEqual({c["stop"] for c in test_calls}, {10.0})
        self.assertEqual(sorted(c["slip"] for c in test_calls), [1.0, 2.0])
        # Validate re-ran exactly the K candidates (no more, no fewer).
        val = [c for c in stub.calls if c["split"] == "validate" and c["slip"] == 1.0]
        self.assertEqual(sorted(c["stop"] for c in val), [8.0, 10.0, 12.0, 14.0])
        # The slippage stress ran only for candidates that already passed (a).
        stress = {c["stop"] for c in stub.calls if c["slip"] == 2.0 and c["split"] != "test"}
        self.assertEqual(stress, {10.0, 12.0, 14.0})

    def test_every_stage_only_sees_bars_before_its_window_end(self):
        """Structural no-lookahead: the data handed to each call ends strictly
        before the end of the window it is scored on, and warmup starts no
        earlier than warmup_days before the window start."""
        _, stub, _ = self._run()
        f, warm = self.fold, self.crit.warmup_days * DAY
        ends = {"train": f.train_end, "validate": f.validate_end, "test": f.test_end}
        starts = {"train": f.train_start, "validate": f.validate_start,
                  "test": f.test_start}
        self.assertTrue(stub.calls)
        for c in stub.calls:
            self.assertLess(c["max_ts"], ends[c["split"]], msg=str(c))
            self.assertGreaterEqual(c["min_ts"], starts[c["split"]] - warm, msg=str(c))
            self.assertEqual(c["scored_from"], starts[c["split"]])
        # Train and validate stages never touch a single test-window bar.
        for c in stub.calls:
            if c["split"] in ("train", "validate"):
                self.assertLess(c["max_ts"], f.test_start)
        # Train never touches validate or test bars.
        for c in stub.calls:
            if c["split"] == "train":
                self.assertLess(c["max_ts"], f.validate_start)

    def test_criteria_are_written_before_any_run(self):
        _, stub, out = self._run()
        self.assertTrue(all(c["criteria_file_existed"] for c in stub.calls))
        with open(os.path.join(out, "selection_criteria.json")) as f:
            reg = json.load(f)
        self.assertEqual(reg["criteria"]["min_test_trades"], 30)
        self.assertEqual(reg["criteria"]["selection_metric"], "sharpe")
        self.assertEqual(reg["grid_size"], 5)
        self.assertIn("not a criterion", reg["criteria"]["ranking"])

    def test_low_test_sample_is_inconclusive_and_never_swapped(self):
        table = {k: dict(v) for k, v in STUB_TABLE.items()}
        table[10.0] = {**table[10.0], "test": (12, 420, 0.9)}   # selected, thin test
        # The runners-up have plenty of test trades and look fine on test:
        table[12.0]["test"] = (60, 3000, 2.0)
        table[14.0]["test"] = (60, 2000, 2.0)
        report, stub, _ = self._run(table)
        fr = report["folds"][0]
        self.assertEqual(fr["selected"]["params"]["stop_ticks"], 10.0)
        self.assertEqual(fr["verdict"], "INCONCLUSIVE_LOW_TEST_SAMPLE")
        self.assertFalse(fr["test_sample_ok"])
        self.assertEqual({c["stop"] for c in stub.calls if c["split"] == "test"},
                         {10.0})

    def test_no_qualified_candidate_means_no_test_run(self):
        table = {k: {**v, "validate": (40, -100, -1.0)} for k, v in STUB_TABLE.items()}
        report, stub, out = self._run(table)
        fr = report["folds"][0]
        self.assertIsNone(fr["selected"])
        self.assertEqual(fr["verdict"], "NO_QUALIFIED_CANDIDATE")
        self.assertEqual([c for c in stub.calls if c["split"] == "test"], [])
        with open(os.path.join(out, "walkforward_summary.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["verdict"], "NO_QUALIFIED_CANDIDATE")
        self.assertEqual(rows[0]["selected_combo_id"], "")
        self.assertEqual(report["aggregate"]["n_selected"], 0)

    def test_top_k_limits_and_deduplicates_by_effective_strategy(self):
        rows = []
        # 6 rows: two inert-multiplier duplicates of the same V3 strategy.
        base = dict(gs.DEFAULTS, volume_filter="V3")
        for cid, mult in enumerate([1.5, 2.0, 2.5]):
            rows.append({"combo_id": cid, **dict(base, v1_mult=mult),
                         **_m(50, 100, sharpe=2.0)})
        for cid, stop in enumerate([6.0, 7.0, 9.0], start=3):
            rows.append({"combo_id": cid, **dict(base, stop_ticks=stop),
                         **_m(50, 100, sharpe=1.0)})
        cands = wf.top_k_candidates(rows, wf.SelectionCriteria(top_k=3))
        self.assertEqual([c["combo_id"] for c in cands], [0, 3, 4])  # 1 and 2 are dupes
        self.assertEqual(len(wf.top_k_candidates(rows, wf.SelectionCriteria(top_k=2))), 2)


# --------------------------------------------------------------------------
# Real-engine no-lookahead proofs
# --------------------------------------------------------------------------

def mutate_from(bars, cutoff):
    """Same bars before `cutoff`; valid but very different bars from it on."""
    m = bars.copy()
    mask = m["timestamp_utc"] >= cutoff
    for c in ("open", "high", "low", "close"):
        m.loc[mask, c] = m.loc[mask, c] + 25.0
    m.loc[mask, "volume"] = m.loc[mask, "volume"] * 3
    return m


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestNoLookaheadRealEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_synthetic_bars(14)
        cls.geom = dict(train_days=6, validate_days=3, test_days=3, step_days=3,
                        anchored=False)
        cls.crit = wf.SelectionCriteria(top_k=4, min_split_trades=1,
                                        min_test_trades=1, warmup_days=2)
        cls.grid = {"stop-ticks": [8, 12], "volume-filter": ["none", "V3"]}
        cls.fold = wf.make_folds(T0, T0 + 14 * DAY, train_days=6, validate_days=3,
                                 test_days=3, step_days=3, anchored=False)[0]
        cls.base = cls._run(cls.bars)

    @classmethod
    def _run(cls, bars):
        td = tempfile.mkdtemp()
        wf.run_walk_forward(bars, cls.grid, gs.resolve_fixed(), criteria=cls.crit,
                            out_dir=td, **cls.geom)
        return td

    def test_validate_and_test_bars_cannot_change_train_results(self):
        mutated = self._run(mutate_from(self.bars, self.fold.validate_start))
        a = os.path.join(self.base, "fold_00_train_grid.csv")
        b = os.path.join(mutated, "fold_00_train_grid.csv")
        with open(a, "rb") as fa, open(b, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())
        self.assertGreater(len(read_rows(a)), 0)
        self.assertTrue(any(int(r["trades"]) > 0 for r in read_rows(a)))  # not vacuous

    def test_test_bars_cannot_change_selection_inputs(self):
        mutated = self._run(mutate_from(self.bars, self.fold.test_start))
        for name in ("fold_00_train_grid.csv", "fold_00_candidates.csv"):
            with open(os.path.join(self.base, name), "rb") as fa, \
                    open(os.path.join(mutated, name), "rb") as fb:
                self.assertEqual(fa.read(), fb.read(), msg=name)
        self.assertGreater(len(read_rows(
            os.path.join(self.base, "fold_00_candidates.csv"))), 0)

    def test_check_is_not_vacuous_validate_bars_do_change_validate_metrics(self):
        """Mutating validate bars leaves train columns identical but changes the
        validate columns -- so the equalities above are genuinely informative."""
        mutated = self._run(mutate_from(self.bars, self.fold.validate_start))
        base = read_rows(os.path.join(self.base, "fold_00_candidates.csv"))
        mut = read_rows(os.path.join(mutated, "fold_00_candidates.csv"))
        self.assertEqual(len(base), len(mut))
        train_cols = [c for c in base[0] if c.startswith("train_")]
        val_cols = [c for c in base[0] if c.startswith("val_")]
        self.assertEqual([[r[c] for c in train_cols] for r in base],
                         [[r[c] for c in train_cols] for r in mut])
        self.assertNotEqual([[r[c] for c in val_cols] for r in base],
                            [[r[c] for c in val_cols] for r in mut])


# --------------------------------------------------------------------------
# CLI end to end
# --------------------------------------------------------------------------

class TestCli(unittest.TestCase):
    ARGS = ["--grid", '{"stop-ticks":[8,12],"volume-filter":["none","V3"]}',
            "--train-days", "6", "--validate-days", "3", "--test-days", "3",
            "--top-k", "3", "--min-split-trades", "1", "--min-test-trades", "1",
            "--warmup-days", "2"]

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.csv = os.path.join(cls.td.name, "bars.csv")
        write_csv(make_synthetic_bars(14), cls.csv)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def _main(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gs.main(argv)
        return rc, buf.getvalue()

    def test_same_data_grid_seed_gives_byte_identical_output_files(self):
        outs = []
        for i in range(2):
            out = os.path.join(self.td.name, f"run{i}")
            rc, _ = self._main([self.csv, *self.ARGS, "--seed", "5", "--out", out])
            self.assertEqual(rc, 0)
            outs.append(out)
        names = sorted(os.listdir(outs[0]))
        self.assertEqual(names, sorted(os.listdir(outs[1])))
        self.assertIn("walkforward_summary.csv", names)
        self.assertIn("fold_00_train_grid.csv", names)
        for n in names:
            with open(os.path.join(outs[0], n), "rb") as fa, \
                    open(os.path.join(outs[1], n), "rb") as fb:
                self.assertEqual(fa.read(), fb.read(), msg=n)

    def test_grid_only_writes_the_full_table(self):
        out = os.path.join(self.td.name, "gridonly")
        rc, _ = self._main([self.csv, "--grid-only", "--out", out, "--grid",
                            '{"stop-ticks":[8,12],"ema-ticks":[1,2],"volume-filter":["none","V1"]}'])
        self.assertEqual(rc, 0)
        rows = read_rows(os.path.join(out, "grid_results.csv"))
        self.assertEqual(len(rows), 8)  # 2*2*2, nothing filtered
        with open(os.path.join(out, "run_manifest.json")) as f:
            self.assertIn("in-sample", json.load(f)["note"])

    def test_unsupported_grid_key_exits_with_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            gs.main([self.csv, "--grid", '{"atr-stop-mult":[1,2]}',
                     "--out", os.path.join(self.td.name, "x")])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("ATR", err.getvalue())

    def test_not_enough_data_exits_with_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            gs.main([self.csv, "--grid", '{"stop-ticks":[8]}', "--train-days", "60",
                     "--out", os.path.join(self.td.name, "y")])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("needs train+validate+test", err.getvalue())

    def test_net_pnl_is_not_a_selectable_metric_on_the_cli(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            gs.main([self.csv, "--selection-metric", "net_pnl"])


if __name__ == "__main__":
    unittest.main()
