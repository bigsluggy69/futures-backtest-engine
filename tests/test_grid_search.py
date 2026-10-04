"""Grid search layer: full cartesian output, rejected parameters, determinism,
and parity with the real run_backtest.py CLI."""
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import grid_search as gs
from tests.synth import make_synthetic_bars, write_csv


def _fake_metrics(trades=0, net=0.0):
    return {"trades": trades, "net_pnl": net, "sharpe": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "avg_winner": 0.0,
            "avg_loser": 0.0}


class TestFullCartesian(unittest.TestCase):
    GRID = {
        "ema-ticks": [1, 2, 4],
        "stop-ticks": [6, 8],
        "target-ticks": [12, 16, 24],
        "volume-filter": ["none", "V1", "V3"],
        "session-start": ["13:30", "14:00"],
        "max-trades-per-day": [1, 2],
    }

    def test_grid_emits_full_cartesian_set_without_filtering(self):
        calls = []

        def fake(bars, params, fixed, slippage_mult=1.0, scored_from=None):
            calls.append(dict(params))
            # Mostly zero-trade / losing results: none may be dropped.
            return _fake_metrics(trades=0, net=-1.0)

        bars = make_synthetic_bars(2)
        with mock.patch.object(gs, "run_single", fake):
            rows = gs.run_grid(bars, self.GRID)

        expected = list(itertools.product(
            [1.0, 2.0, 4.0], [6.0, 8.0], [12.0, 16.0, 24.0],
            ["none", "V1", "V3"], ["13:30", "14:00"], [1, 2]))
        self.assertEqual(len(expected), 3 * 2 * 3 * 3 * 2 * 2)
        self.assertEqual(len(rows), len(expected))
        self.assertEqual(len(calls), len(expected))  # every combo actually ran

        got = {(r["ema_ticks"], r["stop_ticks"], r["target_ticks"],
                r["volume_filter"], r["session_start"], r["max_trades_per_day"])
               for r in rows}
        self.assertEqual(got, set(expected))
        self.assertEqual(len(got), len(rows))  # no duplicates either
        self.assertEqual([r["combo_id"] for r in rows], list(range(len(rows))))
        self.assertEqual(gs.grid_size(self.GRID), len(expected))

        # Params outside the grid are held at their defaults in every row.
        for r in rows:
            self.assertEqual(r["expiry_min"], gs.DEFAULTS["expiry_min"])
            self.assertEqual(r["cooldown_min"], gs.DEFAULTS["cooldown_min"])

    def test_csv_contains_every_row_including_zero_trade_combos(self):
        bars = make_synthetic_bars(6)
        grid = {"volume-filter": ["none", "V1"], "stop-ticks": [8, 12]}
        rows = gs.run_grid(bars, grid)
        self.assertEqual(len(rows), 4)
        # V1 abstains on 6 days of history -> zero trades, still present.
        self.assertTrue(any(r["volume_filter"] == "V1" and r["trades"] == 0
                            for r in rows))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g.csv")
            gs.write_results_csv(rows, path)
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        self.assertEqual(len(lines), 1 + 4)
        self.assertEqual(lines[0].split(","), list(gs.RESULT_COLUMNS))

    def test_grid_key_order_does_not_change_the_table(self):
        a = {"stop-ticks": [8, 12], "ema-ticks": [1, 2]}
        b = {"ema-ticks": [1, 2], "stop-ticks": [8, 12]}
        self.assertEqual([p for _, p in gs.iter_grid(a)],
                         [p for _, p in gs.iter_grid(b)])

    def test_flag_spellings_are_equivalent(self):
        a = gs.canonicalize_grid({"--ema-ticks": [1, 2]})
        b = gs.canonicalize_grid({"ema_ticks": [1, 2]})
        self.assertEqual(a, b)


class TestGridValidation(unittest.TestCase):
    def test_atr_and_trailing_stops_are_rejected(self):
        for key in ("atr-stop-mult", "atr-period", "trail-ticks", "trailing-stop"):
            with self.assertRaises(ValueError) as cm:
                gs.canonicalize_grid({key: [1, 2]})
            self.assertIn("no ATR-based or trailing stops", str(cm.exception))

    def test_fixed_engine_settings_are_not_grid_dimensions(self):
        for key in ("max-holding-min", "daily-loss-cap", "max-spread-ticks",
                    "contracts", "seed", "tick-value"):
            with self.assertRaises(ValueError) as cm:
                gs.canonicalize_grid({key: [1, 2]})
            self.assertIn("fixed engine setting", str(cm.exception))

    def test_unknown_key_and_bad_values_rejected(self):
        bad = [
            {"nonsense": [1]},
            {"ema-ticks": 2},                      # scalar, not a list
            {"ema-ticks": []},
            {"ema-ticks": [1, 1.0]},               # duplicate after casting
            {"expiry-min": [30.5]},                # non-integer
            {"volume-filter": ["V9"]},
            {"session-start": ["1330"]},
            {"session-end": ["25:00"]},
            {"stop-ticks": [float("nan")]},
            {"stop-ticks": [True]},
            {},
        ]
        for grid in bad:
            with self.assertRaises(ValueError, msg=str(grid)):
                gs.canonicalize_grid(grid)

    def test_default_grid_is_valid(self):
        self.assertEqual(gs.grid_size(gs.DEFAULT_GRID), 96)

    def test_inert_multipliers_share_an_effective_key(self):
        base = dict(gs.DEFAULTS)
        a = {**base, "volume_filter": "V3", "v1_mult": 1.5}
        b = {**base, "volume_filter": "V3", "v1_mult": 3.0}
        self.assertEqual(gs.effective_key(a), gs.effective_key(b))
        c = {**base, "volume_filter": "V1", "v1_mult": 1.5}
        d = {**base, "volume_filter": "V1", "v1_mult": 3.0}
        self.assertNotEqual(gs.effective_key(c), gs.effective_key(d))
        grid = {"volume-filter": ["none", "V3"], "v1-mult": [1.5, 3.0]}
        self.assertEqual(gs.grid_size(grid), 4)
        self.assertEqual(gs.effective_grid_size(grid), 2)


class TestGridDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = make_synthetic_bars(6)
        cls.grid = {"ema-ticks": [1, 2], "stop-ticks": [8, 12],
                    "volume-filter": ["none", "V3"]}

    def _csv_bytes(self, rows):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "g.csv")
            gs.write_results_csv(rows, p)
            with open(p, "rb") as f:
                return f.read()

    def test_same_data_grid_seed_gives_byte_identical_csv(self):
        a = self._csv_bytes(gs.run_grid(self.bars, self.grid, {"seed": 7}))
        b = self._csv_bytes(gs.run_grid(self.bars.copy(), dict(self.grid), {"seed": 7}))
        self.assertEqual(a, b)

    def test_worker_count_does_not_change_the_bytes(self):
        seq = self._csv_bytes(gs.run_grid(self.bars, self.grid, workers=1))
        par = self._csv_bytes(gs.run_grid(self.bars, self.grid, workers=2))
        self.assertEqual(seq, par)

    def test_numpy_scalars_never_reach_the_csv(self):
        text = self._csv_bytes(gs.run_grid(self.bars, self.grid)).decode()
        self.assertNotIn("np.", text)
        self.assertNotIn("numpy", text)


class TestParityWithCli(unittest.TestCase):
    """run_single must build exactly what run_backtest.py builds."""

    def test_metrics_match_run_backtest_cli(self):
        bars = make_synthetic_bars(8)
        params = {**gs.DEFAULTS, "ema_ticks": 3.0, "stop_ticks": 6.0,
                  "target_ticks": 12.0, "volume_filter": "V3",
                  "cooldown_min": 10, "max_trades_per_day": 3,
                  "session_start": "14:00", "session_end": "19:00"}
        fixed = gs.resolve_fixed({"max_holding_min": 90, "daily_loss_cap": 300.0})
        mine = gs.run_single(bars, params, fixed)
        self.assertGreater(mine["trades"], 0)  # parity must not be vacuous

        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, "bars.csv")
            write_csv(bars, csv)
            out = subprocess.run(
                [sys.executable, "run_backtest.py", csv, "--instrument", "MES",
                 "--ema-ticks", "3", "--stop-ticks", "6", "--target-ticks", "12",
                 "--volume-filter", "V3", "--cooldown-min", "10",
                 "--max-trades-per-day", "3", "--session-start", "14:00",
                 "--session-end", "19:00", "--max-holding-min", "90",
                 "--daily-loss-cap", "300"],
                capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, msg=out.stderr)
        cli = json.loads(out.stdout)
        self.assertEqual(set(cli), set(gs.METRIC_NAMES))
        for k in gs.METRIC_NAMES:
            self.assertAlmostEqual(mine[k], cli[k], places=6, msg=k)

    def test_doubled_slippage_costs_more(self):
        bars = make_synthetic_bars(8)
        fixed = gs.resolve_fixed()
        base = gs.run_single(bars, dict(gs.DEFAULTS), fixed)
        stress = gs.run_single(bars, dict(gs.DEFAULTS), fixed, slippage_mult=2.0)
        self.assertGreater(base["trades"], 0)
        # Same trades, 1 extra tick/side adverse on every round trip (2 ticks
        # x $1.25) -- never better than the base case.
        self.assertLess(stress["net_pnl"], base["net_pnl"])


if __name__ == "__main__":
    unittest.main()
