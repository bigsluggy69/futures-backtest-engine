"""Tests for spread-series hardening and the --tick-value CLI override."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from backtest.engine import Engine
from tests.conftest import make_bars, breakout_retest_rows, build_engine, make_cfg


def make_cfg_spread(**overrides):
    d = dict(max_trades_per_day=3, cooldown_after_loss_min=0,
             session_start="00:00", session_end="23:59",
             max_holding_min=120, seed=42)
    d.update(overrides)
    from backtest.engine import EngineConfig
    return EngineConfig(**d)


class TestSpreadSeries(unittest.TestCase):
    def test_gappy_series_does_not_raise(self):
        # Previously .loc[ts] raised KeyError on any missing timestamp.
        eng = build_engine(cfg=make_cfg_spread())
        idx = [pd.Timestamp("2024-01-02 00:10", tz="UTC"),
               pd.Timestamp("2024-01-02 00:19", tz="UTC")]
        eng.set_spread_series(pd.Series([1.0, 1.0], index=idx))
        trades = eng.run(make_bars(breakout_retest_rows()))
        self.assertEqual(len(trades), 1)  # spread 1.0 <= max 2 -> allowed

    def test_leading_gap_falls_back_to_default(self):
        eng = build_engine(cfg=make_cfg_spread())
        # Series starts after the entry bar: ffill has nothing to carry, so
        # the constant default (1 tick) applies — no NaN, no exception.
        idx = [pd.Timestamp("2024-01-02 00:25", tz="UTC")]
        eng.set_spread_series(pd.Series([1.0], index=idx))
        trades = eng.run(make_bars(breakout_retest_rows()))
        self.assertEqual(len(trades), 1)

    def test_wide_spread_blocks_entry(self):
        eng = build_engine(cfg=make_cfg_spread(max_spread_ticks=2))
        idx = [pd.Timestamp("2024-01-02 00:19", tz="UTC")]
        eng.set_spread_series(pd.Series([5.0], index=idx))
        trades = eng.run(make_bars(breakout_retest_rows()))
        self.assertEqual(trades, [])  # 5 ticks > max 2 -> signal dropped

    def test_ffill_carries_last_known_spread(self):
        eng = build_engine(cfg=make_cfg_spread(max_spread_ticks=2))
        # Wide spread posted at 00:10 carries forward to the 00:19 entry bar.
        idx = [pd.Timestamp("2024-01-02 00:10", tz="UTC")]
        eng.set_spread_series(pd.Series([5.0], index=idx))
        trades = eng.run(make_bars(breakout_retest_rows()))
        self.assertEqual(trades, [])


class TestTickValueCLI(unittest.TestCase):
    def _run_cli(self, *extra):
        rows = breakout_retest_rows()
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, "bars.csv")
            pd.DataFrame(
                [(pd.Timestamp(t, tz="UTC"), o, h, l, c, v)
                 for t, o, h, l, c, v in rows],
                columns=["timestamp_utc", "open", "high", "low", "close",
                         "volume"],
            ).to_csv(csv, index=False)
            cmd = [sys.executable, "run_backtest.py", csv,
                   "--instrument", "MES",
                   "--ema-ticks", "0",
                   "--session-start", "00:00", "--session-end", "23:59",
                   "--max-trades-per-day", "3", "--cooldown-min", "0"] + list(extra)
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, msg=out.stderr)
        return json.loads(out.stdout)

    def test_default_tick_value(self):
        m = self._run_cli()
        self.assertEqual(m["trades"], 1)
        self.assertAlmostEqual(m["net_pnl"], 20.00)  # 18 x $1.25 - $2.50

    def test_tick_value_override(self):
        m = self._run_cli("--tick-value", "5.0")
        self.assertEqual(m["trades"], 1)
        # Same trade at $5/tick: 18 x $5 - $2.50 = $87.50.
        self.assertAlmostEqual(m["net_pnl"], 87.50)

    def test_volume_filter_flag(self):
        m = self._run_cli("--volume-filter", "V3")
        # Short synthetic history -> V3 abstains -> no trades, no crash.
        self.assertEqual(m["trades"], 0)


if __name__ == "__main__":
    unittest.main()
