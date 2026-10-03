"""Proof 3: correct P&L accounting with costs — hand-computed expectations.

Scenario (tests.conftest.breakout_retest_rows, MES @ $1.25/tick):
  signal bar close = 110.50 -> stop   = 110.50 -  8*0.25 = 108.50
                            -> target = 110.50 + 16*0.25 = 114.50
  entry = next bar open 109.50 + 1 tick slip = 109.75
  exit  = target 114.50 - 1 tick slip        = 114.25
  gross = (114.25 - 109.75)/0.25 ticks * $1.25/tick = 18 * 1.25 = $22.50
  commission = $1.25 * 2 sides = $2.50
  net = $20.00
"""
import unittest

from backtest.metrics import compute_metrics
from tests.conftest import make_bars, breakout_retest_rows, build_engine, make_cfg


class TestPnL(unittest.TestCase):
    def test_pnl_with_costs_exact(self):
        trades = build_engine().run(make_bars(breakout_retest_rows()))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.exit_reason, "target")
        self.assertEqual(t.exit_price, 114.25)
        self.assertAlmostEqual(t.gross_pnl, 22.50)
        self.assertAlmostEqual(t.net_pnl, 20.00)

    def test_metrics_on_known_trade_set(self):
        trades = build_engine().run(make_bars(breakout_retest_rows()))
        m = compute_metrics(trades)
        self.assertEqual(m["trades"], 1)
        self.assertAlmostEqual(m["net_pnl"], 20.00)
        self.assertEqual(m["win_rate"], 1.0)
        self.assertAlmostEqual(m["avg_winner"], 20.00)
        self.assertEqual(m["avg_loser"], 0.0)
        self.assertEqual(m["max_drawdown"], 0.0)
        # Single winning trade, no losses: profit factor is infinite.
        self.assertEqual(m["profit_factor"], float("inf"))

    def test_stop_loss_trade_accounts_costs(self):
        """Same setup, price falls through the stop after entry.
        entry 109.75; stop 108.50 -> exit 108.50 - 0.25 slip = 108.25
        gross = (108.25-109.75)/0.25 = -6 ticks * $1.25 = -$7.50; net = -$10.00."""
        rows = breakout_retest_rows(target_bar_high=110.0, target_bar_low=100.0)
        trades = build_engine().run(make_bars(rows))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.exit_reason, "stop_loss")
        self.assertAlmostEqual(t.gross_pnl, -7.50)
        self.assertAlmostEqual(t.net_pnl, -10.00)

    def test_stop_fills_first_when_bar_covers_both(self):
        """Conservative tie-break: bar range covers stop AND target -> stop."""
        rows = breakout_retest_rows(target_bar_high=120.0, target_bar_low=100.0)
        trades = build_engine().run(make_bars(rows))
        self.assertEqual(trades[0].exit_reason, "stop_loss")

    def test_max_trades_per_day_guard(self):
        cfg = make_cfg(max_trades_per_day=0)
        trades = build_engine(cfg=cfg).run(make_bars(breakout_retest_rows()))
        self.assertEqual(trades, [])

    def test_session_window_guard(self):
        # Scenario entries happen at 00:19 -> blocked by a 12:00-20:00 window.
        cfg = make_cfg(session_start="12:00", session_end="20:00")
        trades = build_engine(cfg=cfg).run(make_bars(breakout_retest_rows()))
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
