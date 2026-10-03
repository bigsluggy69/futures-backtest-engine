"""Proof 1: no lookahead — signals use closed bars only; entries fill on the
next bar's OPEN, never on the signal bar itself."""
import unittest

import pandas as pd

from tests.conftest import (make_bars, breakout_retest_rows, build_engine,
                            SPEC, make_cfg)


class TestNoLookahead(unittest.TestCase):
    def test_entry_fills_at_next_bar_open(self):
        bars = make_bars(breakout_retest_rows())
        trades = build_engine().run(bars)

        self.assertEqual(len(trades), 1)
        t = trades[0]
        # The retest signal is produced when bar 00:18 closes the 00:16-00:18
        # 2m bucket. The fill must be the NEXT bar (00:19) at its OPEN (109.50)
        # plus 1 tick of adverse slippage (MES tick_size 0.25).
        self.assertEqual(t.entry_time, pd.Timestamp("2024-01-02 00:19", tz="UTC"))
        self.assertEqual(t.entry_price, 109.75)
        # Fill must be strictly after the signal bar's close time.
        self.assertGreater(t.entry_time,
                           pd.Timestamp("2024-01-02 00:18", tz="UTC"))

    def test_entry_price_ignores_entry_bar_close_and_range(self):
        """Mutating the entry bar's high/low/close must not change the fill
        price: only its open (known at fill time) may be used."""
        baseline = make_bars(breakout_retest_rows())
        mutated = make_bars(breakout_retest_rows(
            entry_bar_high=999.0, entry_bar_low=1.0, entry_bar_close=777.0))

        t_base = build_engine().run(baseline)[0]
        t_mut = build_engine().run(mutated)[0]

        self.assertEqual(t_mut.entry_price, t_base.entry_price)
        self.assertEqual(t_mut.entry_price, 109.75)

    def test_signal_bar_close_price_not_used_as_fill(self):
        """If the engine were filling at the signal bar's close (classic
        lookahead), the entry would be 110.5 (+slip). It must not be."""
        t = build_engine().run(make_bars(breakout_retest_rows()))[0]
        self.assertNotEqual(t.entry_price, 110.75)
        self.assertEqual(t.entry_price, 109.75)

    def test_invalidated_breakout_never_trades(self):
        """A 15m CLOSE back inside the level must invalidate the setup; a later
        touch of the level must then be ignored (state is back to IDLE)."""
        rows = [
            ("2024-01-01 00:00", 100, 110, 100, 105, 10),
            ("2024-01-01 00:01", 100, 105, 90, 100, 10),
        ]
        # 15m bar closes at 111 > PDH 110 -> breakout @00:15.
        for m in range(15):
            rows.append((f"2024-01-02 00:{m:02d}", 111, 111, 111, 111, 10))
        # Next 15m bar: wicks to 112 but never touches 110, CLOSES at 109
        # (inside) -> INVALIDATION confirmed when the bar completes at 00:30.
        for m in range(15, 30):
            rows.append((f"2024-01-02 00:{m:02d}", 110.5, 112, 110.5, 109, 10))
        rows += [
            ("2024-01-02 00:30", 109, 109, 109, 109, 10),
            # Touch of the level AFTER invalidation -> must produce no trade.
            ("2024-01-02 00:31", 109, 110.5, 110.0, 110.2, 10),
            ("2024-01-02 00:32", 110.2, 110.5, 110.0, 110.2, 10),
            ("2024-01-02 00:33", 110.2, 110.3, 110.1, 110.2, 10),
        ]
        trades = build_engine().run(make_bars(rows))
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
