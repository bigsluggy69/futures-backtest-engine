"""Tests for the invalidation-bar opposite-breakout edge case.

A 15m bar that invalidates one direction AND closes beyond the opposite
prior-day level must start the opposite setup on that SAME bar, instead of
missing it until the next 15m close.
"""
import unittest

import pandas as pd

from backtest.costs import INSTRUMENTS
from backtest.strategies.pdh_pdl_breakout import PdhPdlBreakoutRetest

SPEC = INSTRUMENTS["MES"]


def make_strat(**kw):
    d = dict(ema_proximity_ticks=0.0, expiry_min=60,
             stop_ticks=8.0, target_ticks=16.0)
    d.update(kw)
    return PdhPdlBreakoutRetest(spec=SPEC, **d)


def feed(strat, rows):
    for t, o, h, l, c, v in rows:
        ts = pd.Timestamp(t, tz="UTC")
        strat.on_bar_close(ts, o, h, l, c, v)


def b15(o, h, l, c, v=100.0):
    return {"bucket": pd.Timestamp("2024-01-02 00:15", tz="UTC"),
            "o": o, "h": h, "l": l, "c": c, "v": v}


TS = pd.Timestamp("2024-01-02 00:30", tz="UTC")


class TestInvalidationOppositeBreakout(unittest.TestCase):
    def _long_watch(self, **kw):
        s = make_strat(**kw)
        feed(s, [
            ("2024-01-01 00:00", 100, 110, 100, 105, 10),
            ("2024-01-01 00:01", 100, 105, 90, 100, 10),
            ("2024-01-02 00:00", 111, 111, 111, 111, 10),  # completes day 1
        ])
        self.assertEqual((s.pdh, s.pdl), (110.0, 90.0))
        s.state = "RETEST_WATCH"
        s.level = 110.0
        s.direction = +1
        s.breakout_time = pd.Timestamp("2024-01-02 00:15", tz="UTC")
        s._breakout_vol = 150.0
        return s

    def test_invalidation_with_opposite_break_starts_short(self):
        s = self._long_watch()
        # 15m closes at 89: invalidates the long (89 < 110) AND breaks PDL.
        self.assertIsNone(s._on_15m_close(b15(95, 96, 88, 89), TS))
        self.assertEqual(s.state, "RETEST_WATCH")
        self.assertEqual(s.direction, -1)
        self.assertEqual(s.level, 90.0)
        self.assertEqual(s._breakout_vol, 100.0)
        self.assertEqual(s._pullback_2m, [])

    def test_plain_invalidation_goes_idle(self):
        s = self._long_watch()
        # 15m closes at 105: inside the level, no opposite break.
        self.assertIsNone(s._on_15m_close(b15(108, 109, 104, 105), TS))
        self.assertEqual(s.state, "IDLE")
        self.assertEqual(s.direction, 0)
        self.assertIsNone(s.level)

    def test_invalidation_at_exact_pdl_is_not_opposite_break(self):
        s = self._long_watch()
        # Close exactly at PDL: breakout needs a CLOSE BEYOND the level.
        self.assertIsNone(s._on_15m_close(b15(95, 96, 89, 90), TS))
        self.assertEqual(s.state, "IDLE")

    def test_short_watch_mirror(self):
        s = self._long_watch()
        s.state = "RETEST_WATCH"
        s.level = 90.0
        s.direction = -1
        # 15m closes at 112: invalidates the short AND breaks PDH.
        self.assertIsNone(s._on_15m_close(b15(108, 113, 107, 112), TS))
        self.assertEqual(s.state, "RETEST_WATCH")
        self.assertEqual(s.direction, +1)
        self.assertEqual(s.level, 110.0)

    def test_opposite_break_respects_v1_gate(self):
        # With V1 and no slot history, the opposite breakout must abstain
        # exactly like a fresh breakout would.
        s = self._long_watch(volume_filter="V1")
        self.assertIsNone(s._on_15m_close(b15(95, 96, 88, 89), TS))
        self.assertEqual(s.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
