"""Tests for the V1-V6 volume confirmation filter.

History is abbreviated via direct deque seeding (slot_median_days / lookbacks
overridden small); comments note what each seed stands in for. The gate logic
under test is identical to production.
"""
import unittest
from collections import deque
from datetime import time as dtime

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
    """Feed 1m rows; returns [(ts, signal_or_None)]."""
    out = []
    for t, o, h, l, c, v in rows:
        ts = pd.Timestamp(t, tz="UTC")
        out.append((ts, strat.on_bar_close(ts, o, h, l, c, v)))
    return out


DAY1 = [
    ("2024-01-01 00:00", 100, 110, 100, 105, 10),  # PDH = 110
    ("2024-01-01 00:01", 100, 105, 90, 100, 10),   # PDL = 90
]


def breakout_15_rows(day="2024-01-02", price=111.0, vol=10.0, n15=15):
    """Fifteen 1m bars forming the breakout 15m bucket + the bar that closes it.

    The 15m bar (bucket 00:00) completes when the 00:15 bar arrives; its close
    (111) exceeds PDH (110) -> breakout evaluated at 00:15.
    """
    rows = [(f"{day} 00:{m:02d}", price, price, price, price, vol)
            for m in range(n15)]
    rows.append((f"{day} 00:15", price, price, price, price, vol))
    return rows


def retest_rows(vols=(10.0, 10.0, 10.0), day="2024-01-02",
                lo1=110.0, hi1=111.0, lo2=110.4, hi2=110.6):
    """2m signal bucket 00:16 (bars 00:16, 00:17) + the bar closing it (00:18).

    Bar 00:16/00:17 lows touch the 110 level -> retest candidate.
    """
    v0, v1, v2 = vols
    return [
        (f"{day} 00:16", 110.5, hi1, lo1, 110.5, v0),
        (f"{day} 00:17", 110.5, hi1, lo1, 110.5, v1),
        (f"{day} 00:18", 110.5, hi2, lo2, 110.5, v2),
    ]


class TestV1BreakoutGate(unittest.TestCase):
    def test_v1_blocks_thin_breakout(self):
        s = make_strat(volume_filter="V1", slot_median_days=3)
        feed(s, DAY1)
        # Three prior days of the 00:00 slot at volume 100 -> median 100.
        s._vol15_by_slot[dtime(0, 0)] = deque([100.0] * 3, maxlen=3)
        feed(s, breakout_15_rows(vol=9.0))   # 15m volume = 135 < 1.5*100
        self.assertEqual(s.state, "IDLE")

    def test_v1_allows_heavy_breakout(self):
        s = make_strat(volume_filter="V1", slot_median_days=3)
        feed(s, DAY1)
        s._vol15_by_slot[dtime(0, 0)] = deque([100.0] * 3, maxlen=3)
        feed(s, breakout_15_rows(vol=20.0))  # 15m volume = 300 >= 150
        self.assertEqual(s.state, "RETEST_WATCH")
        self.assertEqual(s.direction, 1)
        self.assertEqual(s.level, 110.0)

    def test_v1_abstains_without_history(self):
        s = make_strat(volume_filter="V1")  # slot_median_days=20, no history
        feed(s, DAY1)
        feed(s, breakout_15_rows())
        self.assertEqual(s.state, "IDLE")


class TestV2BreakoutGate(unittest.TestCase):
    def _seed_after_day1(self, s, seed):
        # Feed day-1 rows + the first day-2 bar (completes day-1's 15m bar),
        # then overwrite the seeded history so the math is exact.
        feed(s, DAY1 + [("2024-01-02 00:00", 111, 111, 111, 111, 10.0)])
        s._recent15_vols = deque(seed, maxlen=s.vol_lookback_15m)

    def test_v2_blocks_thin_breakout(self):
        s = make_strat(volume_filter="V2", vol_lookback_15m=10)
        self._seed_after_day1(s, [100.0] * 10)
        rest = [(f"2024-01-02 00:{m:02d}", 111, 111, 111, 111, 5.0)
                for m in range(1, 15)]
        rest.append(("2024-01-02 00:15", 111, 111, 111, 111, 5.0))
        feed(s, rest)  # 15m volume = 75 < 2.0 * 100
        self.assertEqual(s.state, "IDLE")

    def test_v2_check_happens_before_record(self):
        """The breakout bar must be evaluated against history EXCLUDING itself.

        Seeded mean = 100 -> gate needs >= 200. Bar volume 206 passes.
        If the bar were recorded first, mean would be 110.6 -> need >= 221.2
        -> the same bar would be blocked. This test discriminates the order.
        """
        s = make_strat(volume_filter="V2", vol_lookback_15m=10)
        self._seed_after_day1(s, [100.0] * 10)
        rest = [(f"2024-01-02 00:{m:02d}", 111, 111, 111, 111, 14.0)
                for m in range(1, 15)]
        rest.append(("2024-01-02 00:15", 111, 111, 111, 111, 14.0))
        feed(s, rest)  # 15m volume = 10 + 14*14 = 206
        self.assertEqual(s.state, "RETEST_WATCH")
        # ...and the bar was recorded afterwards (history grew by one).
        self.assertEqual(list(s._recent15_vols), [100.0] * 9 + [206.0])


class TestV3EntryGate(unittest.TestCase):
    def _setup(self, **kw):
        s = make_strat(volume_filter="V3", vol_lookback_2m=5, **kw)
        feed(s, DAY1 + breakout_15_rows())  # V3 does not gate the breakout
        self.assertEqual(s.state, "RETEST_WATCH")
        # Five prior 2m bars at volume 10 (abbreviated history).
        s._recent2_vols = deque([10.0] * 5, maxlen=5)
        s._recent2_ranges = deque([1.0] * 5, maxlen=5)
        return s

    def test_v3_allows_expansion(self):
        s = self._setup()
        sigs = feed(s, retest_rows(vols=(25.0, 25.0, 10.0)))
        sig = [sg for _, sg in sigs if sg is not None]
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0]["direction"], 1)

    def test_v3_blocks_and_stays_watching(self):
        s = self._setup()
        sigs = feed(s, retest_rows(vols=(5.0, 5.0, 10.0)))
        self.assertTrue(all(sg is None for _, sg in sigs))
        # Rejection does NOT cancel the setup.
        self.assertEqual(s.state, "RETEST_WATCH")
        # A later loud retest still triggers.
        sigs = feed(s, [
            ("2024-01-02 00:19", 110.5, 111.0, 109.5, 110.5, 30.0),
            ("2024-01-02 00:20", 110.5, 111.0, 109.5, 110.5, 30.0),
            ("2024-01-02 00:21", 110.5, 110.6, 110.4, 110.5, 10.0),
        ])
        sig = [sg for _, sg in sigs if sg is not None]
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0]["direction"], 1)


class TestV4EntryGate(unittest.TestCase):
    def _setup(self, breakout_vol=20.0, pullback_vol=20.0):
        s = make_strat(volume_filter="V4")
        feed(s, DAY1)
        rows = [(f"2024-01-02 00:{m:02d}", 111, 111, 111, 111, breakout_vol)
                for m in range(14)]
        rows.append(("2024-01-02 00:14", 111, 111, 111, 111, breakout_vol))
        rows.append(("2024-01-02 00:15", 111, 111, 111, 111, pullback_vol))
        feed(s, rows)
        self.assertEqual(s.state, "RETEST_WATCH")
        return s

    def test_v4_quiet_pullback_allows(self):
        # Breakout 15m: 15 x 20 = 300 -> 20/min. Pullback 2m: (20+1)/2 per min
        # = 10.5 < 0.7 * 20 = 14 -> pass.
        s = self._setup(breakout_vol=20.0, pullback_vol=1.0)
        sigs = feed(s, retest_rows())
        sig = [sg for _, sg in sigs if sg is not None]
        self.assertEqual(len(sig), 1)

    def test_v4_loud_pullback_blocks(self):
        s = self._setup(breakout_vol=20.0, pullback_vol=20.0)
        sigs = feed(s, retest_rows())
        self.assertTrue(all(sg is None for _, sg in sigs))
        self.assertEqual(s.state, "RETEST_WATCH")


class TestV5EntryGate(unittest.TestCase):
    def _setup(self, **kw):
        s = make_strat(volume_filter="V5", slot_median_days=3, **kw)
        feed(s, DAY1 + breakout_15_rows())  # V5 does not gate the breakout
        self.assertEqual(s.state, "RETEST_WATCH")
        # Three prior days of the 00:16 2m slot at volume 10.
        s._vol2_by_slot[dtime(0, 16)] = deque([10.0] * 3, maxlen=3)
        return s

    def test_v5_allows(self):
        s = self._setup()
        sigs = feed(s, retest_rows(vols=(20.0, 20.0, 10.0)))  # 2m vol 40
        sig = [sg for _, sg in sigs if sg is not None]
        self.assertEqual(len(sig), 1)

    def test_v5_blocks(self):
        s = self._setup()
        sigs = feed(s, retest_rows(vols=(5.0, 5.0, 10.0)))    # 2m vol 10 < 15
        self.assertTrue(all(sg is None for _, sg in sigs))
        self.assertEqual(s.state, "RETEST_WATCH")


class TestV6EntryGate(unittest.TestCase):
    def _setup(self, **kw):
        s = make_strat(volume_filter="V6", vol_lookback_2m=5, **kw)
        feed(s, DAY1 + breakout_15_rows())  # V6 does not gate the breakout
        self.assertEqual(s.state, "RETEST_WATCH")
        s._recent2_vols = deque([10.0] * 5, maxlen=5)
        s._recent2_ranges = deque([2.0] * 5, maxlen=5)
        return s

    def test_v6_allows_volume_with_range(self):
        s = self._setup()
        # Signal 2m: vol 50 > mean(~12); range 3.0 >= 1.2 * median(2.0).
        sigs = feed(s, [
            ("2024-01-02 00:16", 110.0, 113.0, 110.0, 112.0, 25.0),
            ("2024-01-02 00:17", 112.0, 113.0, 110.0, 112.0, 25.0),
            ("2024-01-02 00:18", 112.0, 112.5, 111.5, 112.0, 10.0),
        ])
        sig = [sg for _, sg in sigs if sg is not None]
        self.assertEqual(len(sig), 1)

    def test_v6_blocks_doi_without_range(self):
        s = self._setup()
        # Volume passes but range 1.0 < 1.2 * 2.0 -> blocked.
        sigs = feed(s, retest_rows(vols=(25.0, 25.0, 10.0)))
        self.assertTrue(all(sg is None for _, sg in sigs))
        self.assertEqual(s.state, "RETEST_WATCH")


class TestFilterValidation(unittest.TestCase):
    def test_invalid_filter_raises(self):
        with self.assertRaises(ValueError):
            make_strat(volume_filter="V7")

    def test_none_is_default_and_unchanged(self):
        s = make_strat()
        feed(s, DAY1 + breakout_15_rows())
        self.assertEqual(s.state, "RETEST_WATCH")
        sigs = feed(s, retest_rows())
        self.assertEqual(len([sg for _, sg in sigs if sg is not None]), 1)


if __name__ == "__main__":
    unittest.main()
