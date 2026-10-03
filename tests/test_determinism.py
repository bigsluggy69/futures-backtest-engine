"""Proof 2: deterministic replay — same seed + same data = identical results."""
import unittest
from dataclasses import asdict

from tests.conftest import make_bars, breakout_retest_rows, build_engine


def serialize(trades):
    return [asdict(t) for t in trades]


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_data_identical_results(self):
        bars = make_bars(breakout_retest_rows())
        r1 = serialize(build_engine().run(bars))
        r2 = serialize(build_engine().run(bars))
        self.assertEqual(r1, r2)

    def test_determinism_on_longer_random_walk(self):
        """A longer pseudo-random series (fixed generator seed at DATA-creation
        time) must replay identically across independent engine runs."""
        import random
        gen = random.Random(7)
        rows = [("2024-01-01 00:00", 100, 110, 100, 105, 10),
                ("2024-01-01 00:01", 100, 105, 90, 100, 10)]
        price = 100.0
        for m in range(6 * 60):  # 6 hours of 1m bars on day 2
            ts = f"2024-01-02 {m // 60:02d}:{m % 60:02d}"
            drift = gen.choice([-0.5, -0.25, 0.0, 0.25, 0.5])
            o = price
            c = price + drift
            h = max(o, c) + 0.25
            l = min(o, c) - 0.25
            rows.append((ts, o, h, l, c, 10))
            price = c
        bars = make_bars(rows)
        r1 = serialize(build_engine().run(bars))
        r2 = serialize(build_engine().run(bars))
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
