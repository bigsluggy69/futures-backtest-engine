"""Deterministic synthetic 1-minute OHLCV generator for research-layer tests.

Not market data. Day-level drift regimes are drawn so that some days close
beyond the prior day's high/low (PDH/PDL breakouts) and the strategy actually
trades. A fixed generator seed makes the output identical on every call.
"""
from __future__ import annotations

import random

import pandas as pd

TICK = 0.25


def _snap(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def make_synthetic_bars(n_days: int = 60, seed: int = 11,
                        start: str = "2024-01-01",
                        session_start_min: int = 13 * 60 + 30,
                        session_minutes: int = 390) -> pd.DataFrame:
    """Session-only 1m bars (default 13:30-20:00 UTC) for `n_days` calendar days."""
    gen = random.Random(seed)
    day0 = pd.Timestamp(start, tz="UTC")
    rows = []
    price = 4000.0
    for d in range(n_days):
        day = day0 + pd.Timedelta(days=d)
        drift = gen.choice([-0.06, -0.03, 0.0, 0.03, 0.06])
        price = _snap(price + gen.choice([-4, -2, 0, 2, 4]))  # overnight gap
        for m in range(session_minutes):
            ts = day + pd.Timedelta(minutes=session_start_min + m)
            o = price
            c = _snap(o + drift + gen.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
            h = _snap(max(o, c) + gen.choice([0.0, 0.25, 0.5]))
            l = _snap(min(o, c) - gen.choice([0.0, 0.25, 0.5]))
            # U-shaped intraday volume plus noise.
            shape = 1.0 + 1.5 * abs(m - session_minutes / 2) / (session_minutes / 2)
            v = int(gen.uniform(60, 140) * shape)
            rows.append((ts, o, h, l, c, v))
            price = c
    return pd.DataFrame(
        rows, columns=["timestamp_utc", "open", "high", "low", "close", "volume"])


def write_csv(bars: pd.DataFrame, path: str) -> None:
    bars.to_csv(path, index=False)
