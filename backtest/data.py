"""CSV loader for 1-minute OHLCV bars.

Required columns: timestamp_utc, open, high, low, close, volume.
The loader validates columns and duplicates and rejects OHLC inconsistencies
so a bad input file fails loudly instead of silently corrupting a backtest.
Non-monotonic timestamps are sorted into ascending order (documented here
rather than failing: the sort is deterministic and the alternative — running
an event engine on out-of-order bars — is worse).
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "volume"]


def load_1m_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps in input data.")
    if not df["timestamp_utc"].is_monotonic_increasing:
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if (df["high"] < df[["open", "close"]].max(axis=1)).any() or (
        df["low"] > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError("OHLC inconsistency detected (high/low bounds violated).")

    return df
