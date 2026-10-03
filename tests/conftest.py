import pandas as pd

from backtest.costs import INSTRUMENTS
from backtest.engine import Engine, EngineConfig
from backtest.strategies.pdh_pdl_breakout import PdhPdlBreakoutRetest


def make_bars(rows):
    """rows: list of (timestamp_str, o, h, l, c, v)."""
    return pd.DataFrame(
        [(pd.Timestamp(t, tz="UTC"), o, h, l, c, v) for t, o, h, l, c, v in rows],
        columns=["timestamp_utc", "open", "high", "low", "close", "volume"],
    )


def breakout_retest_rows(entry_open=109.5, entry_bar_high=110.0,
                         entry_bar_low=109.0, entry_bar_close=109.5,
                         target_bar_high=115.0, target_bar_low=109.0):
    """Deterministic scenario:
      Day 1: high 110, low 90 -> PDH=110, PDL=90.
      Day 2 00:00-00:14: flat at 111 -> 15m close 111 > 110 = BREAKOUT @00:15.
      00:16-00:17: low touches 110 -> RETEST signal when bar 00:18 closes the
                   2m bucket.
      00:19: entry at open=109.50 (signal must fill HERE, not earlier).
      00:20: high 115 -> target (114.50) hit.
    """
    rows = [
        ("2024-01-01 00:00", 100, 110, 100, 105, 10),  # sets day1 high
        ("2024-01-01 00:01", 100, 105, 90, 100, 10),   # sets day1 low
    ]
    # Day 2: 15 consecutive 1m bars flat at 111 (00:00..00:14).
    for m in range(15):
        rows.append((f"2024-01-02 00:{m:02d}", 111, 111, 111, 111, 10))
    rows += [
        ("2024-01-02 00:15", 111, 111, 110.5, 110.5, 10),
        ("2024-01-02 00:16", 110.5, 111, 110.0, 110.5, 10),  # touch of level
        ("2024-01-02 00:17", 110.5, 111, 110.0, 110.5, 10),
        ("2024-01-02 00:18", 110.5, 110.6, 110.4, 110.5, 10),
        ("2024-01-02 00:19", entry_open, entry_bar_high, entry_bar_low,
         entry_bar_close, 10),                                # ENTRY bar
        ("2024-01-02 00:20", 110, target_bar_high, target_bar_low, 112, 10),
        ("2024-01-02 00:21", 112, 112, 112, 112, 10),
    ]
    return rows


SPEC = INSTRUMENTS["MES"]


def make_cfg(**overrides):
    defaults = dict(
        max_trades_per_day=3,
        cooldown_after_loss_min=0,
        session_start="00:00",
        session_end="23:59",
        max_holding_min=120,
        seed=42,
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


def build_engine(spec=SPEC, cfg=None, **strat_kwargs):
    if cfg is None:
        cfg = make_cfg()
    defaults = dict(ema_proximity_ticks=0.0, expiry_min=60,
                    stop_ticks=8.0, target_ticks=16.0)
    defaults.update(strat_kwargs)
    strat = PdhPdlBreakoutRetest(spec=spec, **defaults)
    return Engine(strat, spec, cfg)
