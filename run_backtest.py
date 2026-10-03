"""CLI entry point: run Strategy #1 on a CSV of 1-minute OHLCV bars.

Usage:
    python run_backtest.py data.csv --instrument MES --ema-ticks 2 \
        --expiry-min 60 --stop-ticks 8 --target-ticks 16
"""
from __future__ import annotations

import argparse
import dataclasses
import json

from backtest.costs import INSTRUMENTS, InstrumentSpec
from backtest.data import load_1m_csv
from backtest.engine import Engine, EngineConfig
from backtest.metrics import compute_metrics
from backtest.strategies.pdh_pdl_breakout import PdhPdlBreakoutRetest


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic futures backtest (research only)")
    p.add_argument("csv", help="Path to 1m OHLCV CSV "
                   "(columns: timestamp_utc,open,high,low,close,volume)")
    p.add_argument("--instrument", default="MES", choices=list(INSTRUMENTS))
    p.add_argument("--tick-value", type=float, default=None,
                   help="Override USD/tick without editing costs.py "
                        "(default: instrument spec)")
    p.add_argument("--ema-ticks", type=float, default=2.0, help="X: EMA proximity in ticks")
    p.add_argument("--expiry-min", type=int, default=60, help="N: retest expiry in minutes")
    p.add_argument("--stop-ticks", type=float, default=8.0)
    p.add_argument("--target-ticks", type=float, default=16.0)
    # Volume confirmation filter.
    p.add_argument("--volume-filter", default="none",
                   choices=["none", "V1", "V2", "V3", "V4", "V5", "V6"])
    p.add_argument("--v1-mult", type=float, default=1.5)
    p.add_argument("--v2-mult", type=float, default=2.0)
    p.add_argument("--v4-pct", type=float, default=0.70)
    p.add_argument("--v5-mult", type=float, default=1.5)
    p.add_argument("--v6-range-mult", type=float, default=1.2)
    # Shared guards.
    p.add_argument("--max-trades-per-day", type=int, default=2)
    p.add_argument("--cooldown-min", type=int, default=30)
    p.add_argument("--session-start", default="13:30")
    p.add_argument("--session-end", default="20:00")
    p.add_argument("--max-spread-ticks", type=int, default=2)
    p.add_argument("--max-holding-min", type=int, default=120)
    p.add_argument("--daily-loss-cap", type=float, default=500.0)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bars = load_1m_csv(args.csv)
    spec = INSTRUMENTS[args.instrument]
    if args.tick_value is not None:
        spec = dataclasses.replace(spec, tick_value=args.tick_value)
    cfg = EngineConfig(
        max_trades_per_day=args.max_trades_per_day,
        cooldown_after_loss_min=args.cooldown_min,
        session_start=args.session_start,
        session_end=args.session_end,
        max_spread_ticks=args.max_spread_ticks,
        max_holding_min=args.max_holding_min,
        daily_loss_cap=args.daily_loss_cap,
        contracts=args.contracts,
        seed=args.seed,
    )
    strat = PdhPdlBreakoutRetest(
        spec=spec,
        ema_proximity_ticks=args.ema_ticks,
        expiry_min=args.expiry_min,
        stop_ticks=args.stop_ticks,
        target_ticks=args.target_ticks,
        volume_filter=args.volume_filter,
        v1_mult=args.v1_mult,
        v2_mult=args.v2_mult,
        v4_pct=args.v4_pct,
        v5_mult=args.v5_mult,
        v6_range_mult=args.v6_range_mult,
    )
    trades = Engine(strat, spec, cfg).run(bars)
    print(json.dumps(compute_metrics(trades), indent=2))


if __name__ == "__main__":
    main()
