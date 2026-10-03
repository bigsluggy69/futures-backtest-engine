"""Event-driven backtest engine.

Core guarantees (enforced structurally, verified by tests):
  * Signals are generated only on *closed* bars.
  * An entry signal produced at the close of bar t is executed at the *open*
    of bar t+1 (plus adverse slippage). No same-bar fills.
  * Same seed + same data => byte-identical results.

Execution model per 1-minute bar t (in order):
  1. If a pending entry exists, re-check shared guards, then fill at bar t open
     with adverse slippage.
  2. If a position is open, evaluate exit conditions using ONLY bar t's OHLC:
       a. stop-loss and profit-target intrabar checks.
          # UNDEFINED: spec does not say which fills first when a bar's range
          # covers both stop and target. Conservative: assume the STOP is hit
          # first (worst case for the strategy).
       b. max holding time: exit at bar t close with slippage.
  3. Bar t is now closed: call strategy.on_bar_close(bar). Any signal emitted
     here becomes a pending entry for bar t+1.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Optional

import pandas as pd

from .costs import InstrumentSpec


@dataclass
class Trade:
    direction: int               # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float           # includes entry slippage
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None  # includes exit slippage
    exit_reason: str = ""
    gross_pnl: float = 0.0
    net_pnl: float = 0.0         # after commissions + slippage (embedded in prices)


@dataclass
class EngineConfig:
    # Shared guards (all constructor-configurable, per spec).
    max_trades_per_day: int = 2
    cooldown_after_loss_min: int = 30
    session_start: str = "13:30"   # UTC HH:MM
    session_end: str = "20:00"     # UTC HH:MM
    max_spread_ticks: int = 2      # skip entry if spread exceeds this
    max_holding_min: int = 120
    daily_loss_cap: float = 500.0  # USD; no new entries once breached
    contracts: int = 1
    seed: int = 42


class Engine:
    def __init__(self, strategy, spec: InstrumentSpec, config: EngineConfig):
        self.strategy = strategy
        self.spec = spec
        self.cfg = config
        # Seeded RNG reserved for future stochastic elements; kept so that any
        # such use is reproducible. The reference strategy is fully deterministic
        # and never draws from it.
        self.rng = random.Random(config.seed)

        self.session_start = _parse_hhmm(config.session_start)
        self.session_end = _parse_hhmm(config.session_end)

        self.trades: list[Trade] = []
        self._pending_entry: Optional[dict] = None
        self._position: Optional[Trade] = None
        self._day: Optional[pd.Timestamp] = None
        self._trades_today = 0
        self._realized_today = 0.0
        self._last_loss_time: Optional[pd.Timestamp] = None
        self._spread_ticks: float = 1.0
        # UNDEFINED: OHLCV data carries no bid/ask. Conservative assumption:
        # a constant 1-tick spread (typical for liquid micro contracts). Provide
        # set_spread_series() if real spread data becomes available.
        self._spread_series: Optional[pd.Series] = None
        self._spread_map: Optional[dict] = None  # built per run() call

    def set_spread_series(self, spread_ticks: pd.Series) -> None:
        """Optional per-bar spread (in ticks), indexed like the input bars."""
        self._spread_series = spread_ticks

    # ---- guards -----------------------------------------------------------

    def _in_session(self, ts: pd.Timestamp) -> bool:
        t = ts.time()
        if self.session_start <= self.session_end:
            return self.session_start <= t < self.session_end
        return t >= self.session_start or t < self.session_end

    def _guards_allow_entry(self, ts: pd.Timestamp) -> bool:
        if not self._in_session(ts):
            return False
        if self._trades_today >= self.cfg.max_trades_per_day:
            return False
        if self._realized_today <= -abs(self.cfg.daily_loss_cap):
            return False
        if self._last_loss_time is not None:
            elapsed = (ts - self._last_loss_time).total_seconds() / 60.0
            if elapsed < self.cfg.cooldown_after_loss_min:
                return False
        # Spread lookup: defensive map built in run() (reindexed + ffill).
        # Missing timestamps fall back to the constant default — never raises.
        if self._spread_map is not None:
            spread = self._spread_map.get(ts, self._spread_ticks)
        else:
            spread = self._spread_ticks
        if spread > self.cfg.max_spread_ticks:
            return False
        return True

    # ---- main loop ---------------------------------------------------------

    def run(self, bars: pd.DataFrame) -> list[Trade]:
        self.strategy.reset()
        slip_price = self.spec.slippage_ticks_per_side * self.spec.tick_size

        # Align an optional per-bar spread series to the bar timestamps.
        # UNDEFINED: ffill only — bfill would leak FUTURE spreads into past
        # bars (lookahead). Leading gaps (no spread known yet) fall back to
        # the constant default via dict.get().
        self._spread_map = None
        if self._spread_series is not None:
            aligned = self._spread_series.reindex(bars["timestamp_utc"]).ffill()
            self._spread_map = {
                t: float(v) for t, v in aligned.items() if not pd.isna(v)
            }

        for row in bars.itertuples(index=False):
            ts = row.timestamp_utc
            o, h, l, c, v = row.open, row.high, row.low, row.close, row.volume

            # Day rollover bookkeeping (calendar day in UTC).
            day = ts.normalize()
            if day != self._day:
                self._day = day
                self._trades_today = 0
                self._realized_today = 0.0

            # 1) Execute pending entry at this bar's OPEN (signal came from the
            #    previous closed bar).
            if self._pending_entry is not None and self._position is None:
                if self._guards_allow_entry(ts):
                    sig = self._pending_entry
                    fill = o + slip_price * sig["direction"]  # adverse slippage
                    self._position = Trade(
                        direction=sig["direction"],
                        entry_time=ts,
                        entry_price=fill,
                    )
                    self._position_stop = sig["stop_price"]
                    self._position_target = sig["target_price"]
                    self._trades_today += 1
                self._pending_entry = None

            # 2) Manage open position using this bar's OHLC only.
            if self._position is not None:
                self._manage_position(ts, o, h, l, c)

            # 3) Bar closed -> feed strategy; it may queue an entry for next bar.
            sig = self.strategy.on_bar_close(ts, o, h, l, c, v)
            if sig is not None and self._position is None:
                self._pending_entry = sig

        # Force-close any open position at the final close (conservative).
        if self._position is not None:
            last = bars.iloc[-1]
            self._close_position(last["timestamp_utc"], last["close"], "end_of_data")

        return self.trades

    # ---- position management ------------------------------------------------

    def _manage_position(self, ts, o, h, l, c) -> None:
        pos = self._position
        d = pos.direction
        stop = self._position_stop
        target = self._position_target

        stop_hit = (l <= stop) if d == 1 else (h >= stop)
        target_hit = (h >= target) if d == 1 else (l <= target)

        if stop_hit and target_hit:
            # UNDEFINED: which filled first is unknowable from OHLC alone.
            # Conservative: stop fills first.
            self._close_position(ts, self._stop_fill(stop, o), "stop_loss")
        elif stop_hit:
            self._close_position(ts, self._stop_fill(stop, o), "stop_loss")
        elif target_hit:
            self._close_position(ts, target, "target")
        elif (ts - pos.entry_time).total_seconds() / 60.0 >= self.cfg.max_holding_min:
            self._close_position(ts, c, "max_holding_time")

    def _stop_fill(self, stop: float, bar_open: float) -> float:
        """If the bar OPENS beyond the stop (gap), the stop fills at the open
        (worse). Conservative: never assume a better-than-open fill."""
        d = self._position.direction
        if d == 1:
            return min(stop, bar_open)
        return max(stop, bar_open)

    def _close_position(self, ts, raw_price, reason) -> None:
        pos = self._position
        slip = self.spec.slippage_ticks_per_side * self.spec.tick_size
        exit_price = raw_price - slip * pos.direction  # adverse on exit too
        pos.exit_time = ts
        pos.exit_price = exit_price
        pos.exit_reason = reason
        price_diff = (pos.exit_price - pos.entry_price) * pos.direction
        ticks = price_diff / self.spec.tick_size
        pos.gross_pnl = ticks * self.spec.tick_value * self.cfg.contracts
        pos.net_pnl = pos.gross_pnl - self.spec.round_trip_commission * self.cfg.contracts
        self.trades.append(pos)
        self._realized_today += pos.net_pnl
        if pos.net_pnl < 0:
            self._last_loss_time = ts
        self._position = None


def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))
