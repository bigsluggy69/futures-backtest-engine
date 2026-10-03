"""Strategy #1: PDH/PDL breakout-retest, implemented as a deterministic FSM.

    IDLE
      -> RETEST_WATCH : a closed 15m bar closes beyond prior-day high/low
                        (volume filter V1/V2 may gate this transition)
      -> ENTRY        : signal returned; engine fills at next 1m open.
                        Triggered when a closed 2m bar touches the level
                        and/or closes within X ticks of the 13 EMA
                        (of 2m closes). Volume filter V3-V6 may gate this.
      Side exits: INVALIDATION (a 15m bar CLOSES back inside the level;
                  wicks ignored) and EXPIRY (no retest within N minutes).
      A 15m bar that invalidates one direction AND closes beyond the
      opposite level starts the opposite setup on that same bar.

All tunables are constructor arguments. Nothing is hardcoded.

Volume confirmation (volume_filter="none" | "V1".."V6", default "none"):
  V1: breakout 15m bar volume >= v1_mult x rolling slot_median_days-day median
      volume of the same 15m clock slot. Gates IDLE -> RETEST_WATCH.
  V2: breakout 15m bar volume >= v2_mult x mean of prior vol_lookback_15m
      15m bars. Gates IDLE -> RETEST_WATCH.
  V3: signal 2m bar volume > mean of prior vol_lookback_2m 2m bars.
      Gates ENTRY.
  V4: mean per-minute volume of the pullback 2m bars (closed 2m bars between
      the breakout 15m close and the signal bar) < v4_pct x the breakout
      15m bar's per-minute volume. Gates ENTRY.
  V5: signal 2m bar volume >= v5_mult x slot_median_days-day median of the
      same 2m clock slot. Gates ENTRY.
  V6: V3 holds AND signal bar range >= v6_range_mult x median 2m range of the
      prior vol_lookback_2m bars. Gates ENTRY.

UNDEFINED choices (most conservative interpretation taken):
  * Prior-day high/low = prior UTC calendar day's full-session high/low,
    known only once that day is complete.
  * Entry direction = breakout direction (long on PDH break, short on PDL).
  * Stop/target distances are in ticks, anchored to the signal bar's close
    (the engine's actual fill is the next bar's open; anchoring to the close
    is deterministic and slightly conservative).
  * "Touch the level" for a long retest means the 2m bar's low <= level.
  * "Within X ticks of the 13 EMA" uses |2m close - EMA13| <= X ticks.
  * EMA13 is seeded with the first closed 2m bar (standard recursive EMA,
    alpha = 2/(13+1)); no warmup burn-in is imposed.
  * The EMA proximity check uses an EMA that INCLUDES the current closed 2m
    bar. This is not lookahead (the bar is complete), but it dampens the
    measured distance versus a prior-bar EMA. Documented, not changed.
  * Volume is read from CLOSED bars only. Every gate is evaluated BEFORE the
    current bar is appended to its own history (slot median / rolling mean),
    so a bar can never contaminate its own benchmark.
  * Slot medians (V1/V5) need slot_median_days observations; rolling means
    need their full lookback. With shorter history the filter ABSTAINS
    (no signal) rather than guessing.
  * A V3-V6 rejection does NOT cancel the setup: the state stays RETEST_WATCH
    and later 2m bars may still trigger. (A rejected candidate bar counts as
    a pullback bar for a later V4 evaluation.)
  * V4 with an empty pullback (signal on the first 2m bar after breakout)
    abstains — there is no pullback to measure.
"""
from __future__ import annotations

from collections import deque
from statistics import mean, median
from typing import Optional

from .base import Strategy
from ..costs import InstrumentSpec

_VOLUME_FILTERS = ("none", "V1", "V2", "V3", "V4", "V5", "V6")


class _AggBar:
    """Incremental aggregator for an N-minute bar built from closed 1m bars."""

    def __init__(self):
        self.bucket = None
        self.o = self.h = self.l = self.c = None
        self.v = 0.0

    def add(self, bucket, o, h, l, c, v):
        """Returns the completed bar dict when a bucket closes, else None."""
        completed = None
        if self.bucket is not None and bucket != self.bucket:
            completed = {"bucket": self.bucket, "o": self.o, "h": self.h,
                         "l": self.l, "c": self.c, "v": self.v}
            self.o = self.h = self.l = self.c = None
            self.v = 0.0
        if self.o is None:
            self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        else:
            self.h = max(self.h, h)
            self.l = min(self.l, l)
            self.c = c
            self.v += v
        self.bucket = bucket
        return completed


class _EMA:
    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1.0)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class PdhPdlBreakoutRetest(Strategy):
    def __init__(
        self,
        spec: InstrumentSpec,
        ema_proximity_ticks: float,     # X: retest within X ticks of 13 EMA
        expiry_min: int,                # N: no retest within N minutes -> EXPIRY
        stop_ticks: float,
        target_ticks: float,
        ema_span: int = 13,
        volume_filter: str = "none",   # "none" | "V1".."V6"
        v1_mult: float = 1.5,
        v2_mult: float = 2.0,
        v4_pct: float = 0.70,
        v5_mult: float = 1.5,
        v6_range_mult: float = 1.2,
        slot_median_days: int = 20,    # history required for V1/V5 slot medians
        vol_lookback_15m: int = 10,    # V2 lookback (15m bars)
        vol_lookback_2m: int = 20,     # V3/V6 lookback (2m bars)
    ):
        if volume_filter not in _VOLUME_FILTERS:
            raise ValueError(f"volume_filter must be one of {_VOLUME_FILTERS}")
        self.spec = spec
        self.X = ema_proximity_ticks
        self.expiry_min = expiry_min
        self.stop_ticks = stop_ticks
        self.target_ticks = target_ticks
        self.ema_span = ema_span
        self.volume_filter = volume_filter
        self.v1_mult = v1_mult
        self.v2_mult = v2_mult
        self.v4_pct = v4_pct
        self.v5_mult = v5_mult
        self.v6_range_mult = v6_range_mult
        self.slot_median_days = slot_median_days
        self.vol_lookback_15m = vol_lookback_15m
        self.vol_lookback_2m = vol_lookback_2m
        self.reset()

    def reset(self) -> None:
        self.state = "IDLE"
        self.level: Optional[float] = None
        self.direction = 0              # +1 long breakout, -1 short breakout
        self.breakout_time = None

        self.pdh: Optional[float] = None
        self.pdl: Optional[float] = None
        self._cur_day = None
        self._cur_day_h = None
        self._cur_day_l = None

        self._agg15 = _AggBar()
        self._agg2 = _AggBar()
        self._ema2 = _EMA(self.ema_span)

        # Volume-filter history. Updated with CLOSED bars only, always AFTER
        # the bar has been evaluated against them (no self-contamination).
        self._vol15_by_slot: dict = {}  # time -> deque of 15m volumes
        self._vol2_by_slot: dict = {}   # time -> deque of 2m volumes
        self._recent15_vols: deque = deque(maxlen=self.vol_lookback_15m)
        self._recent2_vols: deque = deque(maxlen=self.vol_lookback_2m)
        self._recent2_ranges: deque = deque(maxlen=self.vol_lookback_2m)
        self._breakout_vol: Optional[float] = None
        self._pullback_2m: list = []    # volumes of 2m bars since breakout

    # -- helpers -------------------------------------------------------------

    def _to_idle(self):
        self.state = "IDLE"
        self.level = None
        self.direction = 0
        self.breakout_time = None
        self._breakout_vol = None
        self._pullback_2m = []

    def _roll_day(self, ts, h, l):
        day = ts.normalize()
        if day != self._cur_day:
            if self._cur_day is not None:
                # Prior day is now COMPLETE -> its high/low become PDH/PDL.
                self.pdh, self.pdl = self._cur_day_h, self._cur_day_l
            self._cur_day = day
            self._cur_day_h, self._cur_day_l = h, l
        else:
            self._cur_day_h = max(self._cur_day_h, h)
            self._cur_day_l = min(self._cur_day_l, l)

    def _slot_deque(self, store: dict, slot, maxlen: int) -> deque:
        dq = store.get(slot)
        if dq is None:
            dq = deque(maxlen=maxlen)
            store[slot] = dq
        return dq

    def _record_15m(self, b15) -> None:
        """Append a CLOSED 15m bar to volume history. Call AFTER evaluation."""
        slot = b15["bucket"].time()
        self._slot_deque(self._vol15_by_slot, slot,
                         self.slot_median_days).append(b15["v"])
        self._recent15_vols.append(b15["v"])

    def _record_2m(self, b2) -> None:
        """Append a CLOSED 2m bar to volume history. Call AFTER evaluation."""
        slot = b2["bucket"].time()
        self._slot_deque(self._vol2_by_slot, slot,
                         self.slot_median_days).append(b2["v"])
        self._recent2_vols.append(b2["v"])
        self._recent2_ranges.append(b2["h"] - b2["l"])

    # -- volume gates (evaluated BEFORE the current bar is recorded) ---------

    def _breakout_volume_ok(self, b15) -> bool:
        """V1/V2 gate the IDLE -> RETEST_WATCH transition."""
        f = self.volume_filter
        if f == "V1":
            vols = self._vol15_by_slot.get(b15["bucket"].time(), [])
            # UNDEFINED: with less than slot_median_days observations the
            # median is not credible — abstain (no signal) rather than guess.
            if len(vols) < self.slot_median_days:
                return False
            return b15["v"] >= self.v1_mult * median(vols)
        if f == "V2":
            if len(self._recent15_vols) < self.vol_lookback_15m:
                return False
            return b15["v"] >= self.v2_mult * mean(self._recent15_vols)
        return True  # "none", V3-V6 do not gate the breakout

    def _entry_volume_ok(self, b2) -> bool:
        """V3-V6 gate the retest -> ENTRY transition."""
        f = self.volume_filter
        if f == "V3":
            if len(self._recent2_vols) < self.vol_lookback_2m:
                return False
            return b2["v"] > mean(self._recent2_vols)
        if f == "V4":
            # UNDEFINED: no pullback bars yet (signal on the first 2m bar
            # after breakout) — abstain, there is nothing to measure.
            if not self._pullback_2m or self._breakout_vol is None:
                return False
            pullback_per_min = mean(v / 2.0 for v in self._pullback_2m)
            breakout_per_min = self._breakout_vol / 15.0
            return pullback_per_min < self.v4_pct * breakout_per_min
        if f == "V5":
            vols = self._vol2_by_slot.get(b2["bucket"].time(), [])
            if len(vols) < self.slot_median_days:
                return False
            return b2["v"] >= self.v5_mult * median(vols)
        if f == "V6":
            if len(self._recent2_vols) < self.vol_lookback_2m:
                return False
            if not b2["v"] > mean(self._recent2_vols):
                return False
            if len(self._recent2_ranges) < self.vol_lookback_2m:
                return False
            return (b2["h"] - b2["l"]) >= self.v6_range_mult * median(
                self._recent2_ranges)
        return True  # "none", V1, V2 do not gate the entry

    # -- main ----------------------------------------------------------------

    def on_bar_close(self, ts, o, h, l, c, v) -> Optional[dict]:
        self._roll_day(ts, h, l)

        # Close any completed higher-timeframe bars first (they use only
        # already-closed 1m bars). Each bar is evaluated against volume
        # history BEFORE being recorded into it.
        b15 = self._agg15.add(ts.floor("15min"), o, h, l, c, v)
        b2 = self._agg2.add(ts.floor("2min"), o, h, l, c, v)

        signal = None

        if b15 is not None:
            signal = self._on_15m_close(b15, ts)
            self._record_15m(b15)

        if b2 is not None:
            # NOTE: the EMA includes this CLOSED 2m bar. Not lookahead (the
            # bar is complete), but the proximity distance is dampened versus
            # a prior-bar EMA. Documented, not changed.
            ema = self._ema2.update(b2["c"])
            sig2 = self._on_2m_close(b2, ema, ts)
            if sig2 is not None:
                signal = sig2
            self._record_2m(b2)
            # Track pullback bars while watching (the signal bar itself is
            # excluded: it is appended only if we are still watching, i.e.
            # no signal was emitted — a rejected candidate counts as pullback).
            if self.state == "RETEST_WATCH":
                self._pullback_2m.append(b2["v"])

        return signal

    # -- FSM transitions -------------------------------------------------------

    def _start_watch(self, b15, ts, direction: int, level: float) -> bool:
        """Attempt IDLE -> RETEST_WATCH. Returns False if the V1/V2 gate
        abstains (state stays IDLE)."""
        if not self._breakout_volume_ok(b15):
            return False
        self.state = "RETEST_WATCH"
        self.level = level
        self.direction = direction
        self.breakout_time = ts
        self._breakout_vol = b15["v"]
        self._pullback_2m = []
        return True

    def _on_15m_close(self, b15, ts) -> Optional[dict]:
        close = b15["c"]
        if self.state == "IDLE":
            if self.pdh is not None and close > self.pdh:
                self._start_watch(b15, ts, +1, self.pdh)
            elif self.pdl is not None and close < self.pdl:
                self._start_watch(b15, ts, -1, self.pdl)
        elif self.state == "RETEST_WATCH":
            # INVALIDATION requires a 15m CLOSE back inside the level.
            inside = close < self.level if self.direction == +1 else close > self.level
            if inside:
                # UNDEFINED (review): a 15m close that invalidates one
                # direction may simultaneously close beyond the OPPOSITE
                # level. Register the opposite breakout on this same bar
                # instead of missing it until the next 15m close.
                if self.direction == +1 and self.pdl is not None and close < self.pdl:
                    self._to_idle()
                    self._start_watch(b15, ts, -1, self.pdl)
                elif self.direction == -1 and self.pdh is not None and close > self.pdh:
                    self._to_idle()
                    self._start_watch(b15, ts, +1, self.pdh)
                else:
                    self._to_idle()
        return None

    def _on_2m_close(self, b2, ema, ts) -> Optional[dict]:
        if self.state != "RETEST_WATCH":
            return None

        # EXPIRY: no retest within expiry_min of the breakout 15m close.
        if (ts - self.breakout_time).total_seconds() / 60.0 > self.expiry_min:
            self._to_idle()
            return None

        # RETEST: touch of the level and/or proximity to 13 EMA.
        if self.direction == +1:
            touched = b2["l"] <= self.level
        else:
            touched = b2["h"] >= self.level
        near_ema = abs(b2["c"] - ema) <= self.X * self.spec.tick_size

        if touched or near_ema:
            # UNDEFINED (review): a V3-V6 rejection does not cancel the
            # setup — the state stays RETEST_WATCH and later 2m bars may
            # still trigger. Only a passing gate produces ENTRY.
            if not self._entry_volume_ok(b2):
                return None
            d = self.direction
            entry_ref = b2["c"]  # UNDEFINED anchor; engine fills next bar open
            stop = entry_ref - d * self.stop_ticks * self.spec.tick_size
            target = entry_ref + d * self.target_ticks * self.spec.tick_size
            self._to_idle()
            return {"direction": d, "stop_price": stop, "target_price": target}
        return None
