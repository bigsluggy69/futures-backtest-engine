"""Cost model: commissions, slippage, instrument tick specs.

All cost parameters are configurable; defaults are per the project spec.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size: float        # price increment per tick
    tick_value: float       # USD per tick per contract
    commission_per_side: float = 1.25   # USD per side per contract
    slippage_ticks_per_side: float = 1.0  # ticks of adverse slippage per side

    @property
    def round_trip_commission(self) -> float:
        return 2.0 * self.commission_per_side


# Tick values verified 2026-10-03 against CME specs (the original brief gave
# per-point values: MES $5/pt, MNQ $20/pt for full-size NQ — corrected here to
# per-tick). MES: 0.25pt tick x $5/pt = $1.25/tick. MNQ: 0.25pt x $2/pt =
# $0.50/tick. MGC: $0.10 tick on 10oz = $1.00/tick.
INSTRUMENTS = {
    "MES": InstrumentSpec("MES", tick_size=0.25, tick_value=1.25),
    "MNQ": InstrumentSpec("MNQ", tick_size=0.25, tick_value=0.50),
    "MGC": InstrumentSpec("MGC", tick_size=0.10, tick_value=1.0),
}
