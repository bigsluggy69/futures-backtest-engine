"""Strategy interface.

A strategy receives only *closed* 1-minute bars, one at a time, and may return
an entry signal dict: {"direction": +1|-1, "stop_price": float,
"target_price": float}. The engine fills it at the NEXT bar's open.
Strategies must maintain all multi-timeframe aggregation internally so that no
information from incomplete or future bars is ever visible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Strategy(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def on_bar_close(self, ts, o, h, l, c, v) -> Optional[dict]: ...
