"""Performance metrics computed from a list of closed Trade objects.

Sharpe convention: computed on per-UTC-day net P&L, annualized with sqrt(252).
Every UTC calendar day from the first trade's exit date to the last trade's
exit date contributes — a day with no trades contributes 0. This is the
conservative choice: idle days dilute a profitable mean instead of being
excluded. Returns 0.0 if there is no variance or fewer than two days.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta


def compute_metrics(trades) -> dict:
    n = len(trades)
    net = [t.net_pnl for t in trades]
    total_net = sum(net)

    wins = [p for p in net if p > 0]
    losses = [p for p in net if p < 0]

    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Max drawdown on cumulative net P&L.
    peak, max_dd, cum = 0.0, 0.0, 0.0
    for p in net:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Daily Sharpe (annualized). Every UTC calendar day in [first exit, last
    # exit] contributes; no-trade days contribute 0 (conservative).
    daily = defaultdict(float)
    for t in trades:
        daily[t.exit_time.normalize()] += t.net_pnl
    rets = []
    if daily:
        day, end = min(daily), max(daily)
        while day <= end:
            rets.append(daily.get(day, 0.0))
            day += timedelta(days=1)
    if len(rets) >= 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sharpe = (mean / math.sqrt(var)) * math.sqrt(252) if var > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "trades": n,
        "net_pnl": round(total_net, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(len(wins) / n, 4) if n else 0.0,
        "avg_winner": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loser": round(sum(losses) / len(losses), 2) if losses else 0.0,
    }
