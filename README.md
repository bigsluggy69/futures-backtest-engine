# Futures Backtest Engine — Research Scaffold

Event-driven, deterministic backtest engine for rule-based futures strategies.
**Research only** — no broker integration, no data APIs, no live trading.

Requires Python 3.11+. Dependencies: `pandas`, `numpy` only (tests run on the
stdlib `unittest` — no pytest needed).

## Layout

```
futures_backtest/
├── backtest/
│   ├── data.py                       # strict 1m OHLCV CSV loader
│   ├── costs.py                      # InstrumentSpec (tick size/value, commission, slippage)
│   ├── engine.py                     # event-driven engine + shared guards
│   ├── metrics.py                    # net P&L, Sharpe, max DD, PF, win rate, ...
│   └── strategies/
│       ├── base.py                   # Strategy interface
│       └── pdh_pdl_breakout.py       # Strategy #1: PDH/PDL breakout-retest FSM
├── tests/                            # unittest suite (39 tests)
└── run_backtest.py                   # CLI entry point
```

## Running on your data

CSV of 1-minute bars with columns: `timestamp_utc, open, high, low, close, volume`
(timestamps parsed as UTC; the loader rejects missing columns, duplicates, and
OHLC inconsistencies).

```bash
python run_backtest.py /path/to/MES_1m.csv --instrument MES \
    --ema-ticks 2 --expiry-min 60 --stop-ticks 8 --target-ticks 16 \
    --session-start 13:30 --session-end 20:00 \
    --max-trades-per-day 2 --cooldown-min 30 \
    --max-spread-ticks 2 --max-holding-min 120 --daily-loss-cap 500 \
    --contracts 1 --seed 42
```

Volume confirmation and cost overrides:

```bash
python run_backtest.py /path/to/MES_1m.csv --instrument MES \
    --volume-filter V3 --v1-mult 1.5 --v2-mult 2.0 --v4-pct 0.70 \
    --v5-mult 1.5 --v6-range-mult 1.2 \
    --tick-value 1.25
```

`--volume-filter` takes `none` (default) or `V1`–`V6` (see below);
`--tick-value` overrides USD/tick without editing `costs.py`.

All strategy tunables (X = EMA proximity ticks, N = expiry minutes, stop/target
distances, session window) are constructor/CLI arguments — nothing is hardcoded,
so the same CLI can be driven by a grid-search wrapper later.

Output is a JSON metrics block: `trades, net_pnl, sharpe, max_drawdown,
profit_factor, win_rate, avg_winner, avg_loser`.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The suite proves the three core guarantees:

1. **No lookahead** (`test_no_lookahead.py`) — the retest signal produced at the
   close of bar 00:18 fills at the **open of bar 00:19**; mutating the entry
   bar's high/low/close does not change the fill; the signal bar's close is
   provably not used as the fill; a 15m close back inside the level invalidates
   the setup (wicks don't, closes do).
2. **Deterministic replay** (`test_determinism.py`) — same seed + same data
   produces byte-identical trade lists across independent engine instances.
3. **P&L accounting with costs** (`test_pnl.py`) — hand-computed expectations:
   slippage 1 tick per side applied adversely to every fill, commission
   $1.25/side ($2.50 round trip), stop-first tie-break when a bar covers both
   stop and target, guard behavior (max trades/day, session window).
4. **Volume filter gates** (`test_volume_filter.py`) — each of V1–V6 blocks
   and allows correctly; the breakout bar is evaluated against history
   *excluding itself* (a dedicated test discriminates check-before-record
   ordering); a V3–V6 rejection leaves the setup in `RETEST_WATCH` so later
   bars may still trigger.
5. **Invalidation edge case** (`test_invalidation_opposite.py`) — a 15m close
   that invalidates one direction and closes beyond the opposite level starts
   the opposite setup on the same bar; plain invalidations still go idle;
   the V1/V2 gate applies to the opposite breakout too.
6. **Spread & CLI** (`test_spread_and_cli.py`) — gappy/misaligned spread
   series never raise (ffill only, constant fallback); `--tick-value`
   overrides USD/tick end-to-end; `--volume-filter` is wired through.

## Engine semantics (the reproducibility contract)

Per 1-minute bar, in strict order:

1. Pending entry (from the previous closed bar) fills at this bar's **open**
   plus adverse slippage, after re-checking all shared guards.
2. Open position is managed using this bar's OHLC only: stop/target intrabar
   checks (stop-first tie-break), then max-holding-time exit at the close.
3. The bar is closed and handed to the strategy; any signal becomes a pending
   entry for the **next** bar.

Multi-timeframe bars (2m, 15m) are aggregated incrementally from closed 1m bars
inside the strategy — a higher-timeframe bar is only visible once complete.
PDH/PDL come from the prior **UTC calendar day**, known only after it completes.

## Strategy #1 FSM

`IDLE → RETEST_WATCH` (15m close beyond PDH/PDL; V1/V2 volume gate may apply)
`→ ENTRY` (closed 2m bar touches the level **and/or** comes within X ticks of
the 13 EMA of 2m closes; V3–V6 volume gate may apply). Side exits:
`INVALIDATION` (15m **close** back inside the level) and `EXPIRY` (no retest
within N minutes). A 15m close that invalidates one direction **and** closes
beyond the opposite level starts the opposite setup on that same bar.

Shared guards (all configurable): max trades/day, cooldown after a loss,
session window, max-spread skip, max holding time, daily loss cap.

## Volume confirmation (`--volume-filter`, default `none`)

| Filter | Definition | Gates |
|---|---|---|
| V1 | Breakout 15m volume ≥ 1.5× 20-day median of the same 15m clock slot | Breakout |
| V2 | Breakout 15m volume ≥ 2.0× mean of prior ten 15m bars | Breakout |
| V3 | Signal 2m bar volume > mean of prior twenty 2m bars | Entry |
| V4 | Pullback 2m per-minute volume < 70% of breakout bar's per-minute volume | Entry |
| V5 | Signal 2m bar volume ≥ 1.5× 20-day median of the same 2m clock slot | Entry |
| V6 | V3 holds **and** signal range ≥ 1.2× median 2m range of prior 20 bars | Entry |

All multipliers are constructor/CLI parameters. Gates read closed bars only
and are evaluated *before* the current bar joins its own history. Short
history → the filter abstains (no signal). A V3–V6 rejection keeps the setup
in `RETEST_WATCH`; it does not cancel it.

## `# UNDEFINED:` markers (conservative interpretations chosen)

Search the code for `# UNDEFINED:` — each marks an ambiguity resolved
conservatively rather than assumed:

- **Tick values** verified against CME specs 2026-10-03 (MES $1.25, MNQ $0.50,
  MGC $1.00 per tick); override per-run with `--tick-value`.
- **Stop/target anchor**: distances anchor to the signal bar's close (fill is
  the next open; anchoring to the close is deterministic and conservative).
- **Stop/target same-bar conflict**: stop fills first (worst case).
- **Gap through stop**: fills at the bar open (worse), never at the stop price.
- **Spread guard**: OHLCV has no quotes; a constant 1-tick spread is assumed,
  or inject real spreads via `Engine.set_spread_series()` (reindexed +
  forward-filled only — backfill would leak future spreads; leading gaps fall
  back to the constant).
- **Sharpe**: daily net P&L, annualized with √252; every UTC day between the
  first and last trade contributes, no-trade days count as 0 (conservative).
- **PDH/PDL**: prior UTC calendar day, full session.
- **Volume gates**: evaluated before the current bar joins its own history;
  short history abstains; V3–V6 rejection keeps `RETEST_WATCH`.
