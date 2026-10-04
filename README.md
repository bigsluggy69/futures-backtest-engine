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
├── tests/                            # unittest suite (39 engine + 50 research-layer tests)
├── run_backtest.py                   # CLI entry point
├── grid_search.py                    # research layer: full-cartesian grid + CLI
└── walk_forward.py                   # research layer: train/validate/test selection
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

## Research layer: grid search + walk-forward

```bash
python grid_search.py data.csv --train-days 60 --validate-days 20 \
    --test-days 20 --top-k 20 --out results/          # rolling (default)
python grid_search.py data.csv --anchored --workers 4 --out results/
python grid_search.py data.csv --grid my_grid.json --out results/
python grid_search.py data.csv --grid-only --out results/   # in-sample, inspection only
```

`--grid` takes a JSON file or inline JSON of `{flag: [values]}`. Only what the
engine supports can be gridded: `ema-ticks, expiry-min, stop-ticks,
target-ticks, volume-filter, v1-mult, v2-mult, v4-pct, v5-mult, v6-range-mult,
session-start, session-end, max-trades-per-day, cooldown-min`. ATR stops and
trailing stops do not exist and are rejected, as are fixed engine settings
(`--max-holding-min`, `--daily-loss-cap`, ...; pass those as flags). The default
grid has 96 combinations. The grid imports the Engine API directly (a fresh
`Engine` per combination) and a parity test checks it against `run_backtest.py`.

**Anti-cherry-picking.** The grid layer never filters or ranks: every
combination of the cartesian product is a row in the CSV, including zero-trade
and losing ones. Same data + grid + seed gives a byte-identical CSV
(`--workers` cannot change the bytes).

**Walk-forward procedure** (per fold; calendar-day windows, half-open, disjoint):

1. Full grid on **train** (`fold_NN_train_grid.csv`, unfiltered).
2. Top-K distinct strategies by `--selection-metric` (`sharpe` or
   `profit_factor`; `net_pnl` is rejected). Combos with fewer than
   `--min-split-trades` trades are not candidates.
3. Re-run the K on **validate**.
4. Select ONE using train + validate only (`fold_NN_candidates.csv`):
   (a) net P&L positive in BOTH train and validate (consistent sign of edge);
   (b) net P&L still positive in both under doubled slippage
   (`--slippage-multiplier`, commissions unchanged). Qualified candidates are
   ranked by their **worst** split on the selection metric, so a consistent
   performer beats a lopsided or high-return one.
5. Run only that candidate on **test** (`walkforward_summary.csv`,
   `walkforward_report.json`). (c) needs >= `--min-test-trades` (30) test
   trades; this is a **verdict gate** (`INCONCLUSIVE_LOW_TEST_SAMPLE`), not a
   selector: choosing a different candidate after seeing the test would be
   selecting on the test set. If nothing qualifies the fold reports
   `NO_QUALIFIED_CANDIDATE` and the test window is never run.

`selection_criteria.json` (criteria, folds, grid) is written before any run.
Each stage is handed only bars before its window end; `--warmup-days` of
strictly-past bars are prepended for PDH/PDL and the 20-day volume medians, and
trades taken during warmup are discarded.

Cost: the engine runs ~9k 1m bars/s, so a 60-day 24h window is ~10 s per
combination. Use `--workers`, and size the grid with that in mind.

Additional `# UNDEFINED:` choices in the research layer (conservative):

- **Windows** are calendar days (UTC), not trading days; a final partial test
  window is dropped. Default `--step-days` = `--test-days` so test windows tile;
  a smaller step makes them overlap and the report flags that totals double-count.
- **Consistent sign of edge** = positive net P&L in both splits with at least
  `--min-split-trades` (default 10) trades each. Consistently negative is not an
  edge. Set it to 0 to drop the trade floor.
- **Inert multipliers**: `v1-mult` etc. do nothing under other volume filters.
  The CSV keeps every combination, but top-K counts distinct strategies and the
  report gives both `grid_size` and `effective_grid_size` (the true trial count).
- **Slippage stress** runs only for candidates that already pass (a); it cannot
  change who qualifies, since qualifying needs (a) and (b).
- **Warmup trades**: a warmup trade still open at the window start can block an
  entry in the first bars (can only suppress a trade, never add one).

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
