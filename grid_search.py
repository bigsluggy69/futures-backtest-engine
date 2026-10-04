"""Grid search over the engine's real tunables (research layer).

Imports the Engine API directly (no shelling out). Every combination of the
cartesian product is run and recorded: NOTHING is filtered or ranked here, the
full table goes to CSV. That is the anti-cherry-picking contract.

Only parameters the engine actually supports can be gridded:
    ema-ticks, expiry-min, stop-ticks, target-ticks, volume-filter,
    v1-mult, v2-mult, v4-pct, v5-mult, v6-range-mult,
    session-start, session-end, max-trades-per-day, cooldown-min
ATR-based stops and trailing stops do NOT exist in the engine yet, so they are
rejected rather than silently ignored.

Determinism: same data + same grid + same fixed args (incl. seed) produces a
byte-identical CSV. Rows are in canonical cartesian order (parameter order is
fixed by PARAM_NAMES, not by dict order; the rightmost parameter varies
fastest), values are formatted explicitly, and no timestamps or timings are
written. `--workers N` cannot change the output: results are collected in
submission order and every run is a pure function of its inputs.

Usage (walk-forward is the default mode):
    python grid_search.py data.csv --train-days 60 --validate-days 20 \
        --test-days 20 --top-k 20 --out results/
    python grid_search.py data.csv --grid-only --out results/   # in-sample only
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import sys
from typing import Any, Optional

import pandas as pd

from backtest.costs import INSTRUMENTS
from backtest.data import load_1m_csv
from backtest.engine import Engine, EngineConfig
from backtest.metrics import compute_metrics
from backtest.strategies.pdh_pdl_breakout import PdhPdlBreakoutRetest

# --------------------------------------------------------------------------
# Parameter registry. Names are the CLI flags with '-' -> '_'.
# --------------------------------------------------------------------------

_VOLUME_FILTERS = ("none", "V1", "V2", "V3", "V4", "V5", "V6")
_HHMM = re.compile(r"^(\d{2}):(\d{2})$")


def _as_float(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"expected a number, got {v!r}")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"expected a finite number, got {v!r}")
    return f


def _as_int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"expected an integer, got {v!r}")
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)


def _as_volume_filter(v: Any) -> str:
    if v not in _VOLUME_FILTERS:
        raise ValueError(f"volume-filter must be one of {_VOLUME_FILTERS}, got {v!r}")
    return v


def _as_hhmm(v: Any) -> str:
    m = _HHMM.match(v) if isinstance(v, str) else None
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"expected 'HH:MM' (UTC), got {v!r}")
    return v


# name -> (caster, default). Order here IS the canonical column / product order.
_REGISTRY: dict[str, tuple] = {
    "ema_ticks": (_as_float, 2.0),
    "expiry_min": (_as_int, 60),
    "stop_ticks": (_as_float, 8.0),
    "target_ticks": (_as_float, 16.0),
    "volume_filter": (_as_volume_filter, "none"),
    "v1_mult": (_as_float, 1.5),
    "v2_mult": (_as_float, 2.0),
    "v4_pct": (_as_float, 0.70),
    "v5_mult": (_as_float, 1.5),
    "v6_range_mult": (_as_float, 1.2),
    "session_start": (_as_hhmm, "13:30"),
    "session_end": (_as_hhmm, "20:00"),
    "max_trades_per_day": (_as_int, 2),
    "cooldown_min": (_as_int, 30),
}
PARAM_NAMES = tuple(_REGISTRY)
DEFAULTS = {k: v[1] for k, v in _REGISTRY.items()}  # identical to run_backtest.py

METRIC_NAMES = ("trades", "net_pnl", "sharpe", "max_drawdown", "profit_factor",
                "win_rate", "avg_winner", "avg_loser")
RESULT_COLUMNS = ("combo_id",) + PARAM_NAMES + METRIC_NAMES

# Reasonable default grid: 3*2*2*2*2*2 = 96 combinations.
DEFAULT_GRID = {
    "ema-ticks": [1, 2, 4],
    "expiry-min": [30, 60],
    "stop-ticks": [8, 12],
    "target-ticks": [16, 24],
    "volume-filter": ["none", "V3"],
    "max-trades-per-day": [1, 2],
}

# Engine/CLI settings that exist but are deliberately NOT grid dimensions.
FIXED_DEFAULTS = {
    "instrument": "MES",
    "tick_value": None,
    "max_spread_ticks": 2,
    "max_holding_min": 120,
    "daily_loss_cap": 500.0,
    "contracts": 1,
    "seed": 42,
}

_NOT_GRIDDABLE = {k.replace("_", "-") for k in FIXED_DEFAULTS}
_NOT_IMPLEMENTED_HINT = ("the engine has no ATR-based or trailing stops yet; "
                         "only fixed tick stops/targets exist")

# volume_filter -> the one multiplier it reads. Others are inert for that filter.
_ACTIVE_MULT = {"V1": "v1_mult", "V2": "v2_mult", "V4": "v4_pct",
                "V5": "v5_mult", "V6": "v6_range_mult"}
_MULT_PARAMS = ("v1_mult", "v2_mult", "v4_pct", "v5_mult", "v6_range_mult")


# --------------------------------------------------------------------------
# Grid handling
# --------------------------------------------------------------------------

def _norm_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_").lower()


def canonicalize_grid(grid: dict) -> dict[str, list]:
    """Validate a flag -> list-of-values dict and return it keyed in canonical
    parameter order with values coerced to canonical types. Parameters absent
    from the grid are held at their default (single value)."""
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty dict of flag -> list of values")
    out: dict[str, list] = {}
    for raw_key, values in grid.items():
        key = _norm_key(str(raw_key))
        if key not in _REGISTRY:
            hint = ""
            if key.replace("_", "-") in _NOT_GRIDDABLE:
                hint = (" It is a fixed engine setting: pass it as a CLI flag "
                        "(e.g. --max-holding-min), not as a grid dimension.")
            elif any(w in key for w in ("atr", "trail")):
                hint = f" ({_NOT_IMPLEMENTED_HINT})"
            supported = ", ".join(k.replace("_", "-") for k in PARAM_NAMES)
            raise ValueError(f"unsupported grid parameter {raw_key!r}.{hint} "
                             f"Supported: {supported}")
        if key in out:
            raise ValueError(f"grid parameter {raw_key!r} given more than once")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"grid[{raw_key!r}] must be a non-empty list of values")
        caster = _REGISTRY[key][0]
        try:
            cast = [caster(v) for v in values]
        except ValueError as exc:
            raise ValueError(f"grid[{raw_key!r}]: {exc}") from None
        if len(set(cast)) != len(cast):
            raise ValueError(f"grid[{raw_key!r}] contains duplicate values; "
                             "duplicates would silently inflate the trial count")
        out[key] = cast
    return {k: out[k] for k in PARAM_NAMES if k in out}


def _full_axes(canon: dict[str, list]) -> list[list]:
    return [canon.get(k, [DEFAULTS[k]]) for k in PARAM_NAMES]


def grid_size(grid: dict) -> int:
    n = 1
    for axis in _full_axes(canonicalize_grid(grid)):
        n *= len(axis)
    return n


def iter_grid(grid: dict):
    """Yield (combo_id, params) for the FULL cartesian product, canonical order."""
    axes = _full_axes(canonicalize_grid(grid))
    for combo_id, values in enumerate(itertools.product(*axes)):
        yield combo_id, dict(zip(PARAM_NAMES, values))


def effective_key(params: dict) -> tuple:
    """Behavioural identity of a parameter set. A volume multiplier that the
    chosen volume filter never reads cannot change any trade, so combinations
    differing only in inert multipliers are the same strategy. Used for
    de-duplicating top-K (the CSV still contains every combination).

    # UNDEFINED: whether inert-multiplier duplicates should count as separate
    # trials. Conservative: keep every row in the CSV (full cartesian), but
    # never let duplicates crowd distinct strategies out of the top-K, and
    # report both counts so the real number of trials is visible.
    """
    active = _ACTIVE_MULT.get(params["volume_filter"])
    return tuple(
        None if (k in _MULT_PARAMS and k != active) else params[k]
        for k in PARAM_NAMES
    )


def effective_grid_size(grid: dict) -> int:
    return len({effective_key(p) for _, p in iter_grid(grid)})


# --------------------------------------------------------------------------
# Single run (direct Engine API)
# --------------------------------------------------------------------------

def resolve_fixed(overrides: Optional[dict] = None) -> dict:
    fixed = dict(FIXED_DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in FIXED_DEFAULTS:
            raise ValueError(f"unknown fixed setting {k!r}; known: {sorted(FIXED_DEFAULTS)}")
        fixed[k] = v
    if fixed["instrument"] not in INSTRUMENTS:
        raise ValueError(f"unknown instrument {fixed['instrument']!r}")
    return fixed


def build_engine(params: dict, fixed: dict, slippage_mult: float = 1.0) -> Engine:
    """Construct exactly what run_backtest.py constructs for the same values.
    (tests/test_grid_search.py asserts parity against the real CLI.)"""
    spec = INSTRUMENTS[fixed["instrument"]]
    if fixed["tick_value"] is not None:
        spec = dataclasses.replace(spec, tick_value=fixed["tick_value"])
    if slippage_mult != 1.0:
        # Stress: scale adverse slippage per side; commissions unchanged.
        spec = dataclasses.replace(
            spec, slippage_ticks_per_side=spec.slippage_ticks_per_side * slippage_mult)
    cfg = EngineConfig(
        max_trades_per_day=params["max_trades_per_day"],
        cooldown_after_loss_min=params["cooldown_min"],
        session_start=params["session_start"],
        session_end=params["session_end"],
        max_spread_ticks=fixed["max_spread_ticks"],
        max_holding_min=fixed["max_holding_min"],
        daily_loss_cap=fixed["daily_loss_cap"],
        contracts=fixed["contracts"],
        seed=fixed["seed"],
    )
    strat = PdhPdlBreakoutRetest(
        spec=spec,
        ema_proximity_ticks=params["ema_ticks"],
        expiry_min=params["expiry_min"],
        stop_ticks=params["stop_ticks"],
        target_ticks=params["target_ticks"],
        volume_filter=params["volume_filter"],
        v1_mult=params["v1_mult"],
        v2_mult=params["v2_mult"],
        v4_pct=params["v4_pct"],
        v5_mult=params["v5_mult"],
        v6_range_mult=params["v6_range_mult"],
    )
    return Engine(strat, spec, cfg)


def normalize_metrics(m: dict) -> dict:
    """Plain int/float values in fixed order (compute_metrics returns numpy
    scalars, whose repr would corrupt a CSV)."""
    out = {}
    for k in METRIC_NAMES:
        out[k] = int(m[k]) if k == "trades" else float(m[k])
    return out


def run_single(bars: pd.DataFrame, params: dict, fixed: dict,
               slippage_mult: float = 1.0,
               scored_from: Optional[pd.Timestamp] = None) -> dict:
    """Run one parameter set on `bars` with a FRESH engine and return metrics.

    `scored_from`: only trades ENTERED at or after this timestamp are scored.
    The caller may prepend earlier (strictly past) bars as indicator warmup;
    trades taken during warmup are discarded.

    # UNDEFINED: a warmup trade still open at `scored_from` can block an entry
    # in the first scored bars (the engine allows one position). Conservative:
    # left as-is (it can only suppress, never add, a trade).
    """
    trades = build_engine(params, fixed, slippage_mult).run(bars)
    if scored_from is not None:
        trades = [t for t in trades if t.entry_time >= scored_from]
    return normalize_metrics(compute_metrics(trades))


# --------------------------------------------------------------------------
# Batch execution (ordered, optionally multi-process)
# --------------------------------------------------------------------------

_W: dict = {}


def _init_worker(bars, fixed, slippage_mult, scored_from):
    _W.update(bars=bars, fixed=fixed, slip=slippage_mult, scored_from=scored_from)


def _work(item):
    combo_id, params = item
    return run_single(_W["bars"], params, _W["fixed"], _W["slip"], _W["scored_from"])


def run_many(bars: pd.DataFrame, items: list, fixed: dict, *,
             slippage_mult: float = 1.0,
             scored_from: Optional[pd.Timestamp] = None,
             workers: int = 1) -> list[dict]:
    """Run [(combo_id, params), ...]; rows come back in input order."""
    if workers > 1 and len(items) > 1:
        with multiprocessing.Pool(
                workers, initializer=_init_worker,
                initargs=(bars, fixed, slippage_mult, scored_from)) as pool:
            metrics = pool.map(_work, items, chunksize=1)  # ordered
    else:
        metrics = [run_single(bars, p, fixed, slippage_mult, scored_from)
                   for _, p in items]
    return [{"combo_id": cid, **params, **m}
            for (cid, params), m in zip(items, metrics)]


def run_grid(bars: pd.DataFrame, grid: dict, fixed: Optional[dict] = None, *,
             slippage_mult: float = 1.0,
             scored_from: Optional[pd.Timestamp] = None,
             workers: int = 1) -> list[dict]:
    """The FULL cartesian product, every combination, no filtering."""
    return run_many(bars, list(iter_grid(grid)), resolve_fixed(fixed),
                    slippage_mult=slippage_mult, scored_from=scored_from,
                    workers=workers)


def split_row(row: dict) -> tuple[dict, dict]:
    return ({k: row[k] for k in PARAM_NAMES}, {k: row[k] for k in METRIC_NAMES})


# --------------------------------------------------------------------------
# Byte-stable output
# --------------------------------------------------------------------------

def fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def write_csv(rows: list[dict], columns, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt_value(r.get(c)) for c in columns])


def write_results_csv(rows: list[dict], path: str) -> None:
    write_csv(rows, RESULT_COLUMNS, path)


def json_clean(obj: Any) -> Any:
    """Make nested data JSON-safe and stable (inf -> 'inf', numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return json_clean(obj.item())
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj in (float("inf"), float("-inf")):
            return "inf" if obj > 0 else "-inf"
        return obj
    return str(obj)


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_clean(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_grid_arg(arg: Optional[str]) -> dict:
    if arg is None:
        return dict(DEFAULT_GRID)
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(arg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grid search + walk-forward validation (research only)")
    p.add_argument("csv", help="1m OHLCV CSV (timestamp_utc,open,high,low,close,volume)")
    p.add_argument("--grid", default=None,
                   help="JSON file path or inline JSON: {flag: [values]} "
                        "(default: a 96-combination grid)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--grid-only", action="store_true",
                   help="run the grid once over ALL data and write the full CSV "
                        "(in-sample; for inspection, not for selecting parameters)")
    # Walk-forward geometry (calendar days, UTC).
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--validate-days", type=int, default=20)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument("--step-days", type=int, default=None,
                   help="fold advance (default: --test-days, i.e. test windows tile)")
    p.add_argument("--anchored", action="store_true",
                   help="anchored (expanding) train window; default is rolling")
    # Selection (declared before any validate/test run).
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--selection-metric", default="sharpe",
                   choices=["sharpe", "profit_factor"],
                   help="net_pnl is intentionally not available")
    p.add_argument("--min-split-trades", type=int, default=10,
                   help="min trades in train AND validate for a sign of edge to count")
    p.add_argument("--min-test-trades", type=int, default=30)
    p.add_argument("--slippage-multiplier", type=float, default=2.0)
    p.add_argument("--warmup-days", type=int, default=30,
                   help="strictly-past bars prepended to each window for indicator "
                        "warmup (their trades are discarded)")
    p.add_argument("--workers", type=int, default=1,
                   help="processes for grid runs; never changes the output")
    # Fixed engine settings (same names/defaults as run_backtest.py).
    p.add_argument("--instrument", default=FIXED_DEFAULTS["instrument"],
                   choices=list(INSTRUMENTS))
    p.add_argument("--tick-value", type=float, default=None)
    p.add_argument("--max-spread-ticks", type=int, default=FIXED_DEFAULTS["max_spread_ticks"])
    p.add_argument("--max-holding-min", type=int, default=FIXED_DEFAULTS["max_holding_min"])
    p.add_argument("--daily-loss-cap", type=float, default=FIXED_DEFAULTS["daily_loss_cap"])
    p.add_argument("--contracts", type=int, default=FIXED_DEFAULTS["contracts"])
    p.add_argument("--seed", type=int, default=FIXED_DEFAULTS["seed"])
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        grid = canonicalize_grid(load_grid_arg(args.grid))
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    fixed = resolve_fixed({
        "instrument": args.instrument, "tick_value": args.tick_value,
        "max_spread_ticks": args.max_spread_ticks,
        "max_holding_min": args.max_holding_min,
        "daily_loss_cap": args.daily_loss_cap, "contracts": args.contracts,
        "seed": args.seed,
    })

    bars = load_1m_csv(args.csv)
    data_hash = file_sha256(args.csv)
    os.makedirs(args.out, exist_ok=True)
    n, n_eff = grid_size(grid), effective_grid_size(grid)

    if args.grid_only:
        rows = run_grid(bars, grid, fixed, workers=args.workers)
        write_results_csv(rows, os.path.join(args.out, "grid_results.csv"))
        write_json({"mode": "grid_only", "grid": grid, "fixed": fixed,
                    "data_sha256": data_hash, "grid_size": n,
                    "effective_grid_size": n_eff,
                    "note": "in-sample over all data; not for parameter selection"},
                   os.path.join(args.out, "run_manifest.json"))
        print(f"grid-only: {len(rows)} combinations -> "
              f"{os.path.join(args.out, 'grid_results.csv')} (in-sample)")
        return 0

    import walk_forward as wf  # lazy: walk_forward imports this module

    try:
        criteria = wf.SelectionCriteria(
            top_k=args.top_k, selection_metric=args.selection_metric,
            min_split_trades=args.min_split_trades,
            min_test_trades=args.min_test_trades,
            slippage_multiplier=args.slippage_multiplier,
            warmup_days=args.warmup_days)
        report = wf.run_walk_forward(
            bars, grid, fixed,
            train_days=args.train_days, validate_days=args.validate_days,
            test_days=args.test_days,
            step_days=args.step_days if args.step_days is not None else args.test_days,
            anchored=args.anchored, criteria=criteria, out_dir=args.out,
            workers=args.workers, data_sha256=data_hash)
    except ValueError as exc:
        parser.error(str(exc))

    agg = report["aggregate"]
    print(f"walk-forward: {agg['n_folds']} folds, grid {n} combos "
          f"({n_eff} distinct), {agg['n_selected']} folds selected a candidate, "
          f"{agg['n_test_sample_ok']} with >= {criteria.min_test_trades} test trades")
    print(f"results in {args.out}/ (walkforward_summary.csv, walkforward_report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
