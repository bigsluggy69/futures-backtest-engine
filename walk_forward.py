"""Walk-forward validation on top of grid_search.

Per fold (windows are half-open [start, end) in UTC calendar days and are
disjoint by construction: train_end == validate_start, validate_end == test_start):

  1. TRAIN     full grid on the train window (every combination, saved to CSV).
  2. TOP-K     the K best distinct strategies by the selection metric on train.
  3. VALIDATE  re-run just those K on the validate window.
  4. SELECT    ONE candidate, using only train + validate information.
  5. TEST      run the single selected candidate on the test window and report.

Selection criteria (declared up front and written to selection_criteria.json
BEFORE any validate or test run happens):

  (a) consistent sign of edge across train and validate: net P&L after costs is
      positive in BOTH splits (and each split has >= min_split_trades trades so
      a one-trade fluke has no sign).
  (b) survives doubled slippage: net P&L stays positive on BOTH train and
      validate when per-side slippage is multiplied by `slippage_multiplier`.
  (c) >= min_test_trades (30) trades in the TEST sample. This one cannot be a
      selection filter: it is only knowable after the test run, and swapping to
      a different candidate because of what the test showed would be selecting
      on the test set. It is therefore a verdict gate: a selected candidate
      with fewer test trades is reported INCONCLUSIVE_LOW_TEST_SAMPLE, never
      replaced.

Highest backtest return is NOT a criterion anywhere: net_pnl is not an allowed
selection metric, and qualified candidates are ranked by their WORST split on
the selection metric (min over train/validate), which rewards consistency.

No lookahead: each stage is handed only bars strictly before its window end;
warmup bars (for PDH/PDL and the 20-day volume slot medians) are strictly
before the window start, and trades taken during warmup are discarded. A
later window can never influence an earlier window's results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import grid_search as gs

DAY = pd.Timedelta(days=1)
_SELECTION_METRICS = ("sharpe", "profit_factor")


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionCriteria:
    top_k: int = 20
    selection_metric: str = "sharpe"
    min_split_trades: int = 10
    min_test_trades: int = 30
    slippage_multiplier: float = 2.0
    warmup_days: int = 30

    def __post_init__(self):
        if self.selection_metric not in _SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {_SELECTION_METRICS}; backtest "
                f"return (net_pnl) is deliberately not allowed, got "
                f"{self.selection_metric!r}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.min_split_trades < 0:
            raise ValueError("min_split_trades must be >= 0")
        if self.min_test_trades < 1:
            raise ValueError("min_test_trades must be >= 1")
        if not self.slippage_multiplier > 1.0:
            raise ValueError("slippage_multiplier must be > 1 (it is a stress)")
        if self.warmup_days < 0:
            raise ValueError("warmup_days must be >= 0")

    def describe(self) -> dict:
        return {
            "top_k": self.top_k,
            "selection_metric": self.selection_metric,
            "min_split_trades": self.min_split_trades,
            "min_test_trades": self.min_test_trades,
            "slippage_multiplier": self.slippage_multiplier,
            "warmup_days": self.warmup_days,
            "a": "net P&L > 0 in train AND validate, each with >= "
                 "min_split_trades trades",
            "b": "net P&L > 0 on train AND validate with per-side slippage x "
                 "slippage_multiplier (commissions unchanged)",
            "c": "test sample >= min_test_trades trades; verdict gate only "
                 "(cannot be a selection filter without peeking at test)",
            "ranking": "qualified candidates ranked by min(train, validate) of "
                       "selection_metric, ties by lower combo_id; backtest "
                       "return is not a criterion",
        }


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validate_start: pd.Timestamp
    validate_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {k: (v if k == "index" else v.isoformat())
                for k, v in self.__dict__.items()}


def make_folds(data_start: pd.Timestamp, data_end_exclusive: pd.Timestamp, *,
               train_days: int, validate_days: int, test_days: int,
               step_days: int, anchored: bool) -> list[Fold]:
    """Calendar-day (UTC) folds.

    rolling : train = [S + i*step, S + i*step + train_days)
    anchored: train = [S, S + i*step + train_days)          (expanding)
    validate and test follow train back-to-back in both modes.

    # UNDEFINED: windows are CALENDAR days, not trading days (weekends and
    # holidays count toward the lengths). Conservative: no assumption about a
    # trading calendar is baked in.
    # UNDEFINED: a final partial window is dropped. Only folds whose full test
    # window fits inside the data are emitted.
    """
    for name, val in (("train_days", train_days), ("validate_days", validate_days),
                      ("test_days", test_days), ("step_days", step_days)):
        if not isinstance(val, int) or val < 1:
            raise ValueError(f"{name} must be a positive integer, got {val!r}")
    start = pd.Timestamp(data_start)
    if start.tzinfo is None:
        raise ValueError("data_start must be timezone-aware (UTC)")
    start = start.normalize()
    end = pd.Timestamp(data_end_exclusive)

    folds: list[Fold] = []
    i = 0
    while True:
        shift = i * step_days
        train_end = start + (train_days + shift) * DAY
        train_start = start if anchored else start + shift * DAY
        validate_end = train_end + validate_days * DAY
        test_end = validate_end + test_days * DAY
        if test_end > end:
            break
        fold = Fold(i, train_start, train_end, train_end, validate_end,
                    validate_end, test_end)
        _check_fold(fold)
        folds.append(fold)
        i += 1
    return folds


def _check_fold(f: Fold) -> None:
    ok = (f.train_start < f.train_end <= f.validate_start < f.validate_end
          <= f.test_start < f.test_end)
    if not ok:
        raise RuntimeError(f"fold {f.index} windows are not ordered/disjoint: {f}")


def oos_windows_overlap(folds: list[Fold]) -> bool:
    """True if any two folds' TEST windows overlap (step_days < test_days)."""
    return any(a.test_end > b.test_start for a, b in zip(folds, folds[1:]))


def bars_for_window(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                    warmup_days: int) -> pd.DataFrame:
    """Bars in [start - warmup, end). Nothing at or after `end` is ever included."""
    ts = bars["timestamp_utc"]
    i0 = int(ts.searchsorted(start - warmup_days * DAY, side="left"))
    i1 = int(ts.searchsorted(end, side="left"))
    return bars.iloc[i0:i1]


# --------------------------------------------------------------------------
# Selection logic (pure: operates on metrics only, no data, no engine)
# --------------------------------------------------------------------------

def criterion_a(cand: dict, criteria: SelectionCriteria) -> bool:
    """Consistent sign of edge: positive net P&L in train AND validate."""
    t, v = cand["train"], cand["validate"]
    return bool(t["trades"] >= criteria.min_split_trades
                and v["trades"] >= criteria.min_split_trades
                and t["net_pnl"] > 0 and v["net_pnl"] > 0)


def criterion_b(cand: dict) -> Optional[bool]:
    """Survives doubled slippage; None if the stress runs were not performed."""
    t2, v2 = cand.get("train_2x"), cand.get("validate_2x")
    if t2 is None or v2 is None:
        return None
    return bool(t2["net_pnl"] > 0 and v2["net_pnl"] > 0)


def select_one(candidates: list[dict], criteria: SelectionCriteria):
    """Pick ONE candidate from train/validate/stress metrics alone.

    Returns (chosen_or_None, evaluated) where `evaluated` mirrors the input
    order with a_consistent_sign / b_survives_2x / qualified / score added.
    Qualified candidates are ranked by the WORST of their train and validate
    selection-metric values (consistency), never by return.
    """
    metric = criteria.selection_metric
    evaluated = []
    for c in candidates:
        e = dict(c)
        e["a_consistent_sign"] = criterion_a(c, criteria)
        e["b_survives_2x"] = criterion_b(c)
        e["qualified"] = bool(e["a_consistent_sign"] and e["b_survives_2x"] is True)
        e["score"] = min(c["train"][metric], c["validate"][metric])
        evaluated.append(e)
    qualified = [e for e in evaluated if e["qualified"]]
    if not qualified:
        return None, evaluated
    chosen = sorted(qualified, key=lambda e: (-e["score"], e["combo_id"]))[0]
    return chosen, evaluated


def verdict_for_test(test_metrics: dict, criteria: SelectionCriteria) -> tuple[bool, str]:
    """Criterion (c) plus a plain profit/loss label. Returns (sample_ok, verdict)."""
    sample_ok = test_metrics["trades"] >= criteria.min_test_trades
    if not sample_ok:
        return False, "INCONCLUSIVE_LOW_TEST_SAMPLE"
    return True, ("TEST_PROFITABLE" if test_metrics["net_pnl"] > 0
                  else "TEST_UNPROFITABLE")


# --------------------------------------------------------------------------
# Stages (each is handed only the bars it is allowed to see)
# --------------------------------------------------------------------------

def top_k_candidates(rows: list[dict], criteria: SelectionCriteria) -> list[dict]:
    """K best DISTINCT strategies on train by the selection metric.

    Combinations with no trades are never candidates; combinations with fewer
    than min_split_trades trades are excluded (their metric is noise). Rows
    that differ only in inert volume multipliers are the same strategy and
    appear once. Ties break on combo_id so the order is deterministic.
    """
    metric = criteria.selection_metric
    floor = max(1, criteria.min_split_trades)
    eligible = [r for r in rows if r["trades"] >= floor]
    ranked = sorted(eligible, key=lambda r: (-r[metric], r["combo_id"]))
    seen, out = set(), []
    for r in ranked:
        params, metrics = gs.split_row(r)
        key = gs.effective_key(params)
        if key in seen:
            continue
        seen.add(key)
        out.append({"combo_id": r["combo_id"], "params": params, "train": metrics})
        if len(out) == criteria.top_k:
            break
    return out


def train_stage(bars, fold: Fold, grid: dict, fixed: dict,
                criteria: SelectionCriteria, workers: int = 1):
    window = bars_for_window(bars, fold.train_start, fold.train_end,
                             criteria.warmup_days)
    rows = gs.run_grid(window, grid, fixed, scored_from=fold.train_start,
                       workers=workers)
    return rows, top_k_candidates(rows, criteria)


def validate_stage(bars, fold: Fold, candidates: list[dict], fixed: dict,
                   criteria: SelectionCriteria, workers: int = 1) -> list[dict]:
    """Re-run the K candidates on validate; run the slippage stress (train and
    validate) only for those that already pass (a) -- it cannot change who is
    qualified, since qualified requires (a) AND (b)."""
    val_window = bars_for_window(bars, fold.validate_start, fold.validate_end,
                                 criteria.warmup_days)
    items = [(c["combo_id"], c["params"]) for c in candidates]
    val_rows = gs.run_many(val_window, items, fixed,
                           scored_from=fold.validate_start, workers=workers)
    out = []
    for c, r in zip(candidates, val_rows):
        e = dict(c)
        e["validate"] = gs.split_row(r)[1]
        out.append(e)

    passing = [e for e in out if criterion_a(e, criteria)]
    if passing:
        stress_items = [(e["combo_id"], e["params"]) for e in passing]
        train_window = bars_for_window(bars, fold.train_start, fold.train_end,
                                       criteria.warmup_days)
        slip = criteria.slippage_multiplier
        t2 = gs.run_many(train_window, stress_items, fixed, slippage_mult=slip,
                         scored_from=fold.train_start, workers=workers)
        v2 = gs.run_many(val_window, stress_items, fixed, slippage_mult=slip,
                         scored_from=fold.validate_start, workers=workers)
        for e, rt, rv in zip(passing, t2, v2):
            e["train_2x"] = gs.split_row(rt)[1]
            e["validate_2x"] = gs.split_row(rv)[1]
    return out


def holdout_stage(bars, fold: Fold, chosen: dict, fixed: dict,
               criteria: SelectionCriteria) -> dict:
    """Run the ONE selected candidate on the test window (plus an informational
    doubled-slippage run that plays no part in selection)."""
    window = bars_for_window(bars, fold.test_start, fold.test_end,
                             criteria.warmup_days)
    return {
        "test": gs.run_single(window, chosen["params"], fixed, 1.0,
                              fold.test_start),
        "test_2x": gs.run_single(window, chosen["params"], fixed,
                                 criteria.slippage_multiplier, fold.test_start),
    }


# --------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------

def _flat(prefix: str, metrics: Optional[dict]) -> dict:
    return {f"{prefix}_{m}": (None if metrics is None else metrics[m])
            for m in gs.METRIC_NAMES}


def _prefixed_cols(prefix: str) -> tuple:
    return tuple(f"{prefix}_{m}" for m in gs.METRIC_NAMES)


CANDIDATE_COLUMNS = (
    ("rank_train", "combo_id") + gs.PARAM_NAMES
    + _prefixed_cols("train") + _prefixed_cols("val")
    + _prefixed_cols("train2x") + _prefixed_cols("val2x")
    + ("a_consistent_sign", "b_survives_2x", "qualified", "score", "selected")
)

SUMMARY_COLUMNS = (
    ("fold", "train_start", "train_end", "validate_start", "validate_end",
     "test_start", "test_end", "grid_size", "effective_grid_size",
     "n_candidates", "n_pass_a", "n_qualified", "selected_combo_id")
    + gs.PARAM_NAMES
    + _prefixed_cols("train") + _prefixed_cols("val")
    + _prefixed_cols("test") + _prefixed_cols("test2x")
    + ("test_sample_ok", "verdict")
)


def _candidate_row(rank: int, e: dict, selected: bool) -> dict:
    row = {"rank_train": rank, "combo_id": e["combo_id"], **e["params"]}
    row.update(_flat("train", e["train"]))
    row.update(_flat("val", e["validate"]))
    row.update(_flat("train2x", e.get("train_2x")))
    row.update(_flat("val2x", e.get("validate_2x")))
    row.update(a_consistent_sign=e["a_consistent_sign"],
               b_survives_2x=e["b_survives_2x"], qualified=e["qualified"],
               score=e["score"], selected=selected)
    return row


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_walk_forward(bars: pd.DataFrame, grid: dict, fixed: dict, *,
                     train_days: int, validate_days: int, test_days: int,
                     step_days: int, anchored: bool,
                     criteria: SelectionCriteria, out_dir: str,
                     workers: int = 1,
                     data_sha256: Optional[str] = None) -> dict:
    grid = gs.canonicalize_grid(grid)
    fixed = gs.resolve_fixed(fixed)
    ts = bars["timestamp_utc"]
    data_start = ts.iloc[0].normalize()
    data_end = ts.iloc[-1].normalize() + DAY
    folds = make_folds(data_start, data_end, train_days=train_days,
                       validate_days=validate_days, test_days=test_days,
                       step_days=step_days, anchored=anchored)
    if not folds:
        have = (data_end - data_start).days
        raise ValueError(
            f"data spans {have} days but one fold needs train+validate+test = "
            f"{train_days + validate_days + test_days} days")

    n_grid, n_eff = gs.grid_size(grid), gs.effective_grid_size(grid)
    overlap = oos_windows_overlap(folds)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-registration: written before any validate/test run.
    gs.write_json({
        "criteria": criteria.describe(),
        "geometry": {"train_days": train_days, "validate_days": validate_days,
                     "test_days": test_days, "step_days": step_days,
                     "anchored": anchored},
        "folds": [f.as_dict() for f in folds],
        "grid": grid, "grid_size": n_grid, "effective_grid_size": n_eff,
        "fixed": fixed, "data_sha256": data_sha256,
        "oos_windows_overlap": overlap,
    }, os.path.join(out_dir, "selection_criteria.json"))

    summary_rows, fold_reports = [], []
    for fold in folds:
        tag = f"fold_{fold.index:02d}"
        train_rows, cands = train_stage(bars, fold, grid, fixed, criteria, workers)
        gs.write_results_csv(train_rows, os.path.join(out_dir, f"{tag}_train_grid.csv"))

        with_val = validate_stage(bars, fold, cands, fixed, criteria, workers)
        chosen, evaluated = select_one(with_val, criteria)

        cand_rows = [_candidate_row(i + 1, e, chosen is not None and e is chosen)
                     for i, e in enumerate(evaluated)]
        gs.write_csv(cand_rows, CANDIDATE_COLUMNS,
                     os.path.join(out_dir, f"{tag}_candidates.csv"))

        row = {"fold": fold.index, "train_start": fold.train_start.isoformat(),
               "train_end": fold.train_end.isoformat(),
               "validate_start": fold.validate_start.isoformat(),
               "validate_end": fold.validate_end.isoformat(),
               "test_start": fold.test_start.isoformat(),
               "test_end": fold.test_end.isoformat(),
               "grid_size": n_grid, "effective_grid_size": n_eff,
               "n_candidates": len(evaluated),
               "n_pass_a": sum(e["a_consistent_sign"] for e in evaluated),
               "n_qualified": sum(e["qualified"] for e in evaluated)}
        report = {"fold": fold.as_dict(), "n_candidates": len(evaluated),
                  "n_pass_a": row["n_pass_a"], "n_qualified": row["n_qualified"],
                  "candidates": evaluated}

        if chosen is None:
            row.update(test_sample_ok=None, verdict="NO_QUALIFIED_CANDIDATE")
            report.update(selected=None, verdict="NO_QUALIFIED_CANDIDATE")
        else:
            res = holdout_stage(bars, fold, chosen, fixed, criteria)
            sample_ok, verdict = verdict_for_test(res["test"], criteria)
            row.update(selected_combo_id=chosen["combo_id"], **chosen["params"])
            row.update(_flat("train", chosen["train"]))
            row.update(_flat("val", chosen["validate"]))
            row.update(_flat("test", res["test"]))
            row.update(_flat("test2x", res["test_2x"]))
            row.update(test_sample_ok=sample_ok, verdict=verdict)
            report.update(selected={"combo_id": chosen["combo_id"],
                                    "params": chosen["params"], **res},
                          test_sample_ok=sample_ok, verdict=verdict)
        summary_rows.append(row)
        fold_reports.append(report)
        print(f"{tag}: {len(cands)} candidates, {row['n_qualified']} qualified -> "
              f"{row['verdict']}", flush=True)

    gs.write_csv(summary_rows, SUMMARY_COLUMNS,
                 os.path.join(out_dir, "walkforward_summary.csv"))

    selected = [r for r in summary_rows if r.get("selected_combo_id") is not None]
    verdicts: dict = {}
    for r in summary_rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    aggregate = {
        "n_folds": len(folds),
        "n_selected": len(selected),
        "n_test_sample_ok": sum(1 for r in selected if r["test_sample_ok"]),
        "verdict_counts": verdicts,
        "test_trades_total": sum(r["test_trades"] for r in selected),
        "test_net_pnl_total": round(sum(r["test_net_pnl"] for r in selected), 2),
        "oos_windows_overlap": overlap,
        "note": ("test windows overlap across folds: totals double-count"
                 if overlap else "test windows are disjoint across folds"),
    }
    full = {"aggregate": aggregate, "folds": fold_reports,
            "grid_size": n_grid, "effective_grid_size": n_eff}
    gs.write_json(full, os.path.join(out_dir, "walkforward_report.json"))
    return full
