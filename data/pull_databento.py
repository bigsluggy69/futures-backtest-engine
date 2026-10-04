#!/usr/bin/env python3
"""DataBento historical pull for MES / MNQ / MGC continuous front-month.

Pulls ohlcv-1m bars from GLBX.MDP3 for the continuous front-month contract of
each root, from a start date to present, and writes Parquet partitioned by
symbol/year-month under data/parquet/.

Safety:
  * Billed get_range call only runs with --confirm AND estimated cost below
    --max-cost (default $25).
  * API key is read ONLY from env DATABENTO_API_KEY. Never logged, never
    written to disk.
  * Idempotent top-ups: a symbol is re-pulled only when its cache does not
    cover the requested range (2-day buffer for T+1 lag); the month
    containing the pull end is always refreshed since it may be partial.

Top-up usage:
  python data/pull_databento.py --start 2026-11-01 --end 2026-11-30 --confirm
pulls just November; older complete months are skipped, not re-billed.

Outputs:
  data/parquet/<SYMBOL>/<YYYY-MM>.parquet
  data/manifests/manifest_<UTC>.json
  data/reports/validation_<UTC>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import databento as db


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
ROOTS = ["MES", "MNQ", "MGC"]
ROLL_RULES_TRY = ["v", "c", "n"]  # try v.0 first, then calendar, then OI
RANK = 0
STYPE_IN = "continuous"
STYPE_OUT_RESOLVE = "instrument_id"
PRICE_SCALE = 1e9

OUT_ROOT = Path("data/parquet")
MANIFEST_DIR = Path("data/manifests")
REPORT_DIR = Path("data/reports")

# Liquid hours per root in UTC, used to classify gaps as "true" vs "quiet".
# Equity index futures US cash session (EST 14:30-21:00, EDT 13:30-20:00).
# COMEX gold floor (EST 13:20-18:30, EDT 12:20-17:30).
LIQUID_HOURS_UTC = {
    "MES": (13, 30, 21, 0),
    "MNQ": (13, 30, 21, 0),
    "MGC": (12, 20, 18, 30),
}
LIQUID_GAP_THRESHOLD_MIN = 3  # gaps >= this many minutes during liquid hours get flagged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def get_api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        sys.exit("ERROR: DATABENTO_API_KEY environment variable is not set.")
    return key


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def months_in_range(start_str: str, end_str: str) -> list[str]:
    start = pd.Timestamp(start_str, tz="UTC").replace(day=1)
    end = pd.Timestamp(end_str, tz="UTC").replace(day=1)
    return [m.strftime("%Y-%m") for m in pd.date_range(start, end, freq="MS")]


def _in_liquid(ts: pd.Timestamp, root: str) -> bool:
    if root not in LIQUID_HOURS_UTC:
        return False
    h0, m0, h1, m1 = LIQUID_HOURS_UTC[root]
    hm = ts.hour * 60 + ts.minute
    return h0 * 60 + m0 <= hm <= h1 * 60 + m1


_CHICAGO = ZoneInfo("America/Chicago")


def _in_maintenance_window(ts: pd.Timestamp) -> bool:
    """Scheduled CME Globex halts, evaluated in America/Chicago.

    Covers the weekend halt (Fri 16:00 CT -> Sun 17:00 CT) and the daily
    15:15-15:30 CT halt. Gaps starting inside these windows are expected
    closures, never feed gaps.

    # UNDEFINED: exact Sunday reopen (17:00 CT typical for CME equity
    # futures; some products differ). Conservative: the whole window counts
    # as expected closure — a real feed gap overlapping it would be masked,
    # which is preferable to ~1300 false TRUE gaps from the daily halt.
    """
    c = ts.tz_convert(_CHICAGO)
    wd, hm = c.weekday(), c.hour * 60 + c.minute
    if wd == 5:  # Saturday: full weekend halt
        return True
    if wd == 4 and hm >= 16 * 60:  # Friday after 16:00 CT
        return True
    if wd == 6 and hm < 17 * 60:  # Sunday before 17:00 CT
        return True
    if wd < 5 and 15 * 60 + 15 <= hm < 15 * 60 + 30:  # daily halt
        return True
    return False


# ---------------------------------------------------------------------------
# [1] Symbology resolution (free)
# ---------------------------------------------------------------------------
def resolve_continuous(client: db.Historical, start_date: str, end_date: str) -> dict:
    """Resolve continuous front-month symbols. Abort if any root fails."""
    resolved: dict[str, dict] = {}
    for root in ROOTS:
        found = None
        for rule in ROLL_RULES_TRY:
            sym = f"{root}.{rule}.{RANK}"
            try:
                res = client.symbology.resolve(
                    dataset=DATASET,
                    symbols=[sym],
                    stype_in=STYPE_IN,
                    stype_out=STYPE_OUT_RESOLVE,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"  resolve {sym}: failed ({type(exc).__name__}: {exc})")
                continue
            mappings = (res or {}).get("result", {}).get(sym) or []
            if mappings:
                found = {
                    "symbol": sym,
                    "roll_rule": rule,
                    "rank": RANK,
                    "mappings": mappings,
                }
                break
        if not found:
            tried = [f"{root}.{r}.{RANK}" for r in ROLL_RULES_TRY]
            sys.exit(
                f"ERROR: could not resolve any continuous symbol for root {root} "
                f"(tried {tried}). Refusing to guess."
            )
        resolved[root] = found
        log(
            f"  resolved {root} -> {found['symbol']} "
            f"({len(found['mappings'])} mappings)"
        )
    return resolved


def extract_rolls(mappings: list[dict]) -> list[dict]:
    """Return list of {date, instrument_id} at each underlying-instrument change."""
    rolls: list[dict] = []
    prev = None
    for m in mappings:
        iid = m.get("s")
        if iid != prev:
            rolls.append({"date": m.get("d0"), "instrument_id": iid})
            prev = iid
    return rolls


# ---------------------------------------------------------------------------
# [2] Cost estimation (free)
# ---------------------------------------------------------------------------
def estimate_cost(
    client: db.Historical,
    resolved: dict,
    start_iso: str,
    end_iso: str,
) -> tuple[float, int, list[dict]]:
    rows: list[dict] = []
    total_cost = 0.0
    total_size = 0
    for root, info in resolved.items():
        sym = info["symbol"]
        size = int(
            client.metadata.get_billable_size(
                dataset=DATASET,
                symbols=[sym],
                stype_in=STYPE_IN,
                schema=SCHEMA,
                start=start_iso,
                end=end_iso,
            )
        )
        try:
            cost = float(
                client.metadata.get_cost(
                    dataset=DATASET,
                    symbols=[sym],
                    stype_in=STYPE_IN,
                    schema=SCHEMA,
                    start=start_iso,
                    end=end_iso,
                )
            )
        except Exception:  # noqa: BLE001
            # Fallback: $70/GB historical rate for ohlcv-1m.
            cost = size / 1e9 * 70.0
        rows.append(
            {
                "root": root,
                "symbol": sym,
                "cost_usd": cost,
                "billable_bytes": size,
            }
        )
        total_cost += cost
        total_size += size
    return total_cost, total_size, rows


# ---------------------------------------------------------------------------
# [4] Normalization
# ---------------------------------------------------------------------------
def _scale_price(s: pd.Series) -> pd.Series:
    """Fixed-point int64 (1 unit = 1e-9) -> float; floats pass through.

    databento's to_df() already scales prices to float in recent client
    versions, while raw DBN/Arrow output is int64. Detect the dtype instead
    of assuming, so a second division can never silently shrink prices 1e9x.
    """
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.astype("float64") / PRICE_SCALE
    if pd.api.types.is_float_dtype(s.dtype):
        return s.astype("float64")
    raise RuntimeError(f"unexpected price dtype {s.dtype}; refusing to guess")


def normalize(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Return canonical frame: timestamp_utc, OHLC floats, volume int, symbol."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "ts_event":
        df = df.reset_index()
    if "ts_event" not in df.columns:
        raise RuntimeError(
            f"Unexpected DataFrame for {sym}: no ts_event. "
            f"Columns={list(df.columns)}, index={df.index.name}"
        )

    if pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
        ts = pd.to_datetime(df["ts_event"], utc=True)
    else:
        ts = pd.to_datetime(df["ts_event"].astype("int64"), unit="ns", utc=True)

    out = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "open": _scale_price(df["open"]),
            "high": _scale_price(df["high"]),
            "low": _scale_price(df["low"]),
            "close": _scale_price(df["close"]),
            "volume": df["volume"].astype("int64"),
            "symbol": sym,
        }
    )
    return out.sort_values("timestamp_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# [6] Validation
# ---------------------------------------------------------------------------
def validate(all_frames: dict[str, pd.DataFrame], resolved: dict) -> dict:
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": {},
    }
    sym_to_root = {info["symbol"]: root for root, info in resolved.items()}

    for sym, df in all_frames.items():
        root = sym_to_root.get(sym, sym.split(".")[0])
        r: dict = {
            "bars": int(len(df)),
            "first_ts": str(df["timestamp_utc"].min()),
            "last_ts": str(df["timestamp_utc"].max()),
        }

        # OHLC sanity
        bad_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
        bad_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
        bad_price = int(
            ((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()
        )
        r["ohlc_violations"] = {
            "high_lt_max_open_close": bad_high,
            "low_gt_min_open_close": bad_low,
            "nonpositive_prices": bad_price,
        }

        # Gap analysis
        s = df.sort_values("timestamp_utc").reset_index(drop=True)
        deltas = s["timestamp_utc"].diff().dt.total_seconds() / 60.0
        gap_idx = deltas[deltas > 1.0].index
        true_gaps: list[dict] = []
        quiet_gaps: list[dict] = []
        maint_gaps: list[dict] = []
        total_gap_min = 0.0
        for i in gap_idx:
            gap_min = float(deltas.loc[i])
            total_gap_min += gap_min - 1.0  # missing minutes between bars
            gap_start = s.loc[i - 1, "timestamp_utc"] + pd.Timedelta(minutes=1)
            entry = {"start": str(gap_start), "minutes": gap_min - 1.0}
            if _in_maintenance_window(gap_start):
                maint_gaps.append(entry)
            elif _in_liquid(gap_start, root) and (gap_min - 1.0) >= LIQUID_GAP_THRESHOLD_MIN:
                true_gaps.append(entry)
            else:
                quiet_gaps.append(entry)
        r["gaps"] = {
            "total_missing_minutes": total_gap_min,
            "true_gap_count": len(true_gaps),
            "quiet_gap_count": len(quiet_gaps),
            "maintenance_gap_count": len(maint_gaps),
            "largest_true_gaps": sorted(true_gaps, key=lambda x: -x["minutes"])[:10],
            "largest_quiet_gaps": sorted(quiet_gaps, key=lambda x: -x["minutes"])[:5],
        }
        report["symbols"][sym] = r
    return report


def print_report(report: dict) -> None:
    for sym, r in report["symbols"].items():
        log(f"  [{sym}] bars={r['bars']:,}")
        log(f"    range: {r['first_ts']} -> {r['last_ts']}")
        v = r["ohlc_violations"]
        log(
            f"    OHLC violations: high<max(o,c)={v['high_lt_max_open_close']}, "
            f"low>min(o,c)={v['low_gt_min_open_close']}, "
            f"nonpos={v['nonpositive_prices']}"
        )
        g = r["gaps"]
        log(
            f"    gaps: missing_minutes={g['total_missing_minutes']:.0f}, "
            f"true={g['true_gap_count']}, quiet={g['quiet_gap_count']}"
        )
        for e in g["largest_true_gaps"][:5]:
            log(f"      TRUE gap at {e['start']} (+{e['minutes']:.0f} min)")


# ---------------------------------------------------------------------------
# [4] Partition write / idempotency
# ---------------------------------------------------------------------------
def latest_cached_bar(sym: str) -> Optional[pd.Timestamp]:
    """Max timestamp_utc in the newest cached partition, or None if no cache.

    Used for top-up decisions: a symbol is only re-pulled when its cache does
    not already cover the requested range.
    """
    sym_dir = OUT_ROOT / sym
    if not sym_dir.exists():
        return None
    files = sorted(sym_dir.glob("*.parquet"))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1], columns=["timestamp_utc"])
    except Exception:
        return None
    if len(df) == 0:
        return None
    return pd.to_datetime(df["timestamp_utc"].max(), utc=True)


def write_partitions(
    sym: str, df: pd.DataFrame, force: bool,
    refresh_months: frozenset = frozenset(),
) -> list[dict]:
    """Write Parquet partitioned by year-month. Returns manifest entries.

    Months in `refresh_months` are always rewritten (they may hold partial
    data, e.g. the month containing the pull end); other existing partitions
    are skipped unless `force`.
    """
    df = df.copy()
    df["_ym"] = df["timestamp_utc"].dt.strftime("%Y-%m")
    sym_dir = OUT_ROOT / sym
    sym_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for ym, group in df.groupby("_ym", sort=True):
        path = sym_dir / f"{ym}.parquet"
        out = (
            group.drop(columns=["_ym"])
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        if path.exists() and not force and ym not in refresh_months:
            log(f"    skip (exists): {path}")
            continue
        out.to_parquet(path, engine="pyarrow", index=False)
        entries.append(
            {
                "symbol": sym,
                "month": ym,
                "path": str(path),
                "bars": int(len(out)),
                "sha256": sha256_file(path),
            }
        )
        log(f"    wrote {path} ({len(out):,} bars)")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2021-01-01", help="YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="YYYY-MM-DD (inclusive)",
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually run the billed get_range call.",
    )
    ap.add_argument(
        "--max-cost",
        type=float,
        default=25.0,
        help="Abort if estimated cost exceeds this (USD).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing partitions.",
    )
    args = ap.parse_args()

    key = get_api_key()
    client = db.Historical(key)

    start_dt = parse_date(args.start)
    end_dt_incl = parse_date(args.end)
    end_dt = end_dt_incl + timedelta(days=1)  # get_range end is exclusive
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    log(f"DataBento pull: {DATASET} {SCHEMA} {args.start} -> {args.end}")
    log(f"Roots: {ROOTS}")

    log("\n[1/6] Resolving continuous symbols (free)...")
    resolved = resolve_continuous(client, args.start, args.end)

    log("\n[2/6] Estimating cost (free)...")
    total_cost, total_size, cost_rows = estimate_cost(
        client, resolved, start_iso, end_iso
    )
    for r in cost_rows:
        log(
            f"  {r['root']:4s} {r['symbol']:12s} "
            f"${r['cost_usd']:.4f}   {r['billable_bytes']/1e6:.2f} MB"
        )
    log(f"  TOTAL: ${total_cost:.4f}   {total_size/1e6:.2f} MB")

    if not args.confirm:
        log("\nABORT: --confirm not passed. No billed calls made.")
        return
    if total_cost > args.max_cost:
        log(
            f"\nABORT: estimated ${total_cost:.4f} exceeds --max-cost "
            f"${args.max_cost:.2f}. Raise --max-cost to proceed."
        )
        return

    today_utc = datetime.now(timezone.utc)
    # The month containing the pull end (bounded by today) can hold partial
    # data — always refresh it; older months are complete and skipped.
    hot_months = frozenset({min(end_dt_incl, today_utc).strftime("%Y-%m")})

    # Idempotency / top-up: skip a symbol only when its cache already covers
    # the requested range. DataBento historical is T+1, so allow a 2-day
    # buffer before deciding the cache is current.
    coverage_buf = pd.Timedelta(days=2)
    symbols_to_pull = []
    for root, info in resolved.items():
        sym = info["symbol"]
        latest = None if args.force else latest_cached_bar(sym)
        if latest is not None and latest >= end_dt - coverage_buf:
            log(f"  {sym}: cache covers through {latest} — skipping.")
        else:
            symbols_to_pull.append((root, info))

    if not symbols_to_pull:
        log("\nNothing to pull. All partitions present.")
        return

    log(f"\n[3/6] Pulling {len(symbols_to_pull)} symbol(s) (BILLED)...")
    all_frames: dict[str, pd.DataFrame] = {}
    rolls_by_symbol: dict[str, list[dict]] = {}
    for root, info in symbols_to_pull:
        sym = info["symbol"]
        log(f"  get_range {sym} ...")
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[sym],
            stype_in=STYPE_IN,
            schema=SCHEMA,
            start=start_iso,
            end=end_iso,
        )
        raw = data.to_df()
        if len(raw) == 0:
            log(f"    WARNING: no data returned for {sym}")
            continue
        df = normalize(raw, sym)
        all_frames[sym] = df
        rolls = extract_rolls(info["mappings"])
        rolls_by_symbol[sym] = rolls
        # Sanity: ~4 quarterly rolls/year expected. Far fewer means the
        # symbology mapping format likely drifted — fail loudly, not silently.
        years = max((end_dt - start_dt).days / 365.25, 1 / 12)
        if len(rolls) < 2 * years:
            log(
                f"    WARNING: only {len(rolls)} rolls detected over "
                f"{years:.1f}y for {sym} (expected ~{int(4 * years)}). "
                f"Symbology mapping format may have changed — check the manifest."
            )
        log(
            f"    {len(df):,} bars, "
            f"{df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}"
        )

    log("\n[4/6] Writing Parquet partitions...")
    manifest_partitions: list[dict] = []
    for sym, df in all_frames.items():
        manifest_partitions.extend(write_partitions(sym, df, args.force, hot_months))

    log("\n[5/6] Validation report...")
    report = validate(all_frames, resolved)
    print_report(report)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rpt_path = REPORT_DIR / f"validation_{stamp}.json"
    rpt_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"  report: {rpt_path}")

    log("\n[6/6] Writing manifest...")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pull_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "schema": SCHEMA,
        "date_range": {"start": args.start, "end": args.end},
        "resolved_symbols": {
            root: {
                "symbol": info["symbol"],
                "roll_rule": info["roll_rule"],
                "rank": info["rank"],
            }
            for root, info in resolved.items()
        },
        "estimated_credit_cost_usd": total_cost,
        "credit_cost_note": "pre-pull estimate from metadata.get_cost; actual billed cost may differ",
        "billable_bytes": total_size,
        "partitions": manifest_partitions,
        "rolls": rolls_by_symbol,
    }
    mpath = MANIFEST_DIR / f"manifest_{stamp}.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    log(f"  manifest: {mpath}")

    # Final summary
    log("\n=== SUMMARY ===")
    total_bars = sum(len(df) for df in all_frames.values())
    for sym, df in all_frames.items():
        log(
            f"  {sym}: {len(df):,} bars, "
            f"{df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}"
        )
    log(f"  Total bars pulled this run: {total_bars:,}")
    log(f"  Credits consumed (estimate): ${total_cost:.4f}")
    log(f"  Manifest: {mpath}")


if __name__ == "__main__":
    main()
