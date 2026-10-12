"""
Walk-Forward Validation
=======================
Run the SAME strategy configuration on three non-overlapping calendar years and
compare the results side-by-side.

WHY THIS MATTERS — OVERFITTING
-------------------------------
Any strategy that is optimised (even informally) on a single historical window
will look better on that window than it deserves to.  Walk-forward validation is
the simplest sanity check: if the edge is REAL it should show up across multiple
independent time periods, not just the one you tuned it on.

HOW IT WORKS
------------
1. Fetch daily + hourly bars for each window once (API calls are slow; we cache
   per year in /tmp so re-runs are instant).
2. Call backtester.run_simulation() for each year with identical strategy config.
3. Print a side-by-side comparison table.

A good strategy shows:
  - Positive total P/L in at least 2/3 windows
  - Sharpe > 0.5 consistently
  - Max drawdown within acceptable range (e.g. < 20%)

A strategy that only works on ONE year is likely overfitted.
"""

import os
import json
import math
import pickle
import hashlib
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

# Import the engine (strategy config, levers, and run_simulation all live there)
import backtester as bt
from backtester import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

# ---------------------------------------------------------------------------
# Walk-forward windows — edit these to taste
# ---------------------------------------------------------------------------
WINDOWS = [
    ("2023-01-01", "2023-12-31", "2023"),
    ("2024-01-01", "2024-12-31", "2024"),
    ("2025-01-01", "2025-12-31", "2025"),
]

CACHE_DIR = "/tmp/wf_cache"
RESULTS_PATH = "/tmp/wf_results.json"
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_or_load(start_date: str, end_date: str, label: str):
    """
    Fetch daily + hourly bars for all SYMBOLS in a date window, caching the
    result to disk.  If the cache file already exists the API is not called.

    Caching strategy
    ----------------
    Key = label + SYMBOLS hash so a change in the symbol list invalidates the
    cache automatically.
    """
    # hashlib, not built-in hash(): Python randomises str hashes per process
    # (PYTHONHASHSEED), so hash() would produce a different cache filename on
    # every run and the cache would never be reused.
    sym_hash = hashlib.md5(",".join(sorted(bt.SYMBOLS)).encode()).hexdigest()[:8]
    cache_path = os.path.join(CACHE_DIR, f"{label}_{sym_hash}.pkl")

    if os.path.exists(cache_path):
        logging.info(f"[{label}] Loading data from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logging.info(f"[{label}] Fetching data for {len(bt.SYMBOLS)} symbols ({start_date} -> {end_date})...")
    all_daily: dict = {}
    all_hourly: dict = {}

    for symbol in bt.SYMBOLS:
        daily = bt.get_historical_bars(symbol, start_date, end_date, bt.TIME_FRAME)
        if not daily.empty:
            all_daily[symbol] = daily

        hourly = bt.get_historical_bars(symbol, start_date, end_date, bt.SCALPING_TIME_FRAME)
        if not hourly.empty:
            all_hourly[symbol] = hourly

    data = (all_daily, all_hourly)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"[{label}] Data cached to {cache_path}")
    return data


def _json_safe(v):
    """Convert NaN/inf to None so json.dumps doesn't choke."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def save_results(results: list[dict], path: str = RESULTS_PATH):
    """
    Persist walk-forward results to a JSON file the Streamlit dashboard can read.

    We strip out the raw trade_history and portfolio_history (too large) and
    keep only the pre-aggregated daily_equity_curve and strategy_breakdown.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": [
            {
                "label":              r["label"],
                "start_date":         r["start_date"],
                "end_date":           r["end_date"],
                "final_equity":       _json_safe(r["final_equity"]),
                "total_pl":           _json_safe(r["total_pl"]),
                "pl_pct":             _json_safe(r["pl_pct"]),
                "max_drawdown":       _json_safe(r["max_drawdown"]),
                "sharpe":             _json_safe(r["sharpe"]),
                "benchmark_pct":      _json_safe(r.get("benchmark_pct")),
                "alpha_pp":           _json_safe(r.get("alpha_pp")),
                "num_trades":         r["num_trades"],
                "daily_equity_curve": r.get("daily_equity_curve", []),
                "strategy_breakdown": r.get("strategy_breakdown", []),
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Results saved to {path}")


def print_comparison(results: list[dict]):
    """Print a clean side-by-side comparison table."""
    col_w = 14
    headers = ["Metric"] + [r["label"] for r in results]
    sep = "-" * (20 + col_w * len(results))

    print(f"\n{'=' * len(sep)}")
    print("  WALK-FORWARD VALIDATION RESULTS")
    print(f"{'=' * len(sep)}")
    print(f"{'Metric':<20}" + "".join(f"{h:>{col_w}}" for h in headers[1:]))
    print(sep)

    def row(name, vals):
        print(f"{name:<20}" + "".join(f"{v:>{col_w}}" for v in vals))

    def fmt_opt(v, spec, suffix=""):
        return (format(v, spec) + suffix) if v is not None else "N/A"

    row("Total P/L ($)", [f"${r['total_pl']:,.0f}" for r in results])
    row("Return (%)", [f"{r['pl_pct']:+.2f}%" for r in results])
    row("SPY Buy-Hold (%)", [fmt_opt(r.get('benchmark_pct'), '+.2f', '%') for r in results])
    row("Alpha (pp)", [fmt_opt(r.get('alpha_pp'), '+.2f') for r in results])
    row("Sharpe", [f"{r['sharpe']:.2f}" if r['sharpe'] == r['sharpe'] else "N/A" for r in results])
    row("Max Drawdown (%)", [f"{r['max_drawdown']:.2f}%" for r in results])
    row("# Trades", [str(r['num_trades']) for r in results])
    row("Final Equity ($)", [f"${r['final_equity']:,.0f}" for r in results])

    print(sep)

    # Verdict
    positive_years = sum(1 for r in results if r["pl_pct"] > 0)
    avg_sharpe = sum(r["sharpe"] for r in results if r["sharpe"] == r["sharpe"]) / max(
        1, sum(1 for r in results if r["sharpe"] == r["sharpe"])
    )
    alphas = [r.get("alpha_pp") for r in results if r.get("alpha_pp") is not None]
    avg_alpha = sum(alphas) / len(alphas) if alphas else None
    alpha_str = f"  |  Avg Alpha: {avg_alpha:+.2f}pp" if avg_alpha is not None else ""
    beat_index_years = sum(1 for a in alphas if a > 0)
    print(f"\nVerdict: {positive_years}/{len(results)} years positive  |  "
          f"Beat SPY in {beat_index_years}/{len(alphas) or len(results)} years  |  "
          f"Avg Sharpe: {avg_sharpe:.2f}{alpha_str}")
    if positive_years >= 2 and avg_sharpe > 0.3:
        print("Strategy shows CONSISTENT edge across time windows.")
    elif positive_years == len(results):
        print("Strategy profitable every year but Sharpe is low — watch drawdowns.")
    else:
        print("WARNING: strategy is not consistent across windows — possible overfitting.")
    print()


if __name__ == "__main__":
    all_results = []

    for start_date, end_date, label in WINDOWS:
        all_daily, all_hourly = fetch_or_load(start_date, end_date, label)

        if not all_daily and not all_hourly:
            logging.warning(f"[{label}] No data available — skipping window.")
            continue

        result = run_simulation(
            start_date=start_date,
            end_date=end_date,
            all_daily_bars=all_daily,
            all_hourly_bars=all_hourly,
            initial_cash=100_000.0,
            label=label,
        )
        all_results.append(result)

    if all_results:
        print_comparison(all_results)
        save_results(all_results)
    else:
        logging.error("No results produced — check data fetch logs above.")
