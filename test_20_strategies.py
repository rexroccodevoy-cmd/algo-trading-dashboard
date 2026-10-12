"""
Strategy Benchmarker — tests all 20 strategies from strategies_20.py
against the existing backtester and ranks them by Sharpe ratio.

Usage
-----
  cd /Users/rex/trading_dashboard
  source venv/bin/activate
  python test_20_strategies.py

What it does
------------
1. Fetches (or reloads from a local pickle cache) all bar data from Alpaca
   for the backtester's configured symbols and date range.
2. Runs each of the 20 strategies in isolation through `run_simulation()` —
   using the same risk levers, position sizing, and stop-loss logic already
   configured in backtester.py so results are comparable.
3. Runs a "current production" baseline using the strategies already in
   backtester.STRATEGIES for direct comparison.
4. Prints a colour-coded ranked table sorted by Sharpe ratio.
5. Saves the full results to strategy_benchmark_results.csv in this directory.

Cache note
----------
Bar data is saved to /tmp/bt_benchmark_cache.pkl after the first fetch.
Delete that file to force a fresh download from Alpaca.
"""

import os
import sys
import pickle
import logging
import csv
from datetime import datetime

import numpy as np
import pandas as pd

# ── Silence the backtester's INFO spam during bulk runs ──────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
bt_logger = logging.getLogger()
bt_logger.setLevel(logging.WARNING)

import backtester as bt
from strategies_20 import ALL_STRATEGIES

CACHE_PATH = "/tmp/bt_benchmark_cache.pkl"
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "strategy_benchmark_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────

def load_or_fetch_data():
    """Return (all_daily_bars, all_hourly_bars), loading from cache if available."""
    if os.path.exists(CACHE_PATH):
        print(f"  Loading bar data from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["daily"], cache["hourly"]

    print(f"  No cache found — fetching {len(bt.SYMBOLS)} symbols from Alpaca...")
    print(f"  Date range: {bt.BACKTEST_START_DATE} → {bt.BACKTEST_END_DATE}")
    all_daily, all_hourly = {}, {}

    for i, symbol in enumerate(bt.SYMBOLS, 1):
        print(f"    [{i:2d}/{len(bt.SYMBOLS)}] {symbol}", end="", flush=True)
        d = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.TIME_FRAME)
        if not d.empty:
            all_daily[symbol] = d
        h = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.SCALPING_TIME_FRAME)
        if not h.empty:
            all_hourly[symbol] = h
        print(" ✓")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"daily": all_daily, "hourly": all_hourly}, f)
    print(f"  Cache saved → {CACHE_PATH}\n")

    return all_daily, all_hourly


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy_test(strategy_instance, timeframe, all_daily_bars, all_hourly_bars, label=None):
    """Temporarily replace bt.STRATEGIES with a single strategy and run the simulation."""
    original_strategies = bt.STRATEGIES
    bt.STRATEGIES = [(strategy_instance, timeframe)]

    try:
        result = bt.run_simulation(
            bt.BACKTEST_START_DATE,
            bt.BACKTEST_END_DATE,
            all_daily_bars,
            all_hourly_bars,
            initial_cash=100_000.0,
            label=label or strategy_instance.name,
        )
    finally:
        bt.STRATEGIES = original_strategies  # always restore

    return result


def run_baseline(all_daily_bars, all_hourly_bars):
    """Run the current production strategy mix for comparison."""
    result = bt.run_simulation(
        bt.BACKTEST_START_DATE,
        bt.BACKTEST_END_DATE,
        all_daily_bars,
        all_hourly_bars,
        initial_cash=100_000.0,
        label="PRODUCTION_BASELINE",
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_BOLD   = "\033[1m"
ANSI_RESET  = "\033[0m"
ANSI_CYAN   = "\033[96m"
ANSI_GREY   = "\033[90m"


def colour_sharpe(sharpe):
    if np.isnan(sharpe):
        return f"{ANSI_GREY}   N/A{ANSI_RESET}"
    if sharpe >= 1.5:
        return f"{ANSI_GREEN}{sharpe:+6.2f}{ANSI_RESET}"
    if sharpe >= 0.5:
        return f"{ANSI_YELLOW}{sharpe:+6.2f}{ANSI_RESET}"
    return f"{ANSI_RED}{sharpe:+6.2f}{ANSI_RESET}"


def colour_pl(pl_pct):
    if pl_pct >= 0:
        return f"{ANSI_GREEN}{pl_pct:+7.2f}%{ANSI_RESET}"
    return f"{ANSI_RED}{pl_pct:+7.2f}%{ANSI_RESET}"


def colour_dd(dd):
    """dd is a negative percentage e.g. -8.5 means 8.5% drawdown."""
    if dd >= -5:
        return f"{ANSI_GREEN}{dd:+6.2f}%{ANSI_RESET}"
    if dd >= -15:
        return f"{ANSI_YELLOW}{dd:+6.2f}%{ANSI_RESET}"
    return f"{ANSI_RED}{dd:+6.2f}%{ANSI_RESET}"


def print_table(rows, baseline):
    """rows: list of dicts with keys: name, sharpe, pl_pct, max_drawdown, num_trades, strategy_type"""
    # Sort by Sharpe (NaN last)
    rows_sorted = sorted(rows, key=lambda r: (np.isnan(r["sharpe"]), -r["sharpe"]))

    hdr = (f"{'#':>3}  {'Strategy':<28} {'Type':<14} "
           f"{'Sharpe':>7} {'Return':>9} {'MaxDD':>8} {'Trades':>7}")
    sep = "─" * len(hdr)

    print(f"\n{ANSI_BOLD}{'='*len(hdr)}{ANSI_RESET}")
    print(f"{ANSI_BOLD}  STRATEGY BENCHMARK RESULTS — ranked by Sharpe Ratio{ANSI_RESET}")
    print(f"  Period: {bt.BACKTEST_START_DATE} → {bt.BACKTEST_END_DATE}  |  Initial capital: $100,000")
    print(f"{ANSI_BOLD}{'='*len(hdr)}{ANSI_RESET}")
    print(f"\n{ANSI_BOLD}{hdr}{ANSI_RESET}")
    print(sep)

    for i, r in enumerate(rows_sorted, 1):
        print(
            f"{i:>3}.  {r['name']:<28} {r['strategy_type']:<14} "
            f"{colour_sharpe(r['sharpe'])}  {colour_pl(r['pl_pct'])}  "
            f"{colour_dd(r['max_drawdown'])}  {r['num_trades']:>6}"
        )

    print(sep)

    # Baseline row
    b = baseline
    b_sharpe = b["sharpe"] if not np.isnan(b["sharpe"]) else float("nan")
    print(
        f"  ★  {'PRODUCTION BASELINE':<28} {'(mixed)':<14} "
        f"{colour_sharpe(b_sharpe)}  {colour_pl(b['pl_pct'])}  "
        f"{colour_dd(b['max_drawdown'])}  {b['num_trades']:>6}"
    )
    print(f"{ANSI_BOLD}{'='*len(hdr)}{ANSI_RESET}")

    # ── Top 5 summary ──────────────────────────────────────────────────────
    top5 = [r for r in rows_sorted if not np.isnan(r["sharpe"])][:5]
    if top5:
        print(f"\n{ANSI_BOLD}{ANSI_CYAN}  TOP 5 STRATEGIES BY SHARPE RATIO:{ANSI_RESET}")
        for rank, r in enumerate(top5, 1):
            print(
                f"    {rank}. {r['name']:<28}  Sharpe {colour_sharpe(r['sharpe'])}  "
                f"Return {colour_pl(r['pl_pct'])}  MaxDD {colour_dd(r['max_drawdown'])}"
            )

    # ── Best Sharpe + low drawdown ─────────────────────────────────────────
    good = [r for r in rows_sorted
            if not np.isnan(r["sharpe"]) and r["max_drawdown"] > -15 and r["sharpe"] > 0]
    if good:
        print(f"\n{ANSI_BOLD}{ANSI_GREEN}  BEST RISK-ADJUSTED (Sharpe > 0, MaxDD < 15%):{ANSI_RESET}")
        for r in good[:5]:
            print(
                f"    • {r['name']:<28}  Sharpe {colour_sharpe(r['sharpe'])}  "
                f"Return {colour_pl(r['pl_pct'])}  MaxDD {colour_dd(r['max_drawdown'])}"
            )
    print()


def save_csv(rows, baseline, path):
    fieldnames = ["rank", "name", "strategy_type", "sharpe", "pl_pct",
                  "total_pl_usd", "max_drawdown", "num_trades"]
    rows_sorted = sorted(rows, key=lambda r: (np.isnan(r["sharpe"]), -r["sharpe"]))

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows_sorted, 1):
            w.writerow({
                "rank": i,
                "name": r["name"],
                "strategy_type": r["strategy_type"],
                "sharpe": "" if np.isnan(r["sharpe"]) else round(r["sharpe"], 4),
                "pl_pct": round(r["pl_pct"], 4),
                "total_pl_usd": round(r.get("total_pl", 0), 2),
                "max_drawdown": round(r["max_drawdown"], 4),
                "num_trades": r["num_trades"],
            })
        # Baseline row
        w.writerow({
            "rank": "BASELINE",
            "name": "PRODUCTION_BASELINE",
            "strategy_type": "mixed",
            "sharpe": "" if np.isnan(baseline["sharpe"]) else round(baseline["sharpe"], 4),
            "pl_pct": round(baseline["pl_pct"], 4),
            "total_pl_usd": round(baseline.get("total_pl", 0), 2),
            "max_drawdown": round(baseline["max_drawdown"], 4),
            "num_trades": baseline["num_trades"],
        })

    print(f"  Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{ANSI_BOLD}╔══════════════════════════════════════════════╗")
    print(f"║   20-Strategy Backtester Benchmark           ║")
    print(f"╚══════════════════════════════════════════════╝{ANSI_RESET}")
    print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Step 1/3 — Loading bar data")
    all_daily, all_hourly = load_or_fetch_data()
    print(f"  {len(all_daily)} symbols with daily bars, {len(all_hourly)} with hourly bars.\n")

    # ── 2. Test each strategy ────────────────────────────────────────────────
    print(f"Step 2/3 — Running {len(ALL_STRATEGIES)} strategy simulations + baseline")
    results = []
    n = len(ALL_STRATEGIES)

    for idx, (strategy, timeframe) in enumerate(ALL_STRATEGIES, 1):
        sname = strategy.name
        print(f"  [{idx:2d}/{n}] {sname:<35}", end="", flush=True)
        try:
            r = run_strategy_test(strategy, timeframe, all_daily, all_hourly)
            sharpe_str = f"{r['sharpe']:+.2f}" if not np.isnan(r["sharpe"]) else "  N/A"
            print(f"  Sharpe: {sharpe_str:>6}  Return: {r['pl_pct']:+.2f}%  MaxDD: {r['max_drawdown']:.2f}%")
            results.append({
                "name": sname,
                "strategy_type": strategy.strategy_type,
                "sharpe": r["sharpe"],
                "pl_pct": r["pl_pct"],
                "total_pl": r["total_pl"],
                "max_drawdown": r["max_drawdown"],
                "num_trades": r["num_trades"],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": sname,
                "strategy_type": strategy.strategy_type,
                "sharpe": float("nan"),
                "pl_pct": 0.0,
                "total_pl": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0,
            })

    print(f"\n  Running production baseline...", end="", flush=True)
    try:
        baseline = run_baseline(all_daily, all_hourly)
        print(f"  Sharpe: {baseline['sharpe']:+.2f}  Return: {baseline['pl_pct']:+.2f}%\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        baseline = {"sharpe": float("nan"), "pl_pct": 0.0, "total_pl": 0.0,
                    "max_drawdown": 0.0, "num_trades": 0}

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print("Step 3/3 — Generating report\n")
    print_table(results, baseline)
    save_csv(results, baseline, RESULTS_CSV)

    # ── Suggest best combos ───────────────────────────────────────────────────
    top_results = sorted(
        [r for r in results if not np.isnan(r["sharpe"])],
        key=lambda r: -r["sharpe"]
    )[:3]

    if top_results:
        print(f"{ANSI_BOLD}  SUGGESTED NEXT STEP:{ANSI_RESET}")
        print("  Consider adding these to backtester.STRATEGIES alongside")
        print("  your current production mix and re-running run_experiments.py:\n")
        for r in top_results:
            # Find the matching entry in ALL_STRATEGIES
            match = next((s for s in ALL_STRATEGIES if s[0].name == r["name"]), None)
            if match:
                strat, tf = match
                cls = strat.__class__.__name__
                print(f"    ({cls}(), \"{tf}\"),  # Sharpe {r['sharpe']:+.2f}, Return {r['pl_pct']:+.2f}%")
        print()


if __name__ == "__main__":
    main()
