"""
Quick ensemble test: run the NEW 5-strategy production mix
(MACD + ROC + ATR Channel + Bollinger + RSI) on the benchmark's cached
2025 data and compare against the old 3-strategy baseline.

Uses /tmp/bt_benchmark_cache.pkl left behind by test_20_strategies.py,
so no API calls are made.
"""

import logging
import pickle

import backtester as bt
from strategies import (
    MacdMomentumStrategy,
    RocMomentumStrategy,
    AtrChannelBreakoutStrategy,
    BollingerBandReversionStrategy,
    RsiMomentumStrategy,
)

# Silence per-trade logging; we only want the summaries.
logging.getLogger().setLevel(logging.WARNING)

CACHE = "/tmp/bt_benchmark_cache.pkl"

with open(CACHE, "rb") as f:
    cache = pickle.load(f)
all_daily, all_hourly = cache["daily"], cache["hourly"]
print(f"Loaded cache: {len(all_daily)} daily / {len(all_hourly)} hourly symbol sets\n")

OLD_MIX = [
    (MacdMomentumStrategy(), "daily"),
    (BollingerBandReversionStrategy(window=20, num_std=2.0), "hourly"),
    (RsiMomentumStrategy(rsi_period=14, oversold=30, overbought=70, trend_sma_period=50), "hourly"),
]

runs = [
    ("OLD 3-strategy mix", OLD_MIX),
    ("NEW 5-strategy mix", bt.STRATEGIES),   # as configured in backtester.py
]

results = []
for label, mix in runs:
    bt.STRATEGIES = mix
    r = bt.run_simulation("2025-01-01", "2026-01-01", all_daily, all_hourly,
                          initial_cash=100_000.0, label=label)
    results.append((label, r))

print(f"\n{'Mix':<22}{'Return':>9}{'Sharpe':>8}{'MaxDD':>9}{'Trades':>8}")
print("-" * 56)
for label, r in results:
    print(f"{label:<22}{r['pl_pct']:>+8.2f}%{r['sharpe']:>8.2f}{r['max_drawdown']:>8.2f}%{r['num_trades']:>8}")

print("\nPer-strategy breakdown (NEW mix):")
for s in results[1][1]["strategy_breakdown"]:
    print(f"  {s['strategy']:<24} {s['trades']:>4} closed | "
          f"P/L ${s['total_pl']:>10,.2f} | win {s['win_rate']:.1f}%")
