"""
Strategy-correlation analysis.

THE QUESTION
------------
Is the ensemble genuinely diversified, or are the strategies redundant bets
on the same market moves?

Two strategies can both be "profitable" yet add nothing to each other: if
their daily return streams are highly correlated (rho > ~0.7), they win and
lose on the same days, so running both just doubles position turnover without
smoothing the equity curve. True diversification needs LOW correlation
(rho < ~0.3): then one strategy's bad day is routinely offset by another's
good day, and the portfolio-level Sharpe beats any individual strategy's.
This is the actual mathematics behind "don't put all eggs in one basket" --
portfolio variance = sum of covariances, not just variances.

THE METHOD
----------
1. Run each strategy SOLO through the bias-free engine, one year at a time,
   on the cached walk-forward data (2023/2024/2025; no API calls).
2. Take each run's daily equity curve, convert to daily % returns, and
   stitch the three years into one return stream per strategy.
3. Pearson-correlate every pair of streams.

USAGE
-----
  python analyze_correlation.py run <key>    # simulate one strategy solo
                                             # (run the 4 keys in parallel)
  python analyze_correlation.py report       # combine + print the matrix

Keys: macd, atr_channel, bollinger, rsi
"""

import json
import logging
import sys

import pandas as pd

import backtester as bt
from walk_forward import WINDOWS, fetch_or_load
from strategies import (
    MacdMomentumStrategy,
    AtrChannelBreakoutStrategy,
    BollingerBandReversionStrategy,
    RsiMomentumStrategy,
)

# The four strategies that survived walk-forward pruning, with the same
# parameters and timeframes they run on in production.
STRATEGY_MAP = {
    "macd":        (MacdMomentumStrategy(), "daily"),
    "atr_channel": (AtrChannelBreakoutStrategy(), "daily"),
    "bollinger":   (BollingerBandReversionStrategy(window=20, num_std=2.0), "hourly"),
    "rsi":         (RsiMomentumStrategy(rsi_period=14, oversold=30, overbought=70,
                                        trend_sma_period=50), "hourly"),
}

RETURNS_PATH = "/tmp/corr_{key}.json"
MATRIX_PATH = "/tmp/corr_matrix.json"


def run_solo(key: str):
    """Simulate one strategy alone across all walk-forward windows and save
    its stitched daily-return stream."""
    strategy, timeframe = STRATEGY_MAP[key]
    bt.STRATEGIES = [(strategy, timeframe)]

    # Trade-by-trade logging would add minutes of I/O across ~3 years.
    logging.getLogger().setLevel(logging.WARNING)

    all_returns = {}
    for start_date, end_date, label in WINDOWS:
        all_daily, all_hourly = fetch_or_load(start_date, end_date, label)
        result = bt.run_simulation(start_date, end_date, all_daily, all_hourly,
                                   initial_cash=100_000.0, label=f"{key}-{label}")

        curve = pd.DataFrame(result["daily_equity_curve"])
        if curve.empty:
            continue
        curve["date"] = pd.to_datetime(curve["date"])
        curve = curve.set_index("date").sort_index()
        # pct_change: each day's return relative to the previous day's equity.
        # Each year starts from fresh $100k, so the year-boundary return is
        # dropped (dropna also removes the first row of each stitch).
        rets = curve["equity"].pct_change().dropna()
        for dt, r in rets.items():
            all_returns[str(dt.date())] = round(float(r), 8)

    path = RETURNS_PATH.format(key=key)
    with open(path, "w") as f:
        json.dump(all_returns, f)
    print(f"{key}: {len(all_returns)} daily returns saved to {path}")


def report():
    """Load all return streams, align on common dates, print the matrix."""
    streams = {}
    for key in STRATEGY_MAP:
        try:
            with open(RETURNS_PATH.format(key=key)) as f:
                streams[key] = pd.Series(json.load(f))
        except FileNotFoundError:
            print(f"WARNING: missing {RETURNS_PATH.format(key=key)} — "
                  f"run `python analyze_correlation.py run {key}` first.")
    if len(streams) < 2:
        print("Need at least 2 return streams for a correlation matrix.")
        return

    df = pd.DataFrame(streams).dropna()  # align: only dates ALL strategies traded through
    corr = df.corr()

    print(f"\nAligned trading days: {len(df)}")
    print("\nSTRATEGY RETURN CORRELATION MATRIX (Pearson)")
    print("=" * 60)
    print(corr.round(2).to_string())
    print()

    # Pairwise interpretation, worst offenders first.
    pairs = []
    keys = list(corr.columns)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda p: -abs(p[2]))

    print("PAIRWISE VERDICTS (sorted by |rho|)")
    print("-" * 60)
    for a, b, rho in pairs:
        if rho > 0.7:
            verdict = "REDUNDANT — same bet twice, consider cutting one"
        elif rho > 0.3:
            verdict = "moderate overlap"
        elif rho > -0.3:
            verdict = "diversifying — this is what we want"
        else:
            verdict = "hedging — negatively correlated"
        print(f"  {a:12s} x {b:12s}  rho = {rho:+.2f}   {verdict}")

    with open(MATRIX_PATH, "w") as f:
        json.dump({"days": len(df), "matrix": corr.round(4).to_dict()}, f, indent=2)
    print(f"\nMatrix saved to {MATRIX_PATH}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run_solo(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "report":
        report()
    else:
        print(__doc__)
