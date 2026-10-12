"""
Runs a single backtester.py experiment with overridden flag constants.
Usage:  python run_combo.py <label> <atr_size> <atr_stop> <throttle>

Each flag is 0 or 1.  Results printed to stdout; designed to be launched in
parallel as separate processes so all 4 combos complete in parallel.
"""
import sys, os

label   = sys.argv[1]
atr_size  = sys.argv[2] == "1"
atr_stop  = sys.argv[3] == "1"
throttle  = sys.argv[4] == "1"

# Patch the flags before backtester does anything with them.
# We do this by setting env vars that backtester reads, but since backtester
# doesn't use env vars we patch via a simple source-level monkeypatch:
# import backtester, then override its module globals before __main__ runs.
# Since we can't re-run __main__ from an import, we copy the simulation logic
# into this driver and just load the module for its helpers + data.

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

import pandas as pd
import ta
import backtester as bt
from strategies import BUY, SELL, HOLD, choose_signal_to_act_on

# --- Apply flag overrides ---
bt.USE_ATR_POSITION_SIZING = atr_size
bt.USE_ATR_STOPS           = atr_stop
bt.USE_DRAWDOWN_THROTTLE   = throttle

# --- Load pre-fetched data cache (avoids 8x API hammering when running in parallel) ---
import pickle
with open('/tmp/bt_data_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
all_daily  = cache['daily']
all_hourly = cache['hourly']

# --- Build timeline (daily bars only -- hourly adds ~8x timestamps for minimal
#     signal gain in our daily/hourly strategy mix; using daily as the driver
#     makes each run ~8x faster while preserving strategy evaluation fidelity
#     since daily strategies run on daily bars anyway, and hourly strategy
#     signals are re-evaluated at each daily bar using the full hourly history) ---
all_timestamps = set()
for sym in bt.SYMBOLS:
    if sym in all_daily: all_timestamps.update(all_daily[sym].index)
sorted_timestamps = sorted(all_timestamps)

current_cash   = 100_000.0
open_positions = {}
trade_history  = []
peak_equity    = current_cash
portfolio_vals = []

for ts in sorted_timestamps:
    # Mark to market
    total_equity = current_cash
    for sym, pos in open_positions.items():
        price = 0
        if sym in all_daily and ts in all_daily[sym].index:
            price = all_daily[sym].loc[ts, 'close']
        if price > 0:
            total_equity += price * pos['qty']

    peak_equity    = max(peak_equity, total_equity)
    risk_mult      = bt.compute_risk_multiplier(total_equity, peak_equity)
    portfolio_vals.append(total_equity)

    for symbol in bt.SYMBOLS:
        daily_up  = all_daily.get(symbol,  pd.DataFrame())
        hourly_up = all_hourly.get(symbol, pd.DataFrame())
        if not daily_up.empty:  daily_up  = daily_up[daily_up.index   <= ts]
        if not hourly_up.empty: hourly_up = hourly_up[hourly_up.index <= ts]

        latest_price = None
        if not hourly_up.empty: latest_price = hourly_up['close'].iloc[-1]
        elif not daily_up.empty: latest_price = daily_up['close'].iloc[-1]
        if latest_price is None: continue

        # ATR
        latest_atr = None
        if len(daily_up) > bt.ATR_PERIOD:
            atr_s = ta.volatility.average_true_range(
                daily_up['high'], daily_up['low'], daily_up['close'], window=bt.ATR_PERIOD
            ).dropna()
            if not atr_s.empty: latest_atr = atr_s.iloc[-1]

        signals = []
        bars_by_tf = {"daily": daily_up, "hourly": hourly_up}
        for strategy, tf in bt.STRATEGIES:
            bars = bars_by_tf[tf]
            if bars.empty or len(bars) < strategy.min_bars_required: continue
            sig = strategy.generate_signal(symbol, bars)
            signals.append(sig)
            trend_sma = sig.details.get("trend_sma")
            if trend_sma is not None:
                if sig.action == BUY  and latest_price < trend_sma: sig.action = HOLD
                if sig.action == SELL and latest_price > trend_sma: sig.action = HOLD

        if symbol not in open_positions and len(open_positions) >= bt.MAX_SIMULTANEOUS_POSITIONS:
            for sig in signals: sig.action = HOLD

        current_qty = open_positions[symbol]['qty'] if symbol in open_positions else 0
        chosen      = choose_signal_to_act_on(signals, current_qty)
        sig_action  = chosen.action if chosen else HOLD
        sig_strat   = chosen.strategy_name if chosen else None

        current_cash, open_positions, td = bt.execute_trade_logic_backtest(
            symbol, sig_action, sig_strat, latest_price, ts,
            current_cash, open_positions, total_equity, latest_atr, risk_mult
        )
        if td: trade_history.append(td)

# --- Final mark-to-market ---
final_equity = current_cash
for sym, pos in open_positions.items():
    if sym in all_daily and not all_daily[sym].empty:
        final_equity += all_daily[sym]['close'].iloc[-1] * pos['qty']

net_pl  = final_equity - 100_000.0
net_pct = net_pl / 100_000.0 * 100

# Max drawdown from portfolio_vals series
import numpy as np
pv   = pd.Series(portfolio_vals)
roll = pv.cummax()
dd   = ((pv - roll) / roll).min() * 100

# Per-strategy stats
closed = [t for t in trade_history if t['P/L'] != 0.0]
by_strat = {}
for t in closed:
    s = t.get('strategy', 'UNKNOWN')
    by_strat.setdefault(s, []).append(t['P/L'])

print(f"\n{'='*60}")
print(f"COMBO: {label}")
print(f"  ATR Sizing: {atr_size}  |  ATR Stops: {atr_stop}  |  Throttle: {throttle}")
print(f"  Net P/L:      {net_pct:+.2f}%  (${net_pl:,.2f})")
print(f"  Max Drawdown: {dd:.2f}%")
print(f"  Closed Trades:{len(closed)}")
print(f"  Per-strategy breakdown:")
for strat, pls in sorted(by_strat.items(), key=lambda x: -sum(x[1])):
    wins = sum(1 for p in pls if p > 0)
    print(f"    {strat:<22} {len(pls):4d} trades  ${sum(pls):>10,.2f}  win%: {wins/len(pls)*100:.1f}")
print(f"{'='*60}")
sys.stdout.flush()
