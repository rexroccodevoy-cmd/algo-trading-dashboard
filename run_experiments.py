"""
Experiment runner: tests all 8 combinations of three drawdown-control levers:
  Lever 1: ATR-based position sizing   (USE_ATR_POSITION_SIZING)
  Lever 2: ATR-based trailing stops    (USE_ATR_STOPS)
  Lever 3: Portfolio drawdown throttle (USE_DRAWDOWN_THROTTLE)

Each run overwrites the flags in backtester.py's globals, runs the backtest,
and prints a consolidated summary table at the end.

This is an example of a "configuration-sweep" experiment pattern common in
quant research: rather than running each variant by hand and comparing logs,
you automate the matrix so all 8 runs happen in one session with identical
random/market conditions (same data, same code) and you get a clean
apples-to-apples comparison.

Why 8? Because each lever is binary (on/off), and 2^3 = 8 combinations.
"""

import sys
import io
import logging
import importlib
import pandas as pd
import itertools

import backtester as bt

COMBINATIONS = list(itertools.product([False, True], repeat=3))

def label(atr_size, atr_stop, throttle):
    parts = []
    if atr_size:   parts.append("AtrSize")
    if atr_stop:   parts.append("AtrStop")
    if throttle:   parts.append("Throttle")
    return "+".join(parts) if parts else "Baseline"

def run_one(atr_size, atr_stop, throttle):
    """Set the three flags on the backtester module and re-run the simulation
    using the already-fetched bar data cached in `bt.all_daily_bars` and
    `bt.all_hourly_bars`.

    Re-importing isn't needed because the simulation logic is in
    `execute_trade_logic_backtest` and the main block -- we replicate the
    main block's simulation loop inline here so we can inject flag overrides
    cleanly without subprocesses.
    """
    import ta
    from strategies import BUY, SELL, HOLD, SmaCrossoverStrategy, choose_signal_to_act_on

    # --- Apply flags ---
    bt.USE_ATR_POSITION_SIZING = atr_size
    bt.USE_ATR_STOPS           = atr_stop
    bt.USE_DRAWDOWN_THROTTLE   = throttle

    # --- Re-init ledgers ---
    current_cash  = 100_000.0
    open_positions = {}
    trade_history  = []
    peak_equity    = current_cash

    all_timestamps = set()
    for sym in bt.SYMBOLS:
        if sym in bt.all_daily_bars:  all_timestamps.update(bt.all_daily_bars[sym].index)
        if sym in bt.all_hourly_bars: all_timestamps.update(bt.all_hourly_bars[sym].index)
    sorted_timestamps = sorted(all_timestamps)

    for current_timestamp in sorted_timestamps:
        # Mark-to-market portfolio equity
        total_equity = current_cash
        for sym, pos in open_positions.items():
            price = 0
            if sym in bt.all_hourly_bars and current_timestamp in bt.all_hourly_bars[sym].index:
                price = bt.all_hourly_bars[sym].loc[current_timestamp, 'close']
            elif sym in bt.all_daily_bars and current_timestamp in bt.all_daily_bars[sym].index:
                price = bt.all_daily_bars[sym].loc[current_timestamp, 'close']
            if price > 0:
                total_equity += price * pos['qty']

        peak_equity = max(peak_equity, total_equity)
        risk_multiplier = bt.compute_risk_multiplier(total_equity, peak_equity)

        for symbol in bt.SYMBOLS:
            daily_up  = bt.all_daily_bars.get(symbol, pd.DataFrame())
            hourly_up = bt.all_hourly_bars.get(symbol, pd.DataFrame())

            if not daily_up.empty:
                daily_up  = daily_up[daily_up.index   <= current_timestamp]
            if not hourly_up.empty:
                hourly_up = hourly_up[hourly_up.index <= current_timestamp]

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
                if not atr_s.empty:
                    latest_atr = atr_s.iloc[-1]

            signals = []
            bars_by_tf = {"daily": daily_up, "hourly": hourly_up}
            for strategy, tf in bt.STRATEGIES:
                bars = bars_by_tf[tf]
                if bars.empty or len(bars) < strategy.min_bars_required:
                    continue
                sig = strategy.generate_signal(symbol, bars)
                signals.append(sig)
                trend_sma = sig.details.get("trend_sma")
                if trend_sma is not None:
                    if sig.action == BUY  and latest_price < trend_sma: sig.action = HOLD
                    if sig.action == SELL and latest_price > trend_sma:  sig.action = HOLD

            if symbol not in open_positions and len(open_positions) >= bt.MAX_SIMULTANEOUS_POSITIONS:
                for sig in signals: sig.action = HOLD

            current_qty = open_positions[symbol]['qty'] if symbol in open_positions else 0
            chosen = choose_signal_to_act_on(signals, current_qty)
            signal_to_act_on = chosen.action if chosen else HOLD
            strategy_name    = chosen.strategy_name if chosen else None

            current_cash, open_positions, td = bt.execute_trade_logic_backtest(
                symbol, signal_to_act_on, strategy_name, latest_price,
                current_timestamp, current_cash, open_positions,
                total_equity, latest_atr, risk_multiplier
            )
            if td:
                trade_history.append(td)

    # --- Compute final equity including open positions ---
    final_equity = current_cash
    for sym, pos in open_positions.items():
        last_price = 0
        if sym in bt.all_daily_bars and not bt.all_daily_bars[sym].empty:
            last_price = bt.all_daily_bars[sym]['close'].iloc[-1]
        elif sym in bt.all_hourly_bars and not bt.all_hourly_bars[sym].empty:
            last_price = bt.all_hourly_bars[sym]['close'].iloc[-1]
        if last_price > 0:
            final_equity += last_price * pos['qty']

    net_pl   = final_equity - 100_000.0
    net_pct  = net_pl / 100_000.0 * 100

    # Max drawdown from trade history via portfolio equity isn't tracked per-bar here,
    # so we approximate from the mark-to-market implied by open position values recorded
    # at each timestamp. Use bt's backtester drawdown logic approach: just report net P/L.
    # (The full drawdown calc requires portfolio_history -- we'll use the bt module for
    # individual runs; this runner is just for the comparison table.)
    closed = [t for t in trade_history if t['P/L'] != 0.0]
    n_trades = len(closed)
    by_strategy = {}
    for t in closed:
        s = t.get('strategy', 'UNKNOWN')
        by_strategy.setdefault(s, []).append(t['P/L'])

    return {
        'label': label(atr_size, atr_stop, throttle),
        'net_pct': net_pct,
        'net_pl': net_pl,
        'n_trades': n_trades,
        'by_strategy': {s: (len(v), sum(v)) for s, v in by_strategy.items()},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("Fetching data (once)...")
    bt.all_daily_bars  = {}
    bt.all_hourly_bars = {}
    for symbol in bt.SYMBOLS:
        d = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.TIME_FRAME)
        if not d.empty: bt.all_daily_bars[symbol] = d
        h = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.SCALPING_TIME_FRAME)
        if not h.empty: bt.all_hourly_bars[symbol] = h

    print(f"Data ready. Running {len(COMBINATIONS)} experiment combinations...\n")

    results = []
    for combo in COMBINATIONS:
        atr_size, atr_stop, throttle = combo
        name = label(atr_size, atr_stop, throttle)
        print(f"  Running: {name}...")
        r = run_one(atr_size, atr_stop, throttle)
        results.append(r)
        print(f"    => {r['net_pct']:+.2f}%  ({r['n_trades']} closed trades)")

    print("\n" + "="*72)
    print(f"{'Combination':<30} {'Net P/L %':>10} {'Net $':>12} {'Trades':>8}")
    print("-"*72)
    for r in sorted(results, key=lambda x: x['net_pct'], reverse=True):
        print(f"  {r['label']:<28} {r['net_pct']:>+9.2f}%  ${r['net_pl']:>10,.2f}  {r['n_trades']:>6}")
    print("="*72)

    print("\nPer-strategy breakdown for each combination:")
    for r in sorted(results, key=lambda x: x['net_pct'], reverse=True):
        print(f"\n  {r['label']}:")
        for strat, (n, pl) in sorted(r['by_strategy'].items(), key=lambda x: -x[1][1]):
            print(f"    {strat:<22} {n:4d} trades  ${pl:>10,.2f}")
