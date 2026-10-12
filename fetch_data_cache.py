"""Fetch all bar data once and save to a pickle cache for fast experiment replay."""
import pickle, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import backtester as bt

all_daily  = {}
all_hourly = {}

for symbol in bt.SYMBOLS:
    logging.info(f"Fetching {symbol} daily...")
    d = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.TIME_FRAME)
    if not d.empty: all_daily[symbol] = d

    logging.info(f"Fetching {symbol} hourly...")
    h = bt.get_historical_bars(symbol, bt.BACKTEST_START_DATE, bt.BACKTEST_END_DATE, bt.SCALPING_TIME_FRAME)
    if not h.empty: all_hourly[symbol] = h

with open('/tmp/bt_data_cache.pkl', 'wb') as f:
    pickle.dump({'daily': all_daily, 'hourly': all_hourly}, f)

logging.info(f"Saved cache: {len(all_daily)} daily, {len(all_hourly)} hourly symbols.")
