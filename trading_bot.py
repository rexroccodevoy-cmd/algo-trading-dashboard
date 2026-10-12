import os
import time
import pandas as pd
import ta
import logging
import uuid
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError # ADD APIError here
from strategies import (
    BUY, SELL, HOLD,
    SmaCrossoverStrategy,
    MacdMomentumStrategy,
    RsiMomentumStrategy,
    BollingerBandReversionStrategy,
    RocMomentumStrategy,
    AtrChannelBreakoutStrategy,
    VwapReversionStrategy,
    OpeningRangeBreakoutStrategy,
    EmaScalpStrategy,
    FastRsiScalpStrategy,
    choose_signal_to_act_on,
)
from alerts import send_alert

# --- Configuration ---
# Configure logging for the bot
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Get Alpaca API keys from environment variables
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

# Initialize Alpaca API client
if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL:
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    logging.info("Connected to Alpaca API for autonomous bot.")
else:
    logging.error("Alpaca API keys not found. Please check your .env file. Exiting bot.")
    exit() # Exit if API keys are missing

# --- Trading Strategy Parameters (customize these!) ---
# Function to load symbols from a file
def load_symbols(path="symbols.txt"):
    """Loads a list of trading symbols from a text file."""
    try:
        with open(path, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
        if not symbols:
            logging.warning(f"No symbols found in {path}. Please add symbols to trade.")
            return ["SPY"] # Default to SPY if file is empty
        return symbols
    except FileNotFoundError:
        logging.error(f"'{path}' not found. Please create this file with one symbol per line. Defaulting to SPY.")
        return ["SPY"]
    except Exception as e:
        logging.error(f"Error loading symbols from {path}: {e}. Defaulting to SPY.")
        return ["SPY"]

SYMBOLS = load_symbols("symbols.txt")

# 3. Restore the original get_latest_bars function.
# Delete the get_latest_bars_batch function and replace it with this version,
# which includes our Quant data integrity checks.

def get_latest_bars(symbol, timeframe_days, time_frame, feed='iex'):
    """Fetches historical bars for a single symbol with stability checks."""
    end_date = pd.Timestamp.now(tz='America/New_York').date()
    start_date = end_date - pd.Timedelta(days=timeframe_days)

    try:
        bars = api.get_bars(
            symbol,
            time_frame,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            feed=feed,
            adjustment='split'
        ).df

        if bars.empty:
            return pd.DataFrame()

        # Quant Data Integrity
        bars = bars.sort_index()
        bars = bars[bars['close'] > 0]

        return bars

    except Exception as e:
        logging.error(f"{symbol}: Error fetching bars: {e}")
        return pd.DataFrame()

FAST_SMA_PERIOD = 20 # Changed back to 20, as 5 might be too fast for daily/hourly. Adjust as needed.
SLOW_SMA_PERIOD = 50 # Changed back to 50, as 15 might be too fast for daily/hourly. Adjust as needed.

PORTFOLIO_ALLOCATION_PCT = 0.05 # Fallback fixed-allocation sizing (only used when ATR sizing is off)
TAKE_PROFIT_PCT = 0.12
TRAILING_STOP_PCT = 0.04        # Fallback flat trailing stop (only used when ATR stops are off)
MIN_TRADE_QTY = 1
MAX_SIMULTANEOUS_POSITIONS = 8  # Backtested: cap=3 -> -17% DD; cap=8 -> -36% but 2.7x return.

# --- Risk-management levers (backtested 2026-06-09 across 8 combinations) ---
# Winner: ATR Sizing + ATR Stops -> +75.6% return, -11.4% drawdown
# (vs Baseline +11.3%, -1.9% on the same daily-bar timeline)
ATR_PERIOD = 14                 # lookback window for Average True Range

# Lever 1: ATR-based position sizing.
# Sizes each trade so a stop-out costs ~RISK_PER_TRADE_PCT of equity regardless
# of how volatile the stock is. High-ATR stocks get fewer shares, low-ATR stocks
# get more -- every position risks approximately the same dollar amount.
USE_ATR_POSITION_SIZING = True
RISK_PER_TRADE_PCT = 0.01       # risk ~1% of equity per trade if stop is hit
ATR_STOP_MULTIPLE = 2.5         # stop distance = ATR * this (must match ATR stops below)

# ATR sizing's blind spot (seen live 2026-06-10: UNH was 52% of the book):
# it caps loss-to-STOP, not loss-to-GAP. A low-ATR stock gets a huge share
# count because its stop is tight -- but if it gaps 15% overnight (earnings,
# news) the stop never executes anywhere near its level. Cap notional per
# position so no single overnight gap can do outsized damage.
MAX_POSITION_PCT = 0.15         # no single position may exceed 15% of equity

# And the opposite failure: when buying power is nearly exhausted, sizing
# rounds down to 1 share ($80-$120 "dust" positions). These can't earn
# anything meaningful but still occupy one of MAX_SIMULTANEOUS_POSITIONS
# slots and pay commission twice. Below this notional, skip the trade.
MIN_POSITION_NOTIONAL = 1000.0  # minimum $ size for a new position

# Lever 2: ATR-based trailing stops.
# Stop distance is volatility-adjusted (ATR * multiple) instead of a flat %.
# Volatile stocks get more breathing room; calm stocks get a tighter leash.
# Fewer premature stop-outs on normal intraday swings.
USE_ATR_STOPS = True

# Lever 3: Portfolio drawdown throttle (recalibrated to reachable thresholds).
# Experiment showed -15%/-25% never triggered on our strategies (max DD ~12%).
# Recalibrated to -5%/-10% so the circuit breaker is actually live.
USE_DRAWDOWN_THROTTLE = True
DRAWDOWN_THROTTLE_LEVELS = [
    (-0.10, 0.0),   # 10%+ below peak equity: stop opening new positions
    (-0.05, 0.5),   # 5%+ below peak equity: halve new position sizes
]
# Lever 4: Market regime filter.
# Mean-reversion strategies (Bollinger, RSI) are suppressed when the broad
# market is in a downtrend. We use SPY vs its 200-day SMA as the regime
# indicator: if SPY price < SMA(200), we're in a "bear" regime and only
# trend-following strategies are allowed to open new long positions.
#
# Why? Mean-reversion assumes prices oscillate around a stable average.
# In a bear market or strong downtrend that assumption breaks down: "oversold"
# just means the stock is in freefall. Filtering these signals avoids the
# pattern of buying every dip only to watch it make a new low.
#
# SELL signals from mean-reversion strategies are NOT filtered in a bear regime
# — exiting longs or shorting into weakness is still fine.
USE_REGIME_FILTER = True
REGIME_SYMBOL     = "SPY"          # broad-market proxy
REGIME_SMA_PERIOD = 200            # SPY 200-day SMA is the most-watched trend line

CHECK_INTERVAL_SECONDS = 60  # 1-minute cycle to keep pace with minute-bar strategies

# Ensure enough data for SMAs + buffer for weekends/holidays/missing data
TREND_FILTER_PERIOD = 200 # Period for long-term trend filter
# DATA_LOOKBACK_DAYS = 300  # Request enough historical bars to calculate 200 SMA
DAILY_LOOKBACK_DAYS = 300
MINUTE_LOOKBACK_DAYS = 2    # 2 days is plenty for all minute-bar strategies

TIME_FRAME = TimeFrame.Day
MINUTE_TIME_FRAME = TimeFrame.Minute

# --- Momentum Scalping Strategy Parameters ---
SCALPING_TIME_FRAME = TimeFrame.Hour # Shorter timeframe for frequent trades
RSI_PERIOD = 14                     # Standard RSI calculation period
RSI_OVERBOUGHT = 70                 # RSI level to consider overbought
RSI_OVERSOLD = 30                   # RSI level to consider oversold
SCALPING_SMA_PERIOD = 50            # SMA filter for scalping trend
SCALPING_TAKE_PROFIT_PCT = 0.0075   # 0.75% Take Profit for quick scalps
SCALPING_STOP_LOSS_PCT = 0.015      # 1.5% Stop Loss for quick scalps
SCALPING_LOOKBACK_DAYS = 60         # Need enough data for RSI and SMA on hourly

# --- Strategy Framework ---
# Each entry pairs a Strategy instance with the bar timeframe it should run on,
# in PRIORITY ORDER: when multiple strategies fire at once, the earlier one in
# this list wins (fast/frequent strategies first, slow trend-followers last).
# To add a new strategy: write a class in strategies.py, then add one line here.
# Backtested over 2025 on our 12-symbol universe (see backtester.py):
#   Bollinger (2.0sigma) + RSI Momentum (45/65)  -> +2.41% net, -16.96% max drawdown
# SMA Crossover fired once all year and lost money -- disabled as dead weight.
# MACD Momentum (added 2026-06-08): backtested on the 24-symbol/cap=8 universe,
# it was the single best strategy -- $8,539 P/L, 77.9% win rate -- and adding it
# *raised* total return (5.95% -> 16.17%) while *lowering* drawdown
# (-35.68% -> -32.96%). It trades on daily bars using a different kind of
# signal (rate-of-change of trend via EMA crossovers) than our hourly
# mean-reversion strategies, so it diversifies rather than competing.
# Re-enable / retune SMA via backtester.py before trusting it with new capital.
STRATEGIES = [
    # --- HFT intraday strategies (1-minute bars, fastest → highest priority) ---
    (FastRsiScalpStrategy(rsi_period=3, oversold=15, overbought=85), "minute"),
    (EmaScalpStrategy(fast=5, slow=13), "minute"),
    (VwapReversionStrategy(std_dev_multiplier=1.5, min_bars=30), "minute"),
    (OpeningRangeBreakoutStrategy(or_minutes=30), "minute"),
    # --- Hourly mean-reversion strategies ---
    (BollingerBandReversionStrategy(window=20, num_std=2.0), "hourly"),
    (RsiMomentumStrategy(
        rsi_period=RSI_PERIOD,
        oversold=RSI_OVERSOLD,
        overbought=RSI_OVERBOUGHT,
        trend_sma_period=SCALPING_SMA_PERIOD,
    ), "hourly"),
    # --- Daily trend-following strategies ---
    (MacdMomentumStrategy(), "daily"),
    # ATR Channel kept after walk-forward validation (2026-06-10): +$2,190
    # net over 2023-25, positive in 2/3 years.
    (AtrChannelBreakoutStrategy(), "daily"),
]



def get_latest_bars_batch(symbols_list, start_date_str, end_date_str, time_frame, feed='iex'):
    """
    Fetches historical bars for MULTIPLE symbols in a single batch call.
    Returns a dictionary of {symbol: DataFrame}.
    """
    all_bars_dfs = {}

    if not symbols_list:
        return all_bars_dfs

    try:
        start_ts = pd.Timestamp(start_date_str, tz='America/New_York')
        end_ts = pd.Timestamp(end_date_str, tz='America/New_York')

        raw_bars_data = api.get_bars(
            symbols_list,
            time_frame,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            feed=feed,
            adjustment='split'
        )

        df_bars = raw_bars_data.df

        if df_bars.empty:
            logging.warning(f"Batch fetch returned an empty DataFrame for {time_frame}.")
            return all_bars_dfs

        # Quant Principle: Robust grouping across alpaca-trade-api versions.
        # Depending on SDK version, multi-symbol results come back either as a
        # MultiIndex (symbol, timestamp) or as a plain DatetimeIndex with a
        # 'symbol' column -- handle both so this keeps working across upgrades.
        has_symbol_index_level = (
            isinstance(df_bars.index, pd.MultiIndex) and 'symbol' in df_bars.index.names
        )
        has_symbol_column = 'symbol' in df_bars.columns

        if has_symbol_index_level or has_symbol_column:
            group_key = {'level': 'symbol'} if has_symbol_index_level else {'by': 'symbol'}
            for symbol_val, group_df in df_bars.groupby(**group_key):
                bars = group_df.droplevel('symbol') if has_symbol_index_level else group_df.drop(columns='symbol')

                if bars.empty:
                    logging.warning(
                        f" {symbol_val}: Empty DataFrame after grouping for {time_frame}. Skipping."
                    )
                    continue

                # Apply data integrity checks
                bars = bars.sort_index()
                bars = bars[bars['close'] > 0]

                if bars.empty:
                    logging.warning(
                        f" {symbol_val}: No valid bars after filtering for {time_frame}. Skipping."
                    )
                    continue

                # Check for stale data (TimeFrame specific)
                if time_frame in [TimeFrame.Hour, TimeFrame.Day] and not bars.empty:
                    last_bar_time = bars.index[-1]
                    now_eastern = pd.Timestamp.now(tz='America/New_York')
                    time_difference = (
                        now_eastern - last_bar_time.tz_convert('America/New_York')
                    )

                    if time_difference.days > 3 and now_eastern.weekday() < 5:
                        logging.warning(
                            f" {symbol_val}: Data latency anomaly! Latest bar timestamp "
                            f"({last_bar_time}) is severely delayed."
                        )

                # Liquidity Filter (Analyze Average Volume)
                if symbol_val not in ["SPY", "QQQ"]:
                    average_volume = bars['volume'].tail(20).mean()

                    if average_volume < 1000:
                        logging.warning(
                            f" {symbol_val}: Rejecting from batch. Average volume "
                            f"({average_volume:.0f}) is too low for safe execution."
                        )
                        continue  # Skip this symbol for lack of liquidity

                all_bars_dfs[symbol_val] = bars

        else:
            # Fallback if not a MultiIndex or 'symbol' level not found
            # (e.g., single symbol, or older pandas behavior)
            # This case should ideally not happen for multiple symbols.
            logging.error(
                f" Batch fetch for {time_frame} returned unexpected DataFrame "
                f"structure (not MultiIndex with 'symbol' level). "
                f"Attempting direct processing if single symbol."
            )

            if len(symbols_list) == 1 and not df_bars.empty:
                symbol_val = symbols_list[0]
                bars = df_bars.sort_index()
                bars = bars[bars['close'] > 0]

                if not bars.empty:
                    all_bars_dfs[symbol_val] = bars
                    # (Simplified: assume other checks handled by single-symbol call
                    # or less critical for one symbol)

            else:
                logging.error(
                    f" Could not process batch DataFrame for {time_frame}. "
                    f"Likely empty or unexpected format for multiple symbols."
                )

        return all_bars_dfs  # Return whatever valid dataframes were collected

    except Exception as e:
        logging.error(
            f" Batch Data Feed Handler Critical error for {time_frame}: {e}"
        )
        return {}

def calculate_dynamic_qty(symbol, latest_price, equity, buying_power, atr_val=None, risk_multiplier=1.0):
    """How many shares to buy for a new position.

    Two modes, selected by USE_ATR_POSITION_SIZING:

    Fixed-allocation (off): spend a constant % of equity. Simple, but blind
    to volatility -- a NVDA position and a KO position both get the same
    dollar amount even though NVDA can swing 5x as much per day.

    ATR sizing (on): size so that IF the stop-loss triggers, the loss is
    approximately RISK_PER_TRADE_PCT of equity -- regardless of which stock.
        shares = (equity * RISK_PCT) / (ATR * stop_multiple)
    A volatile stock has a bigger ATR -> larger stop distance -> fewer shares.
    A calm stock has a smaller ATR -> more shares. Both risk the same dollars.

    risk_multiplier is injected by the portfolio drawdown throttle: 0.5 means
    "only take half-sized positions right now", 0.0 means "don't open at all".
    """
    try:
        if risk_multiplier <= 0:
            return 0

        if USE_ATR_POSITION_SIZING and atr_val is not None and atr_val > 0:
            dollar_risk = equity * RISK_PER_TRADE_PCT * risk_multiplier
            stop_distance = atr_val * ATR_STOP_MULTIPLE
            calculated_qty = int(dollar_risk // stop_distance)
        else:
            target_cash = equity * PORTFOLIO_ALLOCATION_PCT * risk_multiplier
            cash_to_use = min(target_cash, buying_power)
            calculated_qty = int(cash_to_use // latest_price)

        # Cap notional exposure: ATR sizing limits loss-to-stop, but a huge
        # share count in a calm stock is still huge gap risk. (See constants.)
        if latest_price > 0:
            max_notional_qty = int((equity * MAX_POSITION_PCT) // latest_price)
            calculated_qty = min(calculated_qty, max_notional_qty)

        # Never spend more than we actually have
        affordable = int(buying_power // latest_price) if latest_price > 0 else 0
        calculated_qty = min(calculated_qty, affordable)

        # Dust filter: a position too small to matter still burns a position
        # slot and two commissions. Skip rather than force a 1-share trade.
        if calculated_qty * latest_price < MIN_POSITION_NOTIONAL:
            logging.info(
                f"{symbol}: Sizing -> SKIP. Computed {calculated_qty} shares "
                f"(${calculated_qty * latest_price:,.2f}) is below the "
                f"${MIN_POSITION_NOTIONAL:,.0f} minimum position size."
            )
            return 0

        logging.info(
            f"{symbol}: Sizing -> {calculated_qty} shares @ ${latest_price:.2f} "
            f"(${calculated_qty * latest_price:,.2f} notional) "
            f"| ATR={atr_val if atr_val is not None else 'N/A'} | multiplier={risk_multiplier:.2f}"
        )
        return calculated_qty

    except Exception as e:
        # Fail SAFE: if sizing itself errors, don't trade at all. The old
        # fallback (return 1 share) turned this exception path into a silent
        # trading mode -- a broken log line here meant the live bot traded
        # 4,361 consecutive 1-share orders while ATR sizing never ran once.
        logging.error(f"{symbol}: Error calculating dynamic quantity: {e}. Skipping trade.")
        return 0


def compute_risk_multiplier(current_equity, peak_equity):
    """Scales down new-position sizes as portfolio sinks below its all-time high.

    Returns 1.0 (full size) when healthy, 0.5 at -5% drawdown, 0.0 at -10%.
    The purpose isn't to prevent the first loss -- it's to prevent a bad
    stretch from compounding into a catastrophic one by automatically reducing
    exposure when the strategies are underperforming.
    """
    if not USE_DRAWDOWN_THROTTLE or peak_equity <= 0:
        return 1.0
    drawdown = (current_equity - peak_equity) / peak_equity
    for trigger, multiplier in DRAWDOWN_THROTTLE_LEVELS:
        if drawdown <= trigger:
            return multiplier
    return 1.0

def place_order(symbol, qty, side, reason):
    """Submit a market order AND send a Slack alert about it.

    Single chokepoint for all order submission: every trade the bot makes
    goes through here, so logging and alerting can never drift out of sync
    with actual orders (the failure mode of sprinkling alerts at call sites).
    Raises on API errors -- callers' existing try/except blocks handle those.
    """
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='day',
        client_order_id=str(uuid.uuid4()),
    )
    logging.info(f"{symbol}: {side.upper()} order submitted for {qty} shares. Reason: {reason}")
    send_alert(f"TRADE: {side.upper()} {qty} {symbol} | {reason}")


def compute_market_regime(spy_daily_bars):
    """Determine whether the broad market is in a 'bull' or 'bear' regime.

    Method: compare the latest SPY closing price to its 200-day SMA.
      - price > SMA(200): 'bull' — trend-following AND mean-reversion strategies active
      - price < SMA(200): 'bear' — mean-reversion BUY signals suppressed

    The 200-day SMA is the single most-watched trend line in professional asset
    management. A stock or index trading below its 200-SMA is, by the standard
    institutional definition, in a downtrend.  We don't need anything fancier.

    Returns 'bull' on any data failure (fail-safe: over-filtering costs money
    too — we'd rather miss a bear signal occasionally than turn the regime
    filter into a permanent off-switch due to a data glitch).
    """
    if not USE_REGIME_FILTER:
        return 'bull'
    if spy_daily_bars is None or spy_daily_bars.empty:
        logging.warning("Regime filter: SPY bars unavailable, defaulting to 'bull'.")
        return 'bull'
    if len(spy_daily_bars) < REGIME_SMA_PERIOD:
        logging.warning(
            f"Regime filter: only {len(spy_daily_bars)} SPY bars, "
            f"need {REGIME_SMA_PERIOD} for SMA. Defaulting to 'bull'."
        )
        return 'bull'

    sma = ta.trend.sma_indicator(spy_daily_bars['close'], window=REGIME_SMA_PERIOD).dropna()
    if sma.empty:
        return 'bull'

    latest_price = spy_daily_bars['close'].iloc[-1]
    latest_sma   = sma.iloc[-1]
    regime = 'bull' if latest_price > latest_sma else 'bear'
    logging.info(
        f"Market regime: {regime.upper()} | "
        f"SPY ${latest_price:.2f} vs 200-SMA ${latest_sma:.2f} "
        f"({'above' if regime == 'bull' else 'BELOW'})"
    )
    return regime


def calculate_sma_crossover_signals(symbol, bars, fast_period, slow_period):
    """Calculates Fast/Slow SMA and generates buy/sell signals."""
    # Ensure enough data for calculation. +1 for .shift(-1) later for proper signal detection if using current bar
    # For .iloc[-1] and .iloc[-2] comparison, we need at least max(periods) + 1 for the SMAs themselves, then +1 more for the 'previous' bar.
    min_bars_needed = max(fast_period, slow_period) + 2
    if bars.empty or len(bars) < min_bars_needed:
        logging.warning(f"{symbol}: Not enough data ({len(bars)} bars) for SMA calculation (needs at least {min_bars_needed} bars).")
        return "NO_DATA", None, None, None, None # Not enough data for calculation

    # Ensure bars are sorted by date
    bars = bars.sort_index()

    # Calculate fast, slow, and 200-day trend SMA
    bars['Fast_SMA'] = ta.trend.sma_indicator(bars['close'], window=fast_period)
    bars['Slow_SMA'] = ta.trend.sma_indicator(bars['close'], window=slow_period)
    bars['Trend_SMA'] = ta.trend.sma_indicator(bars['close'], window=TREND_FILTER_PERIOD)
    bars['ATR'] = ta.volatility.average_true_range(bars['high'], bars['low'], bars['close'], window=ATR_PERIOD)

    bars.dropna(subset=['Fast_SMA', 'Slow_SMA', 'Trend_SMA', 'ATR'], inplace=True)

    if bars.empty or len(bars) < 2: # Need at least 2 data points for crossover logic after dropping NaNs
        logging.warning(f"{symbol}: Not enough data after dropping NaNs for trend filter calculation.")
        return "NO_DATA", None, None, None, None

    # Get the latest and previous values for crossover detection
    latest_fast_sma = bars['Fast_SMA'].iloc[-1]
    latest_slow_sma = bars['Slow_SMA'].iloc[-1]
    prev_fast_sma = bars['Fast_SMA'].iloc[-2]
    prev_slow_sma = bars['Slow_SMA'].iloc[-2]
    latest_trend_sma = bars['Trend_SMA'].iloc[-1] # Get latest 200 SMA

    # Get the latest and previous values for crossover detection
    latest_fast_sma = bars['Fast_SMA'].iloc[-1]
    latest_slow_sma = bars['Slow_SMA'].iloc[-1]
    prev_fast_sma = bars['Fast_SMA'].iloc[-2]
    prev_slow_sma = bars['Slow_SMA'].iloc[-2]
    latest_trend_sma = bars['Trend_SMA'].iloc[-1]
    latest_atr = bars['ATR'].iloc[-1]  # Ensure this is calculated and available

    signal = "HOLD"

    # Check for crossover conditions
    if prev_fast_sma < prev_slow_sma and latest_fast_sma >= latest_slow_sma:
        signal = "BUY"
    elif prev_fast_sma > prev_slow_sma and latest_fast_sma <= latest_slow_sma:
        signal = "SELL"

    return (
        signal,
        latest_fast_sma,
        latest_slow_sma,
        latest_trend_sma,
        latest_atr,
    )  # Single, consistent return


def calculate_momentum_scalping_signals(symbol, bars):
    """Calculates RSI-based buy/sell signals with an SMA trend filter."""
    min_bars_needed = max(RSI_PERIOD, SCALPING_SMA_PERIOD) + 1  # Need enough bars for both indicators

    if bars.empty or len(bars) < min_bars_needed:
        logging.warning(
            f"{symbol}: Not enough data ({len(bars)} bars) for Momentum Scalping calculation "
            f"(needs at least {min_bars_needed} bars)."
        )
        return "NO_DATA", None, None, None, None

    bars = bars.sort_index()

    # Calculate RSI
    bars['RSI'] = ta.momentum.rsi(bars['close'], window=RSI_PERIOD)

    # Calculate SMA filter
    bars['Scalping_SMA'] = ta.trend.sma_indicator(
        bars['close'],
        window=SCALPING_SMA_PERIOD
    )

    bars.dropna(subset=['RSI', 'Scalping_SMA'], inplace=True)

    if bars.empty:
        logging.warning(
            f"{symbol}: Not enough data after dropping NaNs for Momentum Scalping signals."
        )
        return "NO_DATA", None, None, None, None

    latest_close = bars['close'].iloc[-1]
    latest_rsi = bars['RSI'].iloc[-1]
    latest_scalping_sma = bars['Scalping_SMA'].iloc[-1]

    signal = "HOLD"

    # Buy when RSI is oversold AND price is above SMA (uptrend for long entry)
    if latest_rsi < RSI_OVERSOLD and latest_close > latest_scalping_sma:
        signal = "BUY"

    # Sell when RSI is overbought AND price is below SMA (downtrend for short entry)
    elif latest_rsi > RSI_OVERBOUGHT and latest_close < latest_scalping_sma:
        signal = "SELL"

    return signal, latest_rsi, latest_scalping_sma

def get_current_position(symbol):
    """Checks if there's an open position for the given symbol."""
    try:
        position = api.get_position(symbol)
        return int(position.qty)  # Positive = long, negative = short
    except Exception as e:
        error_text = str(e).lower()
        # No position is normal, so don't log it as an error
        if (
            "position does not exist" in error_text
            or "no open position" in error_text
            or "not found" in error_text
        ):
            logging.debug(f"{symbol}: No open position found.") # Debug level for no position
            return 0
        # Only log genuine unexpected errors
        logging.error(f"{symbol}: Error checking position: {e}")
        return 0


def execute_trade_logic(symbol, signal_to_act_on, current_position, latest_price, account, position_details, atr_val=None, risk_multiplier=1.0):
    """Executes a trade for the already-arbitrated `signal_to_act_on`
    ("BUY", "SELL", or "HOLD"). Signal selection now happens upstream in
    run_strategy_for_symbol via choose_signal_to_act_on  this function only
    handles risk checks (stop-loss/take-profit) and order placement."""
    try:
        available_buying_power = float(account.buying_power)
        total_equity = float(account.equity)

        # Use appropriate allocation based on strategy (you could make this dynamic later)
        # For now, we use the main PORTFOLIO_ALLOCATION_PCT for all new trades
        qty_to_trade = calculate_dynamic_qty(
            symbol, latest_price, total_equity, available_buying_power,
            atr_val=atr_val, risk_multiplier=risk_multiplier
        )

        # --- TRAILING STOP-LOSS & TAKE-PROFIT CHECK (Highest Priority) ---
        if current_position != 0 and position_details is not None:
            try:
                avg_entry_price = float(position_details.avg_entry_price)
                current_price = float(position_details.current_price)

                high_water_mark = float(getattr(position_details, 'high_water_mark', avg_entry_price))
                low_water_mark = float(getattr(position_details, 'low_water_mark', avg_entry_price))

                if current_position > 0:  # LONG POSITION RULES
                    tp_price = avg_entry_price * (1 + TAKE_PROFIT_PCT)
                    if current_price >= tp_price:
                        logging.warning(
                            f"{symbol} TAKE-PROFIT TRIGGERED (LONG):"
                            f" ${current_price:.2f} >= target ${tp_price:.2f}. Closing."
                        )
                        place_order(symbol, abs(current_position), 'sell',
                                    f"Take-profit long @ ${current_price:.2f} (target ${tp_price:.2f})")
                        return

                    # ATR trailing stop: give volatile stocks more room, calm stocks less.
                    # Falls back to flat % if ATR not available.
                    if USE_ATR_STOPS and atr_val is not None:
                        stop_offset = atr_val * ATR_STOP_MULTIPLE
                        trail_stop_price = high_water_mark - stop_offset
                    else:
                        trail_stop_price = high_water_mark * (1 - TRAILING_STOP_PCT)
                    if current_price <= trail_stop_price:
                        logging.warning(
                            f"{symbol} TRAILING STOP TRIGGERED (LONG):"
                            f" ${current_price:.2f} <= stop ${trail_stop_price:.2f}. Closing."
                        )
                        place_order(symbol, abs(current_position), 'sell',
                                    f"Trailing stop long @ ${current_price:.2f} (stop ${trail_stop_price:.2f})")
                        return

                elif current_position < 0:  # SHORT POSITION RULES
                    tp_price = avg_entry_price * (1 - TAKE_PROFIT_PCT)
                    if current_price <= tp_price:
                        logging.warning(
                            f"{symbol} TAKE-PROFIT TRIGGERED (SHORT):"
                            f" ${current_price:.2f} <= target ${tp_price:.2f}. Covering."
                        )
                        place_order(symbol, abs(current_position), 'buy',
                                    f"Take-profit short @ ${current_price:.2f} (target ${tp_price:.2f})")
                        return

                    if USE_ATR_STOPS and atr_val is not None:
                        stop_offset = atr_val * ATR_STOP_MULTIPLE
                        trail_stop_price = low_water_mark + stop_offset
                    else:
                        trail_stop_price = low_water_mark * (1 + TRAILING_STOP_PCT)
                    if current_price >= trail_stop_price:
                        logging.warning(
                            f"{symbol} TRAILING STOP TRIGGERED (SHORT):"
                            f" ${current_price:.2f} >= stop ${trail_stop_price:.2f}. Covering."
                        )
                        place_order(symbol, abs(current_position), 'buy',
                                    f"Trailing stop short @ ${current_price:.2f} (stop ${trail_stop_price:.2f})")
                        return

            except APIError as pos_api_e:
                error_message = pos_api_e.args[0]
                if "insufficient qty available" in error_message.lower():
                    logging.warning(f"{symbol}: Trailing Stop/Take-Profit order failed (Position already closed?): {error_message}")
                else:
                    logging.error(f"{symbol}: Trailing Stop/Take-Profit order failed (Alpaca API): {error_message}")
                return
            except Exception as pos_err:
                logging.error(f"{symbol}: An unexpected error in Trailing Stop/Take-Profit check: {pos_err}")
                return
        # --- END OF TRAILING STOP-LOSS & TAKE-PROFIT CHECK ---

        # Now execute based on the chosen signal_to_act_on
        if signal_to_act_on == "BUY":
            if current_position == 0:  # No position, go long
                estimated_cost = qty_to_trade * latest_price
                if estimated_cost > available_buying_power:
                    logging.warning(
                        f"{symbol}: Insufficient buying power for BUY order. "
                        f"Requested: {qty_to_trade} shares (${estimated_cost:.2f}), "
                        f"Available: ${available_buying_power:.2f}. Skipping trade."
                    )
                    return
                place_order(symbol, qty_to_trade, 'buy',
                            f"New long @ ~${latest_price:.2f} (signal: {signal_to_act_on})")

            elif current_position < 0:  # Currently short, cover short and potentially go long
                qty_to_cover_and_long = abs(current_position) + qty_to_trade
                estimated_cost = qty_to_cover_and_long * latest_price

                if estimated_cost > available_buying_power:
                    if abs(current_position) * latest_price <= available_buying_power:
                        logging.warning(
                            f"{symbol}: Insufficient buying power to cover short and go long. "
                            f"Attempting to only cover the short position ({abs(current_position)} shares)."
                        )
                        qty_to_trade_adjusted = abs(current_position)
                    else:
                        logging.warning(
                            f"{symbol}: Insufficient buying power to even cover the short position ({abs(current_position)} shares). "
                            f"Requested total: {qty_to_cover_and_long} shares (${estimated_cost:.2f}), "
                            f"Available: ${available_buying_power:.2f}. Skipping trade."
                        )
                        return
                else:
                    qty_to_trade_adjusted = qty_to_cover_and_long

                if qty_to_trade_adjusted < MIN_TRADE_QTY:
                    logging.warning(f"{symbol}: Calculated BUY quantity {qty_to_trade_adjusted} is below MIN_TRADE_QTY ({MIN_TRADE_QTY}). Skipping trade.")
                    return

                place_order(symbol, qty_to_trade_adjusted, 'buy',
                            f"Cover short (and flip long) @ ~${latest_price:.2f}")
            else:  # current_position > 0 (already long)
                logging.info(f"{symbol}: Already long {current_position} shares on BUY signal. Holding.")

        elif signal_to_act_on == "SELL":
            if current_position > 0:  # Currently long, close long and potentially go short
                qty_to_sell_and_short = current_position + qty_to_trade

                if qty_to_sell_and_short < MIN_TRADE_QTY:
                    logging.warning(f"{symbol}: Calculated SELL quantity {qty_to_sell_and_short} is below MIN_TRADE_QTY ({MIN_TRADE_QTY}). Skipping trade.")
                    return

                place_order(symbol, qty_to_sell_and_short, 'sell',
                            f"Close long (and flip short) @ ~${latest_price:.2f}")
            elif current_position == 0:  # No position, open a new short
                estimated_cost_to_short = qty_to_trade * latest_price
                if estimated_cost_to_short > available_buying_power * 0.5:
                    logging.warning(
                        f"{symbol}: Potential margin issue for new SHORT order. "
                        f"Notional value: ${estimated_cost_to_short:.2f}. "
                        f"Available Buying Power: ${available_buying_power:.2f}. Skipping trade."
                    )
                    return

                if qty_to_trade < MIN_TRADE_QTY:
                    logging.warning(f"{symbol}: Calculated SELL quantity {qty_to_trade} is below MIN_TRADE_QTY ({MIN_TRADE_QTY}). Skipping trade.")
                    return

                place_order(symbol, qty_to_trade, 'sell',
                            f"New short @ ~${latest_price:.2f} (signal: {signal_to_act_on})")
            else:  # current_position < 0 (already short)
                logging.info(f"{symbol}: Already short {abs(current_position)} shares on SELL signal. Holding.")
        elif signal_to_act_on == "HOLD":
            logging.info(f"{symbol}: Hold signal. No action taken.")

    except APIError as api_e:
        error_message = api_e.args[0]
        if "insufficient qty available" in error_message.lower() or "not enough buying_power" in error_message.lower():
            logging.error(f"{symbol}: Trade ERROR (Alpaca API - Insufficient Funds/Shares): {error_message}.")
        else:
            logging.error(f"{symbol}: Trade ERROR (Alpaca API): {error_message}")
    except Exception as e:
        logging.error(f"{symbol}: An unexpected error occurred during trade execution: {e}")


def run_strategy_for_symbol(symbol, account_state, positions_cache, daily_bars, hourly_bars,
                            minute_bars=None, risk_multiplier=1.0, market_regime='bull'):
    """Runs every strategy in STRATEGIES against the appropriate timeframe,
    arbitrates between their signals, and executes the chosen trade."""
    logging.info(f"Checking for signals for {symbol}...")

    if minute_bars is None:
        minute_bars = pd.DataFrame()
    bars_by_timeframe = {"daily": daily_bars, "hourly": hourly_bars, "minute": minute_bars}

    # Prefer hourly price for "current" price  it's closer to real-time.
    latest_price = None
    if not hourly_bars.empty:
        latest_price = hourly_bars['close'].iloc[-1]
    elif not daily_bars.empty:
        latest_price = daily_bars['close'].iloc[-1]

    if latest_price is None:
        logging.warning(f"{symbol}: No valid latest price available. Skipping.")
        return

    # Compute ATR from daily bars for use by both position sizing (lever 1) and
    # trailing stops (lever 2). Always computed generically here so it's available
    # regardless of which strategies are enabled in STRATEGIES.
    latest_atr = None
    if not daily_bars.empty and len(daily_bars) > ATR_PERIOD:
        atr_series = ta.volatility.average_true_range(
            daily_bars['high'], daily_bars['low'], daily_bars['close'], window=ATR_PERIOD
        ).dropna()
        if not atr_series.empty:
            latest_atr = atr_series.iloc[-1]

    # --- Run every configured strategy on its designated timeframe ---
    signals = []
    for strategy, timeframe in STRATEGIES:
        bars = bars_by_timeframe[timeframe]
        if bars.empty:
            continue

        signal = strategy.generate_signal(symbol, bars)
        signals.append(signal)

        if signal.action != HOLD or signal.details.get("reason") is None:
            logging.info(f"{symbol}: [{strategy.name}/{timeframe}] -> {signal.action} | {signal.details}")

        # The 200-SMA trend filter only makes sense for the trend-following
        # SMA crossover strategy: don't go long below the long-term trend, or
        # short above it. This used to be hard-coded for "primary_signal";
        # now it's applied generically to whichever signal carries a trend_sma.
        trend_sma = signal.details.get("trend_sma")
        if trend_sma is not None:
            if signal.action == BUY and latest_price < trend_sma:
                logging.info(
                    f" {symbol}: {strategy.name} BUY signal ignored. "
                    f"Price (${latest_price:.2f}) is below 200 SMA Trend (${trend_sma:.2f})."
                )
                signal.action = HOLD
            elif signal.action == SELL and latest_price > trend_sma:
                logging.info(
                    f" {symbol}: {strategy.name} SELL (short) signal ignored. "
                    f"Price (${latest_price:.2f}) is above 200 SMA Trend (${trend_sma:.2f})."
                )
                signal.action = HOLD

        # --- Lever 4: Market regime filter ---
        # Mean-reversion strategies are suppressed when SPY is below its 200-SMA.
        # We only block NEW longs (BUY) — exits (SELL) are still allowed so we
        # can close existing positions cleanly even in a bear market.
        if (market_regime == 'bear'
                and strategy.strategy_type == 'mean_reversion'
                and signal.action == BUY):
            logging.info(
                f" {symbol}: {strategy.name} BUY suppressed — "
                f"bear regime (SPY below {REGIME_SMA_PERIOD}-SMA)."
            )
            signal.action = HOLD

    # --- Portfolio Position Cap ---
    # If we don't already hold this symbol and we're at the simultaneous-position
    # limit, block new entries (mirrors backtester.py so live behavior matches
    # what we backtested -- this cap is what keeps drawdown in the tested range).
    position_obj = positions_cache.get(symbol, None)
    current_position = int(position_obj.qty) if position_obj is not None else 0

    if current_position == 0 and len(positions_cache) >= MAX_SIMULTANEOUS_POSITIONS:
        logging.info(
            f"{symbol}: At max simultaneous positions ({MAX_SIMULTANEOUS_POSITIONS}). "
            f"Blocking new entries, holding."
        )
        for signal in signals:
            signal.action = HOLD

    # --- Quant Principle: Signal Arbitration ---
    chosen_signal = choose_signal_to_act_on(signals, current_position)
    signal_to_act_on = chosen_signal.action if chosen_signal else HOLD
    if chosen_signal:
        logging.info(f"{symbol}: Arbitration chose {chosen_signal.strategy_name} -> {signal_to_act_on}")
    else:
        logging.info(f"{symbol}: No actionable signal from any strategy. Holding.")

    execute_trade_logic(
        symbol,
        signal_to_act_on,
        current_position,
        latest_price,
        account_state,
        position_obj,
        atr_val=latest_atr,
        risk_multiplier=risk_multiplier,
    )


# --- Main Autonomous Bot Loop ---
if __name__ == "__main__":
    logging.info("Starting autonomous trading bot in Stable Mode...")
    send_alert("Trading bot started.")
    peak_equity = 0.0  # initialised to 0 so first real equity reading sets the peak
    # Previous-cycle state, so throttle/regime alerts fire only on CHANGE
    # (alerting every 5-minute cycle while a state persists would be spam).
    prev_risk_multiplier = 1.0
    prev_regime = None
    while True:
        try:
            logging.info("=========================================")
            logging.info("STARTING NEW EXECUTION CYCLE...")

            # Fetch Global State once per cycle
            account_state = api.get_account()
            raw_positions = api.list_positions()
            positions_cache = {pos.symbol: pos for pos in raw_positions}

            current_equity = float(account_state.equity)
            logging.info(f"Equity: ${current_equity:,.2f}")

            # --- Lever 3: portfolio drawdown throttle ---
            # Track the running all-time-high equity so we can detect drawdowns.
            # peak_equity persists across loop iterations (initialised once above).
            peak_equity = max(peak_equity, current_equity)
            risk_multiplier = compute_risk_multiplier(current_equity, peak_equity)
            if risk_multiplier < 1.0:
                logging.warning(
                    f"DRAWDOWN THROTTLE ACTIVE: equity ${current_equity:,.2f} is "
                    f"{(current_equity/peak_equity - 1)*100:.1f}% below peak "
                    f"${peak_equity:,.2f}. New position sizes scaled to {risk_multiplier:.0%}."
                )
            if risk_multiplier != prev_risk_multiplier:
                send_alert(
                    f"RISK: drawdown throttle changed {prev_risk_multiplier:.0%} -> "
                    f"{risk_multiplier:.0%} (equity ${current_equity:,.0f}, "
                    f"{(current_equity/peak_equity - 1)*100:.1f}% below peak)"
                )
                prev_risk_multiplier = risk_multiplier

            # --- Batch-fetch bars for ALL symbols in just 2 API calls ---
            now_et = pd.Timestamp.now(tz='America/New_York')
            daily_start = (now_et.date() - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)).isoformat()
            hourly_start = (now_et.date() - pd.Timedelta(days=SCALPING_LOOKBACK_DAYS)).isoformat()
            today_str = now_et.date().isoformat()

            minute_start = (now_et.date() - pd.Timedelta(days=MINUTE_LOOKBACK_DAYS)).isoformat()

            all_daily_bars = get_latest_bars_batch(SYMBOLS, daily_start, today_str, TIME_FRAME)
            all_hourly_bars = get_latest_bars_batch(SYMBOLS, hourly_start, today_str, SCALPING_TIME_FRAME)
            all_minute_bars = get_latest_bars_batch(SYMBOLS, minute_start, today_str, MINUTE_TIME_FRAME)

            # Compute market regime once per cycle from SPY's already-fetched bars.
            # SPY is in SYMBOLS so its bars come back in the same batch call above —
            # no extra API call needed.
            spy_bars = all_daily_bars.get(REGIME_SYMBOL, pd.DataFrame())
            market_regime = compute_market_regime(spy_bars)
            if market_regime == 'bear':
                logging.warning(
                    f"BEAR REGIME ACTIVE: SPY is below its {REGIME_SMA_PERIOD}-day SMA. "
                    f"Mean-reversion BUY signals (Bollinger, RSI) will be suppressed this cycle."
                )
            if market_regime != prev_regime:
                if prev_regime is not None:  # don't alert on the first cycle's "None -> bull"
                    send_alert(f"REGIME: market regime changed {prev_regime.upper()} -> {market_regime.upper()} "
                               f"(SPY vs {REGIME_SMA_PERIOD}-day SMA)")
                prev_regime = market_regime

            for symbol in SYMBOLS:
                try:
                    daily_bars = all_daily_bars.get(symbol, pd.DataFrame())
                    hourly_bars = all_hourly_bars.get(symbol, pd.DataFrame())
                    minute_bars = all_minute_bars.get(symbol, pd.DataFrame())
                    run_strategy_for_symbol(
                        symbol, account_state, positions_cache,
                        daily_bars, hourly_bars, minute_bars,
                        risk_multiplier=risk_multiplier,
                        market_regime=market_regime,
                    )
                except Exception as e:
                    logging.error(f"{symbol}: Loop Error - {e}")

        except Exception as global_err:
            logging.error(f"Global Error: {global_err}")
            send_alert(f"ERROR: bot main loop hit an exception (still running): {global_err}")

        logging.info(f"Cycle Complete. Sleeping {CHECK_INTERVAL_SECONDS}s...")
        time.sleep(CHECK_INTERVAL_SECONDS)


