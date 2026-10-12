import os
import time
import numpy as np
import pandas as pd
import ta
import logging
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError
from strategies import (
    BUY, SELL, HOLD,
    SmaCrossoverStrategy,
    MacdMomentumStrategy,
    RsiMomentumStrategy,
    BollingerBandReversionStrategy,
    RocMomentumStrategy,
    AtrChannelBreakoutStrategy,
    choose_signal_to_act_on,
)

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

if ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL:
    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    logging.info("Connected to Alpaca API for backtester.")
else:
    logging.error("Alpaca API keys not found. Please check your .env file. Exiting backtester.")
    exit()

# --- 2. STRATEGY PARAMETERS (Mirroring trading_bot.py) ---
# Expanded beyond tech/growth (AAPL...PLTR were all correlated -- they tend to
# crash together) by adding financials (JPM, BAC, V), healthcare (JNJ, UNH),
# energy (XOM), staples (PG, KO, WMT), and industrials (CAT, HD, DIS).
# Genuine sector diversification is one of the most reliable ways to reduce
# portfolio-level drawdown: when tech sells off, staples/healthcare often don't.
SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ", "AMD", "NFLX", "PLTR",
    "JPM", "BAC", "V", "JNJ", "UNH", "XOM", "PG", "KO", "WMT", "CAT", "HD", "DIS",
    # Round 2 additions (2026-06-08): filling out sector gaps for diversification --
    # semiconductors, industrials/aerospace, energy, healthcare/pharma, communications,
    # financials, and consumer staples that weren't yet represented.
    "INTC", "MU", "AVGO",          # semiconductors (complement NVDA/AMD)
    "BA", "GE", "LMT",             # industrials / aerospace & defense
    "CVX", "COP",                  # energy (complement XOM)
    "PFE", "MRK", "ABBV",          # healthcare / pharma (complement JNJ/UNH)
    "T", "VZ",                     # telecom / communications
]

# --- Trading Parameters ---
FAST_SMA_PERIOD = 20
SLOW_SMA_PERIOD = 50
TREND_FILTER_PERIOD = 200
DAILY_LOOKBACK_DAYS = 300
TIME_FRAME = TimeFrame.Day

# Momentum Scalping Strategy Parameters
SCALPING_TIME_FRAME = TimeFrame.Hour
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65                 # Lowered from 70 to exit shorts quicker
RSI_OVERSOLD = 45                   # Raised from 30 to buy strong pullbacks more frequently
SCALPING_SMA_PERIOD = 50            
SCALPING_TAKE_PROFIT_PCT = 0.035     # Increased to 3.5% for bigger wins
SCALPING_STOP_LOSS_PCT = 0.045       # Increased to 4.5% to survive shakeouts
SCALPING_LOOKBACK_DAYS = 60

# --- Backtesting Specific Parameters ---
BACKTEST_START_DATE = "2025-01-01"
BACKTEST_END_DATE = "2026-01-01"
BACKTEST_PORTFOLIO_ALLOCATION_PCT = 0.03
BACKTEST_TAKE_PROFIT_PCT = 0.12     # Let profits run up to 12% on daily crossovers
BACKTEST_TRAILING_STOP_PCT = 0.05   # Give daily trends a wider 5% trailing cushion
BACKTEST_MIN_TRADE_QTY = 1

# --- Realistic transaction cost model ---
# Clean backtests that ignore costs almost always outperform live trading.
# Two sources of friction every real trade incurs:
#
# 1. Commission: the fee your broker charges per order leg.
#    Alpaca paper & live are technically $0, but we model a conservative
#    $0.50/trade to stress-test the strategy -- if it can't survive $0.50
#    it isn't a robust edge. Real-world costs (exchange fees, regulatory
#    fees) are typically $0.003-0.01/share, roughly $0.20-1.00/trade for
#    the position sizes we're using.
#
# 2. Slippage: the difference between the price you see on the chart and the
#    price you actually get filled at. Market orders move against you slightly
#    because you're crossing the bid-ask spread. For liquid large-caps (SPY,
#    AAPL etc.) the spread is ~$0.01-0.05, so 0.05% of price is a reasonable
#    conservative estimate. This is applied ADVERSELY at execution:
#       buy orders fill at  price * (1 + SLIPPAGE_PCT)  -- slightly worse
#       sell orders fill at price * (1 - SLIPPAGE_PCT)  -- slightly worse
#    Together these two frictions compound across hundreds of trades and
#    will visibly drag down the gross P/L numbers.
COMMISSION_PER_TRADE = 0.50     # $ per order leg (entry AND exit each cost this)
SLIPPAGE_PCT         = 0.0005   # 0.05% adverse price movement on execution

# Backtested: cap=3 -> +2.24% P/L / -16.92% drawdown; cap=8 -> +5.95% / -35.68%.
# Raising the cap roughly doubles both return AND drawdown -- a real
# risk/reward dial, not a free lunch. Going with 8 for more upside; revisit
# if live drawdown becomes uncomfortable (try something between 3 and 8).
MAX_SIMULTANEOUS_POSITIONS = 8
ATR_PERIOD = 14                     # Average True Range period for volatility-adjusted stops

# --- Risk-management levers (backtested 2026-06-09, winning combo: Levers 1+2) ---
# AtrSize+AtrStop -> +75.6% return, -11.4% max drawdown on 2025 data.
# Lever 3 (throttle) had no effect with -15%/-25% triggers (max DD was ~12%);
# recalibrated to -5%/-10% in live bot but kept at those levels here for
# historical reproducibility. Toggle any lever False to test degraded configs.
USE_ATR_POSITION_SIZING = True
RISK_PER_TRADE_PCT = 0.01

# Notional cap + dust floor (mirrors trading_bot.py, added 2026-06-10):
# ATR sizing caps loss-to-STOP, not loss-to-GAP -- a calm stock gets a huge
# share count (UNH hit 52% of the live book). Cap notional per position, and
# skip positions too small to matter (they waste a position slot + commission).
MAX_POSITION_PCT = 0.15
MIN_POSITION_NOTIONAL = 1000.0           # risk ~1% of equity per trade if the stop is hit
ATR_STOP_MULTIPLE = 2.5             # stop distance = ATR * this (must match live bot)

USE_ATR_STOPS = True

USE_DRAWDOWN_THROTTLE = True
DRAWDOWN_THROTTLE_LEVELS = [
    (-0.10, 0.0),   # 10%+ below peak: stop opening new positions
    (-0.05, 0.5),   # 5%+ below peak: halve new position sizes
]

# Lever 4: Market regime filter (mirrors trading_bot.py)
USE_REGIME_FILTER = True
REGIME_SYMBOL     = "SPY"
REGIME_SMA_PERIOD = 200

# Benchmark for relative-performance reporting. Same ticker as the regime
# filter, but a separate constant: the two roles are independent (you might
# someday regime-filter on VIX but still benchmark against SPY).
BENCHMARK_SYMBOL = "SPY"

# --- Strategy Framework (mirrors trading_bot.py so backtest results predict live behavior) ---
# NOTE (2026-06-09): order now matches trading_bot.py exactly. Arbitration
# acts on the FIRST non-HOLD signal when flat, so priority order changes
# which strategy wins when several fire at once -- the backtester and bot
# previously disagreed (Bollinger first here, MACD first live), meaning the
# backtest wasn't faithfully simulating the live bot's choices.
STRATEGIES = [
    # EXPERIMENT 4: adding MACD momentum on daily bars. It looks at a
    # different signal entirely (rate-of-change of trend, via EMA
    # crossovers) than our existing mean-reversion (Bollinger/RSI)
    # strategies, so it should fire at different times -- testing whether
    # it adds genuinely new profitable trades or just dilutes the cap.
    (MacdMomentumStrategy(), "daily"),
    # EXPERIMENT 5 (2026-06-09): top-2 performers from the 20-strategy
    # benchmark, both daily trend-followers. Solo on 2025 they beat the
    # baseline (+15.06% and +16.84% vs +6.36%) -- this tests whether they
    # ADD to the ensemble or just compete with MACD for the same trades.
    # WALK-FORWARD VERDICT (2026-06-10, bias-free engine, 2023/24/25):
    #   ATR Channel: -$4,864 / +$6,108 / +$946 = +$2,190 net, 2/3 years up. KEPT.
    #   ROC:           +$501 / -$2,734 / -$2,905 = -$5,138 net, 1/3 years up. CUT.
    # (RocMomentumStrategy(), "daily"),
    (AtrChannelBreakoutStrategy(), "daily"),
    # EXPERIMENT 3 (reverted): widening to 2.5 std devs flipped Bollinger from
    # +$2,095 to -$1,890 (win rate 56% -> 41%)! Counter-intuitive but makes
    # sense: a 2.5-sigma move is rare enough that it's usually driven by real
    # news/trend shifts (which continue) rather than noise (which reverts) --
    # exactly the "catching a falling knife" failure mode of mean reversion.
    # The tighter 2.0 band, paradoxically, is the better mean-reversion filter
    # here because it catches routine noise-driven oscillations that actually
    # do revert. Keeping the original.
    (BollingerBandReversionStrategy(window=20, num_std=2.0), "hourly"),
    # EXPERIMENT 2 (reverted): loosening RSI to 50/55 fired 55 trades instead
    # of 7, but it crowded out Bollinger's better trades via the
    # MAX_SIMULTANEOUS_POSITIONS=3 cap, making overall P/L *worse*
    # ($2,215 vs $2,406). More trades from a weaker strategy can starve a
    # stronger one of position "slots"  keeping the original thresholds.
    (RsiMomentumStrategy(
        rsi_period=RSI_PERIOD,
        oversold=RSI_OVERSOLD,
        overbought=RSI_OVERBOUGHT,
        trend_sma_period=SCALPING_SMA_PERIOD,
    ), "hourly"),
    # --- EXPERIMENT 1: SMA Crossover temporarily disabled ---
    # It fired once in a year and lost money; testing whether removing it
    # changes overall performance meaningfully (if not, it was dead weight).
    # (SmaCrossoverStrategy(
    #     fast_period=FAST_SMA_PERIOD,
    #     slow_period=SLOW_SMA_PERIOD,
    #     trend_filter_period=TREND_FILTER_PERIOD,
    # ), "daily"),
]


# --- 4. DATA FETCHING HELPER ---
def get_historical_bars(symbol, start_date_str, end_date_str, time_frame, feed='iex'):
    """Fetches historical bars for backtesting."""
    try:
        start_ts = pd.Timestamp(start_date_str, tz='America/New_York')
        end_ts = pd.Timestamp(end_date_str, tz='America/New_York')

        bars = api.get_bars(
            symbol,
            time_frame,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            feed=feed,
            adjustment='split'
        ).df

        if bars.empty:                                                                                                                 
            logging.warning(f"{symbol}: No historical bars fetched for {time_frame} from {start_date_str} to {end_date_str} using feed='{feed}'.")
            return pd.DataFrame()

        bars = bars.sort_index()
        bars = bars[bars['close'] > 0] # Filter out zero-price anomalies

        return bars
    except Exception as e:
        logging.error(f"{symbol}: Error fetching historical bars for backtesting: {e}")
        return pd.DataFrame()

# --- 5. POSITION SIZING HELPER ---
def calculate_dynamic_qty(symbol, latest_price, equity, buying_power, atr_val=None, risk_multiplier=1.0):
    """Calculates how many shares to buy/short for a new position.

    Two sizing modes, selected by USE_ATR_POSITION_SIZING:

    1. Fixed-allocation (the original approach): spend a constant % of total
       equity on every position, e.g. always put ~3% of the book into AAPL
       and ~3% into a much-more-volatile name like PLTR. The dollar amount
       invested is the same either way -- but the *risk* isn't, because PLTR
       can swing 3x as much in a day. This is why a fixed-count position cap
       doesn't really control risk: it controls how many "3% slots" are
       filled, not how much any of them can actually hurt you.

    2. ATR-based ("volatility parity" / "risk parity") sizing: instead of
       asking "how many dollars should I put in?", ask "how many SHARES can
       I hold such that, if my stop-loss is hit, I lose about the same dollar
       amount no matter which stock this is?"

           dollar_risk_per_trade = equity * RISK_PER_TRADE_PCT      # e.g. 1% of equity
           stop_distance_per_share = atr_val * ATR_STOP_MULTIPLE    # how far price must move to stop us out
           shares = dollar_risk_per_trade / stop_distance_per_share

       A volatile stock has a bigger ATR -> bigger stop_distance -> fewer
       shares bought. A calm stock has a smaller ATR -> more shares. Both
       end up risking ~the same dollars if the stop is hit. This is the
       textbook fix for "one wild position blowing up the whole portfolio"
       -- a failure mode a simple position-count cap literally cannot see,
       because it only counts positions, not their risk.

    `risk_multiplier` is how the portfolio-level drawdown throttle (lever 3)
    talks to this function: when we're in a deep drawdown, the throttle
    passes in e.g. 0.5 or 0.0 to shrink (or block) new positions, on top of
    whatever sizing mode is active.
    """
    try:
        if risk_multiplier <= 0:
            return 0

        if USE_ATR_POSITION_SIZING and atr_val is not None and atr_val > 0:
            dollar_risk = equity * RISK_PER_TRADE_PCT * risk_multiplier
            stop_distance = atr_val * ATR_STOP_MULTIPLE
            calculated_qty = int(dollar_risk // stop_distance)
        else:
            target_cash_allocation = equity * BACKTEST_PORTFOLIO_ALLOCATION_PCT * risk_multiplier
            cash_to_use = min(target_cash_allocation, buying_power)
            calculated_qty = int(cash_to_use // latest_price)

        # Cap notional exposure: ATR sizing limits loss-to-stop, but a huge
        # share count in a calm stock is still huge overnight-gap risk.
        if latest_price > 0:
            max_notional_qty = int((equity * MAX_POSITION_PCT) // latest_price)
            calculated_qty = min(calculated_qty, max_notional_qty)

        # Never let sizing exceed what we can actually afford.
        affordable_qty = int(buying_power // latest_price) if latest_price > 0 else 0
        calculated_qty = min(calculated_qty, affordable_qty)

        # Dust filter: skip positions too small to matter rather than forcing
        # a 1-share trade that occupies a position slot and pays commissions.
        if calculated_qty * latest_price < MIN_POSITION_NOTIONAL:
            return 0

        return calculated_qty
    except Exception as e:
        # Fail SAFE: a sizing error means "don't trade", never "trade 1 share".
        logging.error(f"{symbol}: Error calculating dynamic quantity: {e}. Skipping trade.")
        return 0


# --- Lever 3 helper: portfolio-level drawdown throttle ---
def compute_risk_multiplier(current_equity, peak_equity):
    """Returns a 0.0-1.0 multiplier that scales down new-position risk as the
    account sinks further below its all-time-high equity ("peak").

    Concept: a position-count cap is "static" -- it applies the same limit
    whether you're up 20% or down 30%. A drawdown throttle is *adaptive*: it
    only tightens up when you're actually in trouble, and otherwise stays out
    of the way. This is the lever that targets max-drawdown most directly,
    because it interrupts the "losing streak compounds into a bigger losing
    streak" feedback loop -- the actual mechanism behind most catastrophic
    drawdowns (you keep betting full-size while your edge has gone cold).
    """
    if not USE_DRAWDOWN_THROTTLE or peak_equity <= 0:
        return 1.0

    drawdown = (current_equity - peak_equity) / peak_equity

    # DRAWDOWN_THROTTLE_LEVELS is ordered worst-first, so the first matching
    # (most severe) threshold wins.
    for trigger, multiplier in DRAWDOWN_THROTTLE_LEVELS:
        if drawdown <= trigger:
            return multiplier

    return 1.0

# --- 6. SIGNAL CALCULATION FUNCTIONS ---
def calculate_sma_crossover_signals(symbol, bars, fast_period, slow_period):
    """Calculates Fast/Slow/Trend SMAs and generates buy/sell crossover signals."""
    min_bars_needed = TREND_FILTER_PERIOD + 2
    if bars.empty or len(bars) < min_bars_needed:
        return "NO_DATA", None, None, None

    bars = bars.sort_index()


    bars['Fast_SMA'] = ta.trend.sma_indicator(bars['close'], window=fast_period)
    bars['Slow_SMA'] = ta.trend.sma_indicator(bars['close'], window=slow_period)
    bars['Trend_SMA'] = ta.trend.sma_indicator(bars['close'], window=TREND_FILTER_PERIOD)

    bars['Fast_SMA'] = ta.trend.sma_indicator(bars['close'], window=fast_period)
    bars['Slow_SMA'] = ta.trend.sma_indicator(bars['close'], window=slow_period)
    bars['Trend_SMA'] = ta.trend.sma_indicator(bars['close'], window=TREND_FILTER_PERIOD)

    # Quant Volatility Calculation: Average True Range (ATR)
    bars['ATR'] = ta.volatility.average_true_range(bars['high'], bars['low'], bars['close'], window=ATR_PERIOD)

    bars.dropna(subset=['Fast_SMA', 'Slow_SMA', 'Trend_SMA', 'ATR'], inplace=True)

    if bars.empty or len(bars) < 2:
        return "NO_DATA", None, None, None

    latest_fast_sma = bars['Fast_SMA'].iloc[-1]
    latest_slow_sma = bars['Slow_SMA'].iloc[-1]
    prev_fast_sma = bars['Fast_SMA'].iloc[-2]
    prev_slow_sma = bars['Slow_SMA'].iloc[-2]
    latest_trend_sma = bars['Trend_SMA'].iloc[-1]

    signal = "HOLD"
    if prev_fast_sma < prev_slow_sma and latest_fast_sma >= latest_slow_sma:
        signal = "BUY"
    elif prev_fast_sma > prev_slow_sma and latest_fast_sma <= latest_slow_sma:
        signal = "SELL"

    latest_atr = bars['ATR'].iloc[-1]
    return signal, latest_fast_sma, latest_slow_sma, latest_trend_sma, latest_atr

def calculate_momentum_scalping_signals(symbol, bars):
    """Calculates RSI-based buy/sell signals on Hourly timeframe."""
    min_bars_needed = RSI_PERIOD + 1
    if bars.empty or len(bars) < min_bars_needed:
        return "NO_DATA", None, None

    bars = bars.sort_index()                                                                                                           

    # Calculate RSI
    bars['RSI'] = ta.momentum.rsi(bars['close'], window=RSI_PERIOD)
    bars.dropna(subset=['RSI'], inplace=True)

    if bars.empty:
        return "NO_DATA", None, None

    latest_rsi = bars['RSI'].iloc[-1]

    signal = "HOLD"
    # Buy when Hourly RSI is oversold (under 45) for a quick pullback entry
    if latest_rsi <= RSI_OVERSOLD:
        signal = "BUY"
    # Sell when Hourly RSI is overbought (above 65) to take quick profits
    elif latest_rsi >= RSI_OVERBOUGHT:
        signal = "SELL"

    return signal, latest_rsi, None


# --- 7. BACKTEST TRADE SIMULATION ENGINE ---
def _exec_price(price, side):
    """Returns price adjusted for slippage.
    Buys fill slightly ABOVE the last close; sells fill slightly BELOW.
    This models crossing the bid-ask spread on a market order -- the most
    common source of small, consistent drag that backtests ignore but live
    trading always feels.
    """
    if side == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def execute_trade_logic_backtest(
    symbol,
    signal_to_act_on,
    strategy_name,
    current_bar_price,
    current_timestamp,
    current_cash,
    open_positions,
    total_portfolio_equity,
    atr_val,
    risk_multiplier=1.0
):
    """Simulates trade execution historically inside the backtest engine."""
    trade_details = None

    position = open_positions.get(symbol, None)
    current_qty = position['qty'] if position else 0
    available_buying_power = current_cash

    # --- TRAILING STOP-LOSS & TAKE-PROFIT CHECK ---
    if current_qty != 0:
        avg_entry_price = position['entry_price']
        high_water_mark = position['high_water_mark']
        low_water_mark = position['low_water_mark']

        if current_qty > 0:  # LONG POSITION
            if current_bar_price > high_water_mark:
                position['high_water_mark'] = current_bar_price

            # Take Profit
            tp_price = avg_entry_price * (1 + BACKTEST_TAKE_PROFIT_PCT)
            if current_bar_price >= tp_price:
                ep = _exec_price(current_bar_price, 'sell')
                profit_loss = (ep - avg_entry_price) * current_qty - COMMISSION_PER_TRADE
                current_cash += ep * current_qty - COMMISSION_PER_TRADE
                trade_details = {
                    'timestamp': current_timestamp, 'symbol': symbol, 'side': 'SELL',
                    'qty': current_qty, 'price': ep, 'P/L': profit_loss,
                    'reason': 'TAKE_PROFIT_LONG', 'position_type': 'long', 'strategy': position.get('strategy', 'UNKNOWN')
                }
                del open_positions[symbol]
                return current_cash, open_positions, trade_details

            # Trailing Stop -- lever 2: ATR-based stop (volatility-adjusted
            # distance) vs. a flat percentage. ATR stops give a calm stock a
            # tight leash and a volatile stock more room to breathe, which
            # should mean fewer "stopped out by routine noise" exits.
            if USE_ATR_STOPS and atr_val is not None:
                stop_offset = atr_val * ATR_STOP_MULTIPLE
            else:
                stop_offset = position['high_water_mark'] * BACKTEST_TRAILING_STOP_PCT
            trail_stop_price = position['high_water_mark'] - stop_offset
            if current_bar_price <= trail_stop_price:
                ep = _exec_price(current_bar_price, 'sell')
                profit_loss = (ep - avg_entry_price) * current_qty - COMMISSION_PER_TRADE
                current_cash += ep * current_qty - COMMISSION_PER_TRADE
                trade_details = {
                    'timestamp': current_timestamp, 'symbol': symbol, 'side': 'SELL',
                    'qty': current_qty, 'price': ep, 'P/L': profit_loss,
                    'reason': 'TRAILING_STOP_LONG', 'position_type': 'long', 'strategy': position.get('strategy', 'UNKNOWN')
                }
                del open_positions[symbol]
                return current_cash, open_positions, trade_details

        elif current_qty < 0:  # SHORT POSITION
            if current_bar_price < low_water_mark:
                position['low_water_mark'] = current_bar_price

            # Take Profit
            tp_price = avg_entry_price * (1 - BACKTEST_TAKE_PROFIT_PCT)
            if current_bar_price <= tp_price:
                ep = _exec_price(current_bar_price, 'buy')
                profit_loss = (avg_entry_price - ep) * abs(current_qty) - COMMISSION_PER_TRADE
                current_cash -= ep * abs(current_qty) + COMMISSION_PER_TRADE
                trade_details = {
                    'timestamp': current_timestamp, 'symbol': symbol, 'side': 'BUY',
                    'qty': abs(current_qty), 'price': ep, 'P/L': profit_loss,
                    'reason': 'TAKE_PROFIT_SHORT', 'position_type': 'short', 'strategy': position.get('strategy', 'UNKNOWN')
                }
                del open_positions[symbol]
                return current_cash, open_positions, trade_details

            # Trailing Stop -- mirror of the long-side logic above.
            if USE_ATR_STOPS and atr_val is not None:
                stop_offset = atr_val * ATR_STOP_MULTIPLE
            else:
                stop_offset = position['low_water_mark'] * BACKTEST_TRAILING_STOP_PCT
            trail_stop_price = position['low_water_mark'] + stop_offset
            if current_bar_price >= trail_stop_price:
                ep = _exec_price(current_bar_price, 'buy')
                profit_loss = (avg_entry_price - ep) * abs(current_qty) - COMMISSION_PER_TRADE
                current_cash -= ep * abs(current_qty) + COMMISSION_PER_TRADE
                trade_details = {
                    'timestamp': current_timestamp, 'symbol': symbol, 'side': 'BUY',
                    'qty': abs(current_qty), 'price': ep, 'P/L': profit_loss,
                    'reason': 'TRAILING_STOP_SHORT', 'position_type': 'short', 'strategy': position.get('strategy', 'UNKNOWN')
                }
                del open_positions[symbol]
                return current_cash, open_positions, trade_details

    qty_to_trade = calculate_dynamic_qty(
        symbol, current_bar_price, total_portfolio_equity, available_buying_power,
        atr_val=atr_val, risk_multiplier=risk_multiplier
    )

    if qty_to_trade == 0:
        return current_cash, open_positions, trade_details

    if signal_to_act_on == "BUY" and current_qty == 0:
        ep = _exec_price(current_bar_price, 'buy')
        total_cost = ep * qty_to_trade + COMMISSION_PER_TRADE
        if current_cash >= total_cost:
            current_cash -= total_cost
            open_positions[symbol] = {
                'qty': qty_to_trade, 'entry_price': ep,
                'high_water_mark': ep, 'low_water_mark': ep,
                'strategy': strategy_name
            }
            trade_details = {
                'timestamp': current_timestamp, 'symbol': symbol, 'side': 'BUY',
                'qty': qty_to_trade, 'price': ep, 'P/L': 0.0, 'reason': signal_to_act_on,
                'position_type': 'long', 'strategy': strategy_name
            }

    elif signal_to_act_on == "SELL" and current_qty == 0:
        ep = _exec_price(current_bar_price, 'sell')
        current_cash += ep * qty_to_trade - COMMISSION_PER_TRADE
        open_positions[symbol] = {
            'qty': -qty_to_trade, 'entry_price': ep,
            'high_water_mark': ep, 'low_water_mark': ep,
            'strategy': strategy_name
        }
        trade_details = {
            'timestamp': current_timestamp, 'symbol': symbol, 'side': 'SELL',
            'qty': qty_to_trade, 'price': ep, 'P/L': 0.0, 'reason': signal_to_act_on,
            'position_type': 'short', 'strategy': strategy_name
        }

    elif signal_to_act_on == "BUY" and current_qty < 0: # Cover Short
        ep = _exec_price(current_bar_price, 'buy')
        profit_loss = (position['entry_price'] - ep) * abs(current_qty) - COMMISSION_PER_TRADE
        current_cash -= ep * abs(current_qty) + COMMISSION_PER_TRADE
        trade_details = {
            'timestamp': current_timestamp, 'symbol': symbol, 'side': 'BUY',
            'qty': abs(current_qty), 'price': ep, 'P/L': profit_loss,
            'reason': 'COVER_SHORT_SIGNAL', 'position_type': 'short', 'strategy': position.get('strategy', 'UNKNOWN')
        }
        del open_positions[symbol]

    elif signal_to_act_on == "SELL" and current_qty > 0: # Close Long
        ep = _exec_price(current_bar_price, 'sell')
        profit_loss = (ep - position['entry_price']) * current_qty - COMMISSION_PER_TRADE
        current_cash += ep * current_qty - COMMISSION_PER_TRADE
        trade_details = {
            'timestamp': current_timestamp, 'symbol': symbol, 'side': 'SELL',
            'qty': current_qty, 'price': ep, 'P/L': profit_loss,
            'reason': 'CLOSE_LONG_SIGNAL', 'position_type': 'long', 'strategy': position.get('strategy', 'UNKNOWN')
        }
        del open_positions[symbol]

    return current_cash, open_positions, trade_details

# --- 8. SIMULATION ENGINE (callable from walk_forward.py or __main__) ---

def run_simulation(start_date, end_date, all_daily_bars, all_hourly_bars,
                   initial_cash=100_000.0, label=""):
    """
    Run one full simulation on pre-fetched bar data.

    Parameters
    ----------
    start_date / end_date : str  — used only for log labelling
    all_daily_bars / all_hourly_bars : dict[symbol -> pd.DataFrame]
    initial_cash : float
    label : str  — prefix for log lines (e.g. "2023" for walk-forward)

    Returns
    -------
    dict with keys: final_equity, total_pl, pl_pct, max_drawdown, sharpe,
                    num_trades, trade_history, portfolio_history
    """
    prefix = f"[{label}] " if label else ""
    logging.info(f"{prefix}Starting simulation {start_date} -> {end_date}, {len(all_daily_bars)} symbols")

    # --- Initialize Backtest Capital & Tracker Ledgers ---
    current_cash = initial_cash
    open_positions = {}
    trade_history = []
    portfolio_history = []
    peak_equity = initial_cash  # running all-time-high equity, for the drawdown throttle (lever 3)

    # All symbols actually present in the provided data (may be a subset of
    # the global SYMBOLS list when called from walk_forward.py).
    active_symbols = sorted(set(list(all_daily_bars.keys()) + list(all_hourly_bars.keys())))

    # Compile the timeline of dates
    all_timestamps = set()
    for symbol in active_symbols:
        if symbol in all_daily_bars:
            all_timestamps.update(all_daily_bars[symbol].index.tolist())
        if symbol in all_hourly_bars:
            all_timestamps.update(all_hourly_bars[symbol].index.tolist())

    sorted_timestamps = sorted(list(all_timestamps))

    # --- Lever 4: Precompute SPY market regime series ---
    # We compute this ONCE before the main loop (not on every bar) for efficiency.
    # The series maps each date to 'bull' or 'bear', forward-filled so every
    # timestamp in the simulation has a value even if SPY didn't trade that day.
    #
    # Inside the loop we look up "the most recent regime <= current_timestamp"
    # so we never look ahead (no future leak: at bar T we only know SPY's SMA
    # up to bar T, not bar T+1).
    spy_regime_series = None
    if USE_REGIME_FILTER and REGIME_SYMBOL in all_daily_bars:
        spy_daily = all_daily_bars[REGIME_SYMBOL].copy()
        sma = ta.trend.sma_indicator(spy_daily['close'], window=REGIME_SMA_PERIOD)
        regime_raw = pd.Series(
            np.where(spy_daily['close'] > sma, 'bull', 'bear'),
            index=spy_daily.index,
        )
        # Forward-fill so gaps (weekends, holidays) carry the last known value.
        # NaN entries at the start (first 200 bars) are filled with 'bull'
        # — we default to permissive until we have enough data.
        regime_raw[sma.isna()] = 'bull'
        spy_regime_series = regime_raw.ffill()

        bear_pct = (spy_regime_series == 'bear').mean() * 100
        logging.info(
            f"{prefix}Regime filter: SPY in bear regime {bear_pct:.1f}% of trading days "
            f"in this window. Mean-reversion BUY signals suppressed on those days."
        )
    elif USE_REGIME_FILTER:
        logging.warning(f"{prefix}Regime filter: SPY bars not available. Defaulting to 'bull' throughout.")

    # Tracks the most recent traded price seen for each symbol, so positions
    # can be marked-to-market even on timestamps where their symbol has no bar.
    #
    # BUG FIX (2026-06-09): previously, a position whose symbol had no bar at
    # the *exact* current timestamp was simply omitted from the equity total --
    # valued at $0. One missing hourly bar on a thinly-traded symbol made
    # equity appear to crash by the position's full value, then "recover" a
    # bar later. This poisoned three things at once:
    #   1. Max drawdown (fake -60..99% spikes, which we saw in the benchmark),
    #   2. The drawdown throttle (fake dips triggered real size cuts),
    #   3. Sharpe (volatility inflated by phantom equity swings).
    last_known_price = {}

    # Run historical simulations sequentially chronologically
    for current_timestamp in sorted_timestamps:
        # Calculate current net portfolio value before any trade updates
        total_portfolio_equity = current_cash
        for pos_symbol, pos_data in open_positions.items():
            latest_pos_price = 0
            if pos_symbol in all_hourly_bars and current_timestamp in all_hourly_bars[pos_symbol].index:
                latest_pos_price = all_hourly_bars[pos_symbol].loc[current_timestamp, 'close']
            elif pos_symbol in all_daily_bars and current_timestamp in all_daily_bars[pos_symbol].index:
                latest_pos_price = all_daily_bars[pos_symbol].loc[current_timestamp, 'close']

            if latest_pos_price > 0:
                last_known_price[pos_symbol] = latest_pos_price
            else:
                # No bar at this exact timestamp: mark to the last traded
                # price (falling back to entry price if we've never seen one).
                latest_pos_price = last_known_price.get(pos_symbol, pos_data['entry_price'])

            total_portfolio_equity += (latest_pos_price * pos_data['qty'])

        # --- Lever 3: update the running peak and derive this bar's risk multiplier ---
        peak_equity = max(peak_equity, total_portfolio_equity)
        risk_multiplier = compute_risk_multiplier(total_portfolio_equity, peak_equity)

        # --- Lever 4: look up market regime at this exact timestamp ---
        # Use the most recent *fully completed* day's SPY regime (strictly
        # earlier calendar day — same lookahead rule as the daily bar slicing:
        # today's SPY close isn't knowable during today's trading hours).
        current_regime = 'bull'
        if spy_regime_series is not None:
            past = spy_regime_series[
                spy_regime_series.index.normalize() < current_timestamp.normalize()
            ]
            if not past.empty:
                current_regime = past.iloc[-1]

        # Calculate strategy signals and update portfolio balances for each stock
        for symbol in active_symbols:
            latest_atr = None # Initialize default value
            daily_bars_up_to_current = pd.DataFrame()
            hourly_bars_up_to_current = pd.DataFrame()

            if symbol in all_daily_bars:
                # LOOKAHEAD FIX (2026-06-09): Alpaca stamps a daily bar at the
                # START of its day (e.g. 05:00 UTC), but the bar's close price
                # isn't determined until the 4pm ET session close. Slicing with
                # `index <= current_timestamp` therefore handed strategies
                # *today's closing price* during today's morning hours -- a
                # classic lookahead leak (it inflated the ensemble backtest to
                # +360%/yr, Sharpe 11). A daily bar is only truly knowable once
                # its calendar day is over, so we require strictly-earlier days.
                # Daily strategies now signal off yesterday's close and execute
                # at today's price -- "next-bar execution", the standard
                # bias-free backtesting convention.
                sym_daily = all_daily_bars[symbol]
                daily_bars_up_to_current = sym_daily[
                    sym_daily.index.normalize() < current_timestamp.normalize()
                ]
            if symbol in all_hourly_bars:
                hourly_bars_up_to_current = all_hourly_bars[symbol][all_hourly_bars[symbol].index <= current_timestamp]

            latest_price_for_symbol = None
            if not hourly_bars_up_to_current.empty:
                latest_price_for_symbol = hourly_bars_up_to_current['close'].iloc[-1]
            elif not daily_bars_up_to_current.empty:
                latest_price_for_symbol = daily_bars_up_to_current['close'].iloc[-1]

            if latest_price_for_symbol is None:
                continue

            # Keep the mark-to-market price cache fresh for the equity calc.
            last_known_price[symbol] = latest_price_for_symbol

            bars_by_timeframe = {"daily": daily_bars_up_to_current, "hourly": hourly_bars_up_to_current}

            # ATR (Average True Range) measures how much a stock typically
            # moves per bar -- our universal "how volatile is this thing right
            # now" number, used by both ATR position sizing (lever 1) and ATR
            # stops (lever 2). Computed generically off daily bars so it's
            # always available, regardless of which strategies are enabled
            # (previously this was only populated when SmaCrossoverStrategy
            # ran, which is disabled -- silently turning ATR stops into a
            # dead feature).
            latest_atr = None
            if not daily_bars_up_to_current.empty and len(daily_bars_up_to_current) > ATR_PERIOD:
                atr_series = ta.volatility.average_true_range(
                    daily_bars_up_to_current['high'], daily_bars_up_to_current['low'],
                    daily_bars_up_to_current['close'], window=ATR_PERIOD
                ).dropna()
                if not atr_series.empty:
                    latest_atr = atr_series.iloc[-1]

            # --- Run every configured strategy on its designated timeframe ---
            signals = []
            for strategy, timeframe in STRATEGIES:
                bars = bars_by_timeframe[timeframe]
                if bars.empty or len(bars) < strategy.min_bars_required:
                    continue

                signal = strategy.generate_signal(symbol, bars)
                signals.append(signal)

                # Apply the 200-SMA trend filter generically (only strategies
                # that report a trend_sma, i.e. SMA crossover, are affected)
                trend_sma = signal.details.get("trend_sma")
                if trend_sma is not None:
                    if signal.action == BUY and latest_price_for_symbol < trend_sma:
                        signal.action = HOLD
                    elif signal.action == SELL and latest_price_for_symbol > trend_sma:
                        signal.action = HOLD

                # --- Lever 4: Market regime filter ---
                # Suppress mean-reversion BUY signals when SPY is in a downtrend.
                # SELL signals are left alone — exiting longs in a bear market is fine.
                if (current_regime == 'bear'
                        and strategy.strategy_type == 'mean_reversion'
                        and signal.action == BUY):
                    signal.action = HOLD

            # --- QUANT PORTFOLIO POSITION CAPPING ---
            # If we don't own the stock and are already at our maximum holdings, block new entries
            if symbol not in open_positions and len(open_positions) >= MAX_SIMULTANEOUS_POSITIONS:
                for signal in signals:
                    signal.action = HOLD
            # ----------------------------------------

            current_qty = open_positions[symbol]['qty'] if symbol in open_positions else 0
            chosen_signal = choose_signal_to_act_on(signals, current_qty)
            signal_to_act_on = chosen_signal.action if chosen_signal else HOLD
            strategy_name = chosen_signal.strategy_name if chosen_signal else None

            # Simulated Trade Executions
            current_cash, open_positions, trade_details = execute_trade_logic_backtest(
                symbol,
                signal_to_act_on,
                strategy_name,
                latest_price_for_symbol,
                current_timestamp,
                current_cash,
                open_positions,
                total_portfolio_equity,
                latest_atr,
                risk_multiplier
            )
 
            if trade_details:
                trade_history.append(trade_details)
                logging.info(
    f" Trade executed on {current_timestamp.date()} | "
    f"{trade_details['symbol']} {trade_details['side']} | "
    f"Qty: {trade_details['qty']} @ ${trade_details['price']:.2f} | "
    f"Reason: {trade_details['reason']} | "
    f"P/L: ${trade_details['P/L']:.2f}"
                )
        portfolio_history.append({'timestamp': current_timestamp, 'equity': total_portfolio_equity})

    logging.info(f"{prefix}Trade simulation complete.")

    # --- SUMMARY STATS REPORTING ---
    logging.info(f"\n{prefix}=========================================")
    logging.info(f"{prefix} HISTORICAL BACKTEST RESULTS SUMMARY")
    logging.info(f"{prefix}=========================================")
    logging.info(f"{prefix}Initial Starting Cash: ${initial_cash:,.2f}")
    logging.info(f"{prefix}Final Liquid Cash Balance: ${current_cash:,.2f}")

    final_equity = current_cash
    for pos_symbol, pos_data in open_positions.items():
        latest_pos_price = 0
        if pos_symbol in all_hourly_bars and not all_hourly_bars[pos_symbol].empty:
            latest_pos_price = all_hourly_bars[pos_symbol]['close'].iloc[-1]
        elif pos_symbol in all_daily_bars and not all_daily_bars[pos_symbol].empty:
            latest_pos_price = all_daily_bars[pos_symbol]['close'].iloc[-1]

        if latest_pos_price > 0:
            final_equity += (latest_pos_price * pos_data['qty'])

    logging.info(f"{prefix}Final Net Equity (Cash + Assets): ${final_equity:,.2f}")

    total_net_pl = final_equity - initial_cash
    pl_pct = (total_net_pl / initial_cash) * 100
    logging.info(f"{prefix}Net Realized Profit/Loss: ${total_net_pl:,.2f} ({pl_pct:+.2f}%)")
    logging.info(f"{prefix}Total Completed Trades: {len(trade_history)}")

    # --- Benchmark: SPY buy-and-hold over the same window ---
    # THE most important context for any backtest result. A strategy that
    # returns +8% in a year where simply buying the index returned +24% has
    # NEGATIVE alpha (-16pp): all that trading, risk, and complexity
    # underperformed doing nothing. Absolute returns flatter; alpha doesn't.
    benchmark_pct = None
    alpha_pp = None
    if BENCHMARK_SYMBOL in all_daily_bars and len(all_daily_bars[BENCHMARK_SYMBOL]) >= 2:
        bench_closes = all_daily_bars[BENCHMARK_SYMBOL]['close']
        benchmark_pct = (bench_closes.iloc[-1] / bench_closes.iloc[0] - 1) * 100
        alpha_pp = pl_pct - benchmark_pct
        logging.info(f"{prefix}Benchmark (SPY buy-and-hold): {benchmark_pct:+.2f}%")
        logging.info(f"{prefix}Alpha vs benchmark: {alpha_pp:+.2f} percentage points")
    else:
        logging.info(f"{prefix}Benchmark: N/A ({BENCHMARK_SYMBOL} bars not available)")

    # --- Per-Strategy P/L Breakdown ---
    # Only closing trades carry a non-zero, realized P/L  entries are
    # recorded with P/L 0.0 and tagged with their originating strategy so we
    # can see which strategy is actually making (or losing) money.
    strategy_breakdown = []
    if trade_history:
        trades_df = pd.DataFrame(trade_history)
        closed = trades_df[trades_df['P/L'] != 0.0]

        logging.info(f"{prefix}--- P/L Breakdown by Strategy ---")
        if closed.empty:
            logging.info(f"{prefix}  No closed trades with realized P/L yet.")
        else:
            summary = closed.groupby('strategy')['P/L'].agg(
                trades='count', total_pl='sum', avg_pl='mean', win_rate=lambda s: (s > 0).mean() * 100
            )
            for strat_name, row in summary.sort_values('total_pl', ascending=False).iterrows():
                logging.info(
                    f" {strat_name:20s} | Closed Trades: {int(row['trades']):4d} | "
                    f"Total P/L: ${row['total_pl']:>10,.2f} | "
                    f"Avg P/L: ${row['avg_pl']:>8,.2f} | "
                    f"Win Rate: {row['win_rate']:5.1f}%"
                )
                strategy_breakdown.append({
                    'strategy':  strat_name,
                    'trades':    int(row['trades']),
                    'total_pl':  round(float(row['total_pl']), 2),
                    'avg_pl':    round(float(row['avg_pl']), 2),
                    'win_rate':  round(float(row['win_rate']), 1),
                })

    # Calculate and Display Max Drawdown + Sharpe Ratio
    if portfolio_history:
        equity_df = pd.DataFrame(portfolio_history)
        equity_df = equity_df.set_index('timestamp')
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min() * 100 if not equity_df.empty else 0.0
        logging.info(f"{prefix}Maximum Backtest Drawdown: {max_drawdown:.2f}%")

        # Sharpe ratio: resample the mixed hourly/daily equity series to
        # end-of-day snapshots, then annualise.
        # Formula: (mean_daily_return * 252) / (std_daily_return * sqrt(252))
        #        = mean_daily_return * sqrt(252) / std_daily_return
        # We assume a 0% risk-free rate (simplest; conservative vs. ~4% T-bill).
        daily_equity = equity_df['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            logging.info(f"{prefix}Annualised Sharpe Ratio:   {sharpe:.2f}")
        else:
            sharpe = float('nan')
            logging.info(f"{prefix}Annualised Sharpe Ratio:   N/A (insufficient data)")

        # Build a serialisable daily equity curve (for the Streamlit dashboard)
        daily_equity_curve = [
            {'date': str(dt.date()), 'equity': round(float(eq), 2)}
            for dt, eq in daily_equity.items()
        ]
    else:
        max_drawdown = 0.0
        sharpe = float('nan')
        daily_equity_curve = []
        logging.info(f"{prefix}Maximum Backtest Drawdown: N/A")
        logging.info(f"{prefix}Annualised Sharpe Ratio:   N/A")
    logging.info("=========================================\n")

    return {
        'label': label,
        'start_date': start_date,
        'end_date': end_date,
        'final_equity': round(final_equity, 2),
        'total_pl': round(total_net_pl, 2),
        'pl_pct': round(pl_pct, 4),
        'max_drawdown': round(max_drawdown, 4),
        'sharpe': sharpe,
        'benchmark_pct': round(benchmark_pct, 4) if benchmark_pct is not None else None,
        'alpha_pp': round(alpha_pp, 4) if alpha_pp is not None else None,
        'num_trades': len(trade_history),
        'strategy_breakdown': strategy_breakdown,
        'daily_equity_curve': daily_equity_curve,
        # Raw lists kept for callers that want to do their own analysis
        'trade_history': trade_history,
        'portfolio_history': portfolio_history,
    }


# --- 9. MAIN ENTRY POINT ---
if __name__ == "__main__":
    all_daily_bars = {}
    all_hourly_bars = {}

    logging.info(f"Fetching data for {len(SYMBOLS)} symbols ({BACKTEST_START_DATE} -> {BACKTEST_END_DATE})...")
    for symbol in SYMBOLS:
        daily = get_historical_bars(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, TIME_FRAME)
        if not daily.empty:
            all_daily_bars[symbol] = daily

        hourly = get_historical_bars(symbol, BACKTEST_START_DATE, BACKTEST_END_DATE, SCALPING_TIME_FRAME)
        if not hourly.empty:
            all_hourly_bars[symbol] = hourly

    if not all_daily_bars and not all_hourly_bars:
        logging.error("No historical data fetched. Exiting.")
        exit()

    run_simulation(BACKTEST_START_DATE, BACKTEST_END_DATE, all_daily_bars, all_hourly_bars)
    logging.info("Backtester successfully finished.")

