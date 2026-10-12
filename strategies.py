"""
Pluggable trading strategy framework.

Each strategy is a small class that looks at a DataFrame of price bars and
produces a signal: "BUY", "SELL", or "HOLD". The bot (and the backtester)
don't need to know *how* a strategy decides — they just call
`strategy.generate_signal(symbol, bars)` and get a Signal back.

This is Python's "interface via inheritance" pattern: `Strategy` defines the
shape every strategy must have, and each subclass fills in the specifics.
Adding a new strategy later means writing one new class — nothing else in
the bot has to change.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import ta


@dataclass
class Signal:
    """The result of a strategy looking at the latest data for a symbol.

    A dataclass is just a regular class where Python auto-generates
    __init__ from the fields below — handy for simple "bag of data" objects.
    """
    action: str                 # "BUY", "SELL", or "HOLD"
    strategy_name: str
    # Free-form indicator values (e.g. {"rsi": 28.4, "sma_fast": 101.2}),
    # kept around for logging/debugging so you can see *why* a signal fired.
    details: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL")


HOLD = "HOLD"
BUY = "BUY"
SELL = "SELL"


class Strategy:
    """Base class — defines the interface every strategy must implement.

    Subclasses must override `min_bars_required` and `_decide`.
    `generate_signal` itself shouldn't need to be overridden: it handles the
    boring, repeated checks (enough data? sorted index?) once, in one place,
    instead of duplicating that boilerplate in every strategy function.

    strategy_type classifies *how* the strategy makes money, which drives the
    market regime filter:
      "trend"          — profits from sustained directional moves. Works best in
                         trending markets. Examples: MACD, SMA Crossover.
      "mean_reversion" — profits from prices snapping back to a typical range.
                         Works best in calm, range-bound markets; breaks down
                         badly in strong trends (oversold just keeps going down).
                         Examples: Bollinger Bands, RSI extremes.
    """

    name: str = "BaseStrategy"
    min_bars_required: int = 0
    strategy_type: str = "trend"   # override in mean-reversion subclasses

    def _decide(self, symbol: str, bars: pd.DataFrame) -> Signal:
        """Override in subclasses: compute indicators and return a Signal."""
        raise NotImplementedError

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal:
        if bars.empty or len(bars) < self.min_bars_required:
            return Signal(HOLD, self.name, {"reason": "not_enough_data", "bars": len(bars)})

        bars = bars.sort_index()
        return self._decide(symbol, bars)


class SmaCrossoverStrategy(Strategy):
    """Trend-following: buy when a fast SMA crosses above a slow SMA (and
    we're above the long-term trend filter), sell on the opposite cross.

    Concept: an SMA (Simple Moving Average) smooths out daily noise. When the
    short-term average crosses above the long-term average, it suggests
    momentum is shifting upward — a classic "golden cross" entry signal.
    """

    name = "SMA_Crossover"

    def __init__(self, fast_period=20, slow_period=50, trend_filter_period=200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_filter_period = trend_filter_period
        # +2 because we need the *previous* bar's SMA values too, to detect
        # a crossover (a change from below to above between two bars).
        self.min_bars_required = trend_filter_period + 2

    def _decide(self, symbol, bars):
        fast = ta.trend.sma_indicator(bars["close"], window=self.fast_period)
        slow = ta.trend.sma_indicator(bars["close"], window=self.slow_period)
        trend = ta.trend.sma_indicator(bars["close"], window=self.trend_filter_period)

        df = pd.DataFrame({"fast": fast, "slow": slow, "trend": trend}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {
            "fast_sma": latest["fast"],
            "slow_sma": latest["slow"],
            "trend_sma": latest["trend"],
        }

        crossed_up = prev["fast"] < prev["slow"] and latest["fast"] >= latest["slow"]
        crossed_down = prev["fast"] > prev["slow"] and latest["fast"] <= latest["slow"]

        if crossed_up:
            return Signal(BUY, self.name, details)
        if crossed_down:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class MacdMomentumStrategy(Strategy):
    """Trend-momentum: buy when the MACD line crosses above its signal line
    (momentum is accelerating upward), sell on the opposite cross.

    Concept walkthrough, since this is new vocabulary:

    1. An EMA (Exponential Moving Average) is like the SMAs we already use,
       except recent prices are weighted more heavily than old ones. That
       makes it react faster to new information — useful for *momentum*
       measures, where "how fast is this changing right now" matters more
       than "what's the average over the last N days".

    2. The "MACD line" is just: EMA(fast) - EMA(slow). When the fast EMA is
       above the slow EMA, this number is positive and growing -> upward
       momentum is *increasing*. When fast dips below slow, it goes negative.

    3. The "signal line" is a smoothed (EMA) version of the MACD line itself.
       It lags the MACD line slightly, the same way a slow SMA lags a fast
       SMA. So MACD-crossing-its-signal-line is the same *kind* of crossover
       idea as `SmaCrossoverStrategy` — just one derivative further removed:
       instead of "is price trending up", it asks "is the trend's *strength*
       turning up". That's a meaningfully different question, which is why
       this strategy tends to fire at different times than our other ones —
       good for diversifying when the ensemble trades.

    `ta.trend.MACD` computes all three series for us; we just need to detect
    where `macd_line` and `macd_signal` cross, exactly like we did by hand
    for the two SMAs in `SmaCrossoverStrategy`.
    """

    name = "MACD_Momentum"

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        # +2: one extra bar so we can compare "previous" vs "latest" and
        # detect a crossover (a flip in which line is on top).
        self.min_bars_required = slow_period + signal_period + 2

    def _decide(self, symbol, bars):
        macd = ta.trend.MACD(
            bars["close"],
            window_fast=self.fast_period,
            window_slow=self.slow_period,
            window_sign=self.signal_period,
        )
        df = pd.DataFrame({
            "macd_line": macd.macd(),
            "signal_line": macd.macd_signal(),
        }).dropna()

        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {
            "macd_line": latest["macd_line"],
            "signal_line": latest["signal_line"],
            "histogram": latest["macd_line"] - latest["signal_line"],
        }

        crossed_up = prev["macd_line"] < prev["signal_line"] and latest["macd_line"] >= latest["signal_line"]
        crossed_down = prev["macd_line"] > prev["signal_line"] and latest["macd_line"] <= latest["signal_line"]

        if crossed_up:
            return Signal(BUY, self.name, details)
        if crossed_down:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class RsiMomentumStrategy(Strategy):
    """Mean-reversion-on-a-trend ("scalping"): buy oversold dips that are
    still in an uptrend, sell overbought rallies that are still in a downtrend.

    Concept: RSI (Relative Strength Index, 0-100) measures how fast and how
    much price has moved recently. Below ~30 is "oversold" (often due for a
    bounce), above ~70 is "overbought" (often due for a pullback). We only
    act on these extremes *with* the prevailing trend (price vs. its SMA),
    which filters out a lot of false signals.
    """

    name = "RSI_Momentum"
    strategy_type = "mean_reversion"

    def __init__(self, rsi_period=14, oversold=30, overbought=70, trend_sma_period=50):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_sma_period = trend_sma_period
        self.min_bars_required = max(rsi_period, trend_sma_period) + 1

    def _decide(self, symbol, bars):
        rsi = ta.momentum.rsi(bars["close"], window=self.rsi_period)
        trend_sma = ta.trend.sma_indicator(bars["close"], window=self.trend_sma_period)

        df = pd.DataFrame({"close": bars["close"], "rsi": rsi, "sma": trend_sma}).dropna()
        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        latest = df.iloc[-1]
        details = {"rsi": latest["rsi"], "trend_sma": latest["sma"], "close": latest["close"]}

        if latest["rsi"] < self.oversold and latest["close"] > latest["sma"]:
            return Signal(BUY, self.name, details)
        if latest["rsi"] > self.overbought and latest["close"] < latest["sma"]:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class BollingerBandReversionStrategy(Strategy):
    """Mean-reversion: buy when price punches below the lower Bollinger Band
    (statistically "too cheap" relative to recent volatility), sell when it
    punches above the upper band ("too expensive").

    Concept: Bollinger Bands are a moving average plus/minus a multiple of
    the rolling standard deviation. Price tends to oscillate within the
    bands; touching/crossing a band is a classic mean-reversion trigger.
    This strategy fires more often than SMA crossovers (which need a slow
    trend shift), making it a good complement for more frequent trading on
    range-bound or choppy symbols.
    """

    name = "Bollinger_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, window=20, num_std=2.0):
        self.window = window
        self.num_std = num_std
        self.min_bars_required = window + 1

    def _decide(self, symbol, bars):
        bb = ta.volatility.BollingerBands(bars["close"], window=self.window, window_dev=self.num_std)
        df = pd.DataFrame({
            "close": bars["close"],
            "upper": bb.bollinger_hband(),
            "lower": bb.bollinger_lband(),
            "mid": bb.bollinger_mavg(),
        }).dropna()

        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        latest = df.iloc[-1]
        details = {
            "close": latest["close"],
            "upper_band": latest["upper"],
            "lower_band": latest["lower"],
            "mid_band": latest["mid"],
        }

        if latest["close"] <= latest["lower"]:
            return Signal(BUY, self.name, details)
        if latest["close"] >= latest["upper"]:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class RocMomentumStrategy(Strategy):
    """Trend-momentum: buy when the Rate of Change crosses from negative to
    positive (price momentum flipping upward), sell on the reverse cross.

    Concept: ROC is the simplest momentum measure there is —
        ROC = (close_now - close_N_bars_ago) / close_N_bars_ago * 100
    i.e. "how far has price moved over the last N bars, in percent". While
    MACD measures momentum *of the trend* (via smoothed EMAs), ROC measures
    raw price momentum directly, no smoothing. That makes it noisier but
    faster to react — it fires earlier in a turn than MACD does, at the cost
    of more false starts.

    The zero line is the natural trigger: ROC > 0 means price is higher than
    it was N bars ago (upward momentum exists at all), ROC < 0 means it's
    lower. Crossing zero is the exact moment momentum changes direction.

    BUY additionally requires close > SMA(50) — the same "only trade with the
    prevailing trend" filter RSI Momentum uses, which kills most of the false
    starts from ROC's noisiness. SELLs are not trend-filtered: momentum
    flipping negative is reason enough to exit.

    Ported from the 20-strategy benchmark (2026-06-09): solo on 2025 daily
    bars it returned +15.06% (Sharpe 0.60, 24 trades) vs baseline +6.36%.
    """

    name = "ROC_Momentum"
    strategy_type = "trend"

    def __init__(self, window=12, trend_sma=50):
        self.window = window
        self.trend_sma = trend_sma
        # +2: need a previous bar too, to detect the zero-line crossing.
        self.min_bars_required = max(window, trend_sma) + 2

    def _decide(self, symbol, bars):
        roc = ta.momentum.roc(bars["close"], window=self.window)
        sma = ta.trend.sma_indicator(bars["close"], window=self.trend_sma)

        df = pd.DataFrame({"roc": roc, "sma": sma, "close": bars["close"]}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        # NOTE: deliberately NOT named "trend_sma" in details — that key would
        # trigger the bot's generic 200-SMA trend filter, double-filtering us.
        details = {"roc": latest["roc"], "sma": latest["sma"]}

        bull_cross = prev["roc"] < 0 and latest["roc"] >= 0 and latest["close"] > latest["sma"]
        bear_cross = prev["roc"] > 0 and latest["roc"] <= 0

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class AtrChannelBreakoutStrategy(Strategy):
    """Volatility breakout: buy when price punches above an ATR-scaled channel
    around a slow EMA, sell when it punches below.

    Concept: this is the *mirror image* of Bollinger reversion. Both draw a
    band around a moving average, but they bet in opposite directions:

      - Bollinger reversion says "price outside the band is an anomaly that
        will snap back" -> BUY at the lower band (fade the move).
      - ATR channel breakout says "price outside the band is a breakout that
        will continue" -> BUY at the *upper* band (ride the move).

    Both can be right — on different stocks at different times. Noise-driven
    excursions revert; news/trend-driven excursions continue. Running both
    means the ensemble has a position for either case, and the arbitration
    logic + position cap decide which gets capital.

    Using ATR for the band width (instead of standard deviation like
    Bollinger) adapts the breakout threshold to each stock's volatility:
    a calm utility stock only needs a small absolute move to count as a
    breakout, while NVDA needs a much bigger one. That's the same
    "volatility-normalised" idea as our ATR position sizing.

    Ported from the 20-strategy benchmark (2026-06-09): solo on 2025 daily
    bars it was the top earner — +16.84% (Sharpe 0.70, 60 trades) vs
    baseline +6.36%.
    """

    name = "ATR_Channel_Breakout"
    strategy_type = "trend"

    def __init__(self, ema_window=20, atr_window=14, multiplier=1.5):
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.multiplier = multiplier
        # +2: need a previous bar to detect the cross from inside to outside.
        self.min_bars_required = max(ema_window, atr_window) + 2

    def _decide(self, symbol, bars):
        ema = ta.trend.ema_indicator(bars["close"], window=self.ema_window)
        atr = ta.volatility.average_true_range(
            bars["high"], bars["low"], bars["close"], window=self.atr_window
        )

        upper = ema + self.multiplier * atr
        lower = ema - self.multiplier * atr

        df = pd.DataFrame({
            "close": bars["close"], "upper": upper, "lower": lower
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"close": latest["close"], "upper": latest["upper"], "lower": latest["lower"]}

        # Require the *crossing* (prev inside, latest outside), not just being
        # outside the band — otherwise we'd re-fire every bar of a long move.
        broke_up = prev["close"] < prev["upper"] and latest["close"] >= latest["upper"]
        broke_dn = prev["close"] > prev["lower"] and latest["close"] <= latest["lower"]

        if broke_up:
            return Signal(BUY, self.name, details)
        if broke_dn:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class VwapReversionStrategy(Strategy):
    """Mean-reversion around the intraday VWAP.

    VWAP (Volume-Weighted Average Price) is the most-watched intraday
    reference price by institutional desks. Price oscillates around VWAP
    throughout the session; large deviations statistically revert.

    We compute today's cumulative VWAP from session open and add ±N
    rolling std-dev bands. A touch of the lower band → BUY (expect
    reversion to VWAP), upper band → SELL.

    Designed for 1-minute bars. min_bars_required = 30 means we need at
    least 30 min of data before firing, so opening volatility doesn't
    trigger false signals.
    """

    name = "VWAP_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, std_dev_multiplier: float = 1.5, min_bars: int = 30):
        self.std_dev_multiplier = std_dev_multiplier
        self.min_bars_required = min_bars

    def _today_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.index.tz is not None:
            et = bars.index.tz_convert("America/New_York")
            return bars[et.normalize() == et[-1].normalize()]
        return bars[bars.index.normalize() == bars.index[-1].normalize()]

    def _decide(self, symbol: str, bars: pd.DataFrame) -> Signal:
        today = self._today_bars(bars)
        if len(today) < self.min_bars_required:
            return Signal(HOLD, self.name, {"reason": "not_enough_intraday_bars", "bars": len(today)})

        typical = (today["high"] + today["low"] + today["close"]) / 3
        vwap = (typical * today["volume"]).cumsum() / today["volume"].cumsum()
        std = typical.rolling(window=min(20, len(today))).std()

        df = pd.DataFrame({"close": today["close"], "vwap": vwap, "std": std}).dropna()
        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        latest = df.iloc[-1]
        upper = latest["vwap"] + self.std_dev_multiplier * latest["std"]
        lower = latest["vwap"] - self.std_dev_multiplier * latest["std"]
        details = {"close": latest["close"], "vwap": latest["vwap"], "upper": upper, "lower": lower}

        if latest["close"] <= lower:
            return Signal(BUY, self.name, details)
        if latest["close"] >= upper:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class OpeningRangeBreakoutStrategy(Strategy):
    """Breakout above/below the first N minutes' high/low.

    The Opening Range (OR) is the high and low set in the first N minutes
    after market open (9:30 AM ET). Breaking above the OR high signals
    bullish continuation; below the OR low signals bearish. One of the
    most consistently documented intraday setups across asset classes.

    Only fires after the OR window closes. Requires timezone-aware bars.
    """

    name = "Opening_Range_Breakout"
    strategy_type = "trend"

    def __init__(self, or_minutes: int = 30):
        self.or_minutes = or_minutes
        self.min_bars_required = or_minutes + 2

    def _decide(self, symbol: str, bars: pd.DataFrame) -> Signal:
        if bars.index.tz is None:
            return Signal(HOLD, self.name, {"reason": "timezone_required"})

        et = bars.index.tz_convert("America/New_York")
        last_et = et[-1]
        market_open = last_et.normalize().replace(hour=9, minute=30)
        or_end = market_open + pd.Timedelta(minutes=self.or_minutes)

        if last_et <= or_end:
            return Signal(HOLD, self.name, {"reason": "still_in_opening_range"})

        or_mask = (et >= market_open) & (et <= or_end)
        or_bars = bars[or_mask]
        if len(or_bars) < self.or_minutes // 2:
            return Signal(HOLD, self.name, {"reason": "opening_range_not_established"})

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        latest_close = bars["close"].iloc[-1]
        prev_close = bars["close"].iloc[-2]

        details = {"close": latest_close, "or_high": or_high, "or_low": or_low}

        if prev_close <= or_high and latest_close > or_high:
            return Signal(BUY, self.name, details)
        if prev_close >= or_low and latest_close < or_low:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class EmaScalpStrategy(Strategy):
    """Fast EMA crossover for intraday scalping.

    5-period EMA (very reactive) crossing a 13-period EMA (moderate).
    On 1-minute bars these fire several times per day per symbol. VWAP
    acts as the intraday trend filter: only BUY above VWAP, only SELL
    below — keeps us trading with the session trend rather than against it.

    Why 5/13? Classic Fibonacci-based scalping periods. More signals
    than 12/26 MACD, less noise than 3/5 pairs.
    """

    name = "EMA_Scalp"
    strategy_type = "trend"

    def __init__(self, fast: int = 5, slow: int = 13):
        self.fast = fast
        self.slow = slow
        self.min_bars_required = slow + 2

    def _intraday_vwap(self, bars: pd.DataFrame) -> Optional[float]:
        if bars.index.tz is not None:
            et = bars.index.tz_convert("America/New_York")
            today = bars[et.normalize() == et[-1].normalize()]
        else:
            today = bars[bars.index.normalize() == bars.index[-1].normalize()]
        if len(today) < 5:
            return None
        typical = (today["high"] + today["low"] + today["close"]) / 3
        vwap = (typical * today["volume"]).cumsum() / today["volume"].cumsum()
        return float(vwap.iloc[-1])

    def _decide(self, symbol: str, bars: pd.DataFrame) -> Signal:
        fast_ema = ta.trend.ema_indicator(bars["close"], window=self.fast)
        slow_ema = ta.trend.ema_indicator(bars["close"], window=self.slow)

        df = pd.DataFrame({"fast": fast_ema, "slow": slow_ema}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        vwap = self._intraday_vwap(bars)
        latest_close = float(bars["close"].iloc[-1])
        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"fast_ema": latest["fast"], "slow_ema": latest["slow"], "vwap": vwap, "close": latest_close}

        crossed_up = prev["fast"] < prev["slow"] and latest["fast"] >= latest["slow"]
        crossed_dn = prev["fast"] > prev["slow"] and latest["fast"] <= latest["slow"]

        above_vwap = vwap is None or latest_close > vwap
        below_vwap = vwap is None or latest_close < vwap

        if crossed_up and above_vwap:
            return Signal(BUY, self.name, details)
        if crossed_dn and below_vwap:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


class FastRsiScalpStrategy(Strategy):
    """Ultra-fast RSI(3) scalper on 1-minute bars.

    RSI(3) reaches extremes (< 15 oversold, > 85 overbought) many times
    per session during short momentum spikes and reliably mean-reverts on
    the 1-3 bar timescale. Tighter thresholds than the daily RSI(14)
    strategy — the shorter the RSI window, the faster it overshoots and
    snaps back, making very high-frequency counter-trend trades viable.
    """

    name = "Fast_RSI_Scalp"
    strategy_type = "mean_reversion"

    def __init__(self, rsi_period: int = 3, oversold: float = 15.0, overbought: float = 85.0):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.min_bars_required = rsi_period + 2

    def _decide(self, symbol: str, bars: pd.DataFrame) -> Signal:
        rsi = ta.momentum.rsi(bars["close"], window=self.rsi_period)
        df = pd.DataFrame({"rsi": rsi, "close": bars["close"]}).dropna()
        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data_after_indicators"})

        latest = df.iloc[-1]
        details = {"rsi": latest["rsi"], "close": latest["close"]}

        if latest["rsi"] < self.oversold:
            return Signal(BUY, self.name, details)
        if latest["rsi"] > self.overbought:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


def choose_signal_to_act_on(signals, current_position):
    """Generalized signal arbitration shared by the live bot and backtester.

    `signals` is a list of Signal objects in priority order (fastest/most
    frequent strategy first). The rule:
      - Flat (no position): act on the first non-HOLD signal we see.
      - In a position: only act on a signal that opposes the current position
        (i.e. an exit/reversal) — we don't want one strategy's BUY piling
        onto a position another strategy already opened.

    Centralizing this means the live bot and the backtester always evaluate
    signals identically — which matters, because if they disagree your
    backtest results don't actually predict what the live bot will do.
    """
    for signal in signals:
        if not signal.is_actionable:
            continue

        if current_position == 0:
            return signal

        wants_to_exit_long = current_position > 0 and signal.action == SELL
        wants_to_exit_short = current_position < 0 and signal.action == BUY
        if wants_to_exit_long or wants_to_exit_short:
            return signal

    return None
