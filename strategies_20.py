"""
20 additional trading strategies for backtesting.

Each strategy subclasses Strategy from strategies.py and implements:
  - name: str
  - min_bars_required: int
  - strategy_type: "trend" | "mean_reversion"
  - _decide(symbol, bars) -> Signal

Strategy catalogue
==================
TREND-FOLLOWING (profit from sustained directional moves):
  1.  EmaCrossoverStrategy        — EMA(9) vs EMA(21) crossover
  2.  TripleEmaCrossoverStrategy  — EMA(5) x EMA(13) x EMA(50) alignment
  3.  DonchianBreakoutStrategy    — price breaks above/below N-day high/low channel
  4.  AdxDiStrategy               — ADX>25 strength filter + DI+/DI- crossover
  5.  SupertrendStrategy          — ATR-based trailing stop/direction indicator
  6.  ParabolicSarStrategy        — Parabolic SAR direction flip
  7.  RocMomentumStrategy         — Rate of Change crosses zero
  8.  TrixStrategy                — TRIX line crosses its signal
  9.  MomentumBreakoutStrategy    — price breaks above rolling N-day high
  10. AroonOscillatorStrategy     — Aroon oscillator crosses ±50 threshold

MEAN-REVERSION (profit from prices snapping back to their typical range):
  11. StochasticReversionStrategy — Stochastic %K/%D crossover from oversold/overbought
  12. WilliamsRStrategy           — Williams %R below -80 (oversold) / above -20 (overbought)
  13. CciReversionStrategy        — CCI below -100 / above +100
  14. KeltnerReversionStrategy    — price touches lower/upper Keltner channel band
  15. ZscoreReversionStrategy     — z-score of price vs rolling mean > 2σ

VOLATILITY / COMBINATION:
  16. AtrChannelBreakoutStrategy  — price breaks above/below ATR-scaled channel from EMA
  17. VolatilitySqueezeStrategy   — Bollinger Bands squeeze inside Keltner Channels releases
  18. MacdRsiComboStrategy        — MACD crossover confirmed by RSI not in opposite extreme
  19. EmaVolumeStrategy           — EMA crossover with volume above rolling average
  20. AdxMacdComboStrategy        — MACD crossover gated by ADX trend strength
"""

import numpy as np
import pandas as pd
import ta

from strategies import Strategy, Signal, BUY, SELL, HOLD


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA Crossover
# ─────────────────────────────────────────────────────────────────────────────
class EmaCrossoverStrategy(Strategy):
    """Buy when the fast EMA crosses above the slow EMA, sell on the reverse.

    EMAs weight recent prices more than SMAs, making crossovers react slightly
    faster to trend changes — useful for capturing momentum earlier.
    """

    name = "EMA_Crossover"
    strategy_type = "trend"

    def __init__(self, fast=9, slow=21, trend_filter=50):
        self.fast = fast
        self.slow = slow
        self.trend_filter = trend_filter
        self.min_bars_required = trend_filter + 2

    def _decide(self, symbol, bars):
        fast_ema = ta.trend.ema_indicator(bars["close"], window=self.fast)
        slow_ema = ta.trend.ema_indicator(bars["close"], window=self.slow)
        trend_ema = ta.trend.ema_indicator(bars["close"], window=self.trend_filter)

        df = pd.DataFrame({"fast": fast_ema, "slow": slow_ema, "trend": trend_ema}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"fast_ema": latest["fast"], "slow_ema": latest["slow"], "trend_ema": latest["trend"]}

        crossed_up = prev["fast"] < prev["slow"] and latest["fast"] >= latest["slow"]
        crossed_dn = prev["fast"] > prev["slow"] and latest["fast"] <= latest["slow"]

        if crossed_up and bars["close"].iloc[-1] > latest["trend"]:
            return Signal(BUY, self.name, details)
        if crossed_dn:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Triple EMA Crossover
# ─────────────────────────────────────────────────────────────────────────────
class TripleEmaCrossoverStrategy(Strategy):
    """Buy when all three EMAs are aligned bullishly (fast > mid > slow),
    sell when aligned bearishly (fast < mid < slow).

    Requiring three levels of alignment reduces false signals in choppy markets —
    all three must agree before we act.
    """

    name = "Triple_EMA_Crossover"
    strategy_type = "trend"

    def __init__(self, fast=5, mid=13, slow=50):
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self.min_bars_required = slow + 2

    def _decide(self, symbol, bars):
        f = ta.trend.ema_indicator(bars["close"], window=self.fast)
        m = ta.trend.ema_indicator(bars["close"], window=self.mid)
        s = ta.trend.ema_indicator(bars["close"], window=self.slow)

        df = pd.DataFrame({"fast": f, "mid": m, "slow": s}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"fast": latest["fast"], "mid": latest["mid"], "slow": latest["slow"]}

        bull_now = latest["fast"] > latest["mid"] > latest["slow"]
        bear_now = latest["fast"] < latest["mid"] < latest["slow"]
        bull_prev = prev["fast"] > prev["mid"] > prev["slow"]
        bear_prev = prev["fast"] < prev["mid"] < prev["slow"]

        if bull_now and not bull_prev:
            return Signal(BUY, self.name, details)
        if bear_now and not bear_prev:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Donchian Channel Breakout
# ─────────────────────────────────────────────────────────────────────────────
class DonchianBreakoutStrategy(Strategy):
    """Buy when price breaks above the N-bar high (new high = strength), sell
    when it breaks below the N-bar low.

    The classic Turtle Trading entry: breakouts tend to persist in trending
    markets because they flush out overhead resistance.
    """

    name = "Donchian_Breakout"
    strategy_type = "trend"

    def __init__(self, window=20):
        self.window = window
        self.min_bars_required = window + 2

    def _decide(self, symbol, bars):
        upper = bars["high"].rolling(self.window).max()
        lower = bars["low"].rolling(self.window).min()

        df = pd.DataFrame({
            "close": bars["close"], "upper": upper, "lower": lower
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"close": latest["close"], "upper": latest["upper"], "lower": latest["lower"]}

        broke_up = prev["close"] < prev["upper"] and latest["close"] >= latest["upper"]
        broke_dn = prev["close"] > prev["lower"] and latest["close"] <= latest["lower"]

        if broke_up:
            return Signal(BUY, self.name, details)
        if broke_dn:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADX + DI Crossover
# ─────────────────────────────────────────────────────────────────────────────
class AdxDiStrategy(Strategy):
    """Enter only when the market is actually trending (ADX > 25), then use
    the directional-indicator (+DI vs -DI) crossover as the direction trigger.

    ADX (Average Directional Index) measures *trend strength* without direction.
    Below 25 = choppy / no trend; above 25 = trend worth following.
    +DI above -DI = bullish trend pressure; the opposite = bearish.
    """

    name = "ADX_DI"
    strategy_type = "trend"

    def __init__(self, window=14, adx_threshold=25):
        self.window = window
        self.adx_threshold = adx_threshold
        self.min_bars_required = window * 2 + 2

    def _decide(self, symbol, bars):
        adx = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=self.window)
        di_pos = ta.trend.adx_pos(bars["high"], bars["low"], bars["close"], window=self.window)
        di_neg = ta.trend.adx_neg(bars["high"], bars["low"], bars["close"], window=self.window)

        df = pd.DataFrame({"adx": adx, "di_pos": di_pos, "di_neg": di_neg}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"adx": latest["adx"], "di_pos": latest["di_pos"], "di_neg": latest["di_neg"]}

        if latest["adx"] < self.adx_threshold:
            return Signal(HOLD, self.name, {**details, "reason": "no_trend"})

        bull_cross = prev["di_pos"] < prev["di_neg"] and latest["di_pos"] >= latest["di_neg"]
        bear_cross = prev["di_pos"] > prev["di_neg"] and latest["di_pos"] <= latest["di_neg"]

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Supertrend
# ─────────────────────────────────────────────────────────────────────────────
class SupertrendStrategy(Strategy):
    """Supertrend: an ATR-based trailing stop that defines the trend direction.

    The indicator draws a band above/below price using ATR multiples; when price
    closes above the band the trend is bullish, below = bearish. A direction flip
    triggers the entry signal.

    Computed manually because ta-lib's supertrend API varies by version.
    """

    name = "Supertrend"
    strategy_type = "trend"

    def __init__(self, window=10, multiplier=3.0):
        self.window = window
        self.multiplier = multiplier
        self.min_bars_required = window * 2 + 2

    def _decide(self, symbol, bars):
        hl2 = (bars["high"] + bars["low"]) / 2
        atr = ta.volatility.average_true_range(bars["high"], bars["low"], bars["close"], window=self.window)

        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        # Calculate Supertrend direction (1 = bullish, -1 = bearish)
        close = bars["close"]
        direction = pd.Series(index=bars.index, dtype=float)
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()

        for i in range(1, len(bars)):
            # Upper band: only moves down (ratchets tight when bearish)
            if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

            # Lower band: only moves up (ratchets tight when bullish)
            if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]

            # Direction: follow price relative to the active band
            prev_dir = direction.iloc[i - 1] if i > 1 else 1.0
            if prev_dir == -1 and close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1   # flip to bullish
            elif prev_dir == 1 and close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1  # flip to bearish
            else:
                direction.iloc[i] = prev_dir

        df = pd.DataFrame({"direction": direction, "close": close}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_dir = df["direction"].iloc[-2]
        curr_dir = df["direction"].iloc[-1]
        details = {"direction": curr_dir, "close": df["close"].iloc[-1]}

        if prev_dir != 1 and curr_dir == 1:
            return Signal(BUY, self.name, details)
        if prev_dir != -1 and curr_dir == -1:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Parabolic SAR
# ─────────────────────────────────────────────────────────────────────────────
class ParabolicSarStrategy(Strategy):
    """Parabolic SAR: a trailing stop that accelerates as the trend matures.

    When SAR flips from above price (bearish) to below price (bullish) → BUY.
    When SAR flips from below to above → SELL.
    Tends to generate more signals than SMA crossovers, useful for capturing
    shorter intermediate trends.
    """

    name = "Parabolic_SAR"
    strategy_type = "trend"

    def __init__(self, step=0.02, max_step=0.2):
        self.step = step
        self.max_step = max_step
        self.min_bars_required = 30

    def _decide(self, symbol, bars):
        # ta provides psar_up (value when bullish) and psar_down (value when bearish)
        psar_up = ta.trend.PSARIndicator(
            bars["high"], bars["low"], bars["close"],
            step=self.step, max_step=self.max_step
        ).psar_up()

        psar_down = ta.trend.PSARIndicator(
            bars["high"], bars["low"], bars["close"],
            step=self.step, max_step=self.max_step
        ).psar_down()

        # psar_up is non-NaN when we're in a bull phase (SAR below price)
        # psar_down is non-NaN when we're in a bear phase (SAR above price)
        bull_phase = ~psar_up.isna()  # True when bullish

        if len(bull_phase) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_bull = bull_phase.iloc[-2]
        curr_bull = bull_phase.iloc[-1]
        close = bars["close"].iloc[-1]
        details = {"close": close, "bull_phase": bool(curr_bull)}

        if not prev_bull and curr_bull:
            return Signal(BUY, self.name, details)
        if prev_bull and not curr_bull:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rate of Change (ROC) Momentum
# ─────────────────────────────────────────────────────────────────────────────
class RocMomentumStrategy(Strategy):
    """Buy when ROC crosses from negative to positive (momentum turning up),
    sell on the reverse.

    ROC = (close_now - close_N_bars_ago) / close_N_bars_ago * 100
    Crossing zero captures the moment raw price momentum changes direction.
    """

    name = "ROC_Momentum"
    strategy_type = "trend"

    def __init__(self, window=12, trend_sma=50):
        self.window = window
        self.trend_sma = trend_sma
        self.min_bars_required = max(window, trend_sma) + 2

    def _decide(self, symbol, bars):
        roc = ta.momentum.roc(bars["close"], window=self.window)
        sma = ta.trend.sma_indicator(bars["close"], window=self.trend_sma)

        df = pd.DataFrame({"roc": roc, "sma": sma, "close": bars["close"]}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"roc": latest["roc"], "sma": latest["sma"]}

        bull_cross = prev["roc"] < 0 and latest["roc"] >= 0 and latest["close"] > latest["sma"]
        bear_cross = prev["roc"] > 0 and latest["roc"] <= 0

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRIX
# ─────────────────────────────────────────────────────────────────────────────
class TrixStrategy(Strategy):
    """TRIX = triple-smoothed EMA, expressed as a % rate of change.

    Because it's triple-smoothed, TRIX filters out more noise than MACD.
    Signal: TRIX crossing its own EMA signal line (same crossover logic as MACD).
    """

    name = "TRIX"
    strategy_type = "trend"

    def __init__(self, window=15, signal=9):
        self.window = window
        self.signal = signal
        self.min_bars_required = window * 3 + signal + 2

    def _decide(self, symbol, bars):
        trix_line = ta.trend.trix(bars["close"], window=self.window)
        signal_line = ta.trend.ema_indicator(trix_line, window=self.signal)

        df = pd.DataFrame({"trix": trix_line, "signal": signal_line}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"trix": latest["trix"], "signal": latest["signal"]}

        bull_cross = prev["trix"] < prev["signal"] and latest["trix"] >= latest["signal"]
        bear_cross = prev["trix"] > prev["signal"] and latest["trix"] <= latest["signal"]

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Momentum Breakout (N-day High)
# ─────────────────────────────────────────────────────────────────────────────
class MomentumBreakoutStrategy(Strategy):
    """Buy when today's close sets a new N-day closing high (breakout strength),
    sell when it sets a new N-day closing low.

    Simpler than Donchian (which uses the high/low wicks), this fires only on
    closing-price highs, reducing noise from intraday spikes.
    """

    name = "Momentum_Breakout"
    strategy_type = "trend"

    def __init__(self, window=20):
        self.window = window
        self.min_bars_required = window + 2

    def _decide(self, symbol, bars):
        rolling_high = bars["close"].rolling(self.window).max().shift(1)
        rolling_low = bars["close"].rolling(self.window).min().shift(1)

        df = pd.DataFrame({
            "close": bars["close"], "r_high": rolling_high, "r_low": rolling_low
        }).dropna()
        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        latest = df.iloc[-1]
        details = {"close": latest["close"], "rolling_high": latest["r_high"], "rolling_low": latest["r_low"]}

        if latest["close"] > latest["r_high"]:
            return Signal(BUY, self.name, details)
        if latest["close"] < latest["r_low"]:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Aroon Oscillator
# ─────────────────────────────────────────────────────────────────────────────
class AroonOscillatorStrategy(Strategy):
    """Aroon oscillator = Aroon Up - Aroon Down, ranging -100 to +100.

    Crossing above +50 signals strong uptrend; below -50 signals strong downtrend.
    This threshold approach reduces whipsaws vs. a simple zero-cross.
    """

    name = "Aroon_Oscillator"
    strategy_type = "trend"

    def __init__(self, window=25, threshold=50):
        self.window = window
        self.threshold = threshold
        self.min_bars_required = window + 2

    def _decide(self, symbol, bars):
        aroon = ta.trend.AroonIndicator(bars["high"], bars["low"], window=self.window)
        osc = aroon.aroon_indicator()

        df = pd.DataFrame({"osc": osc}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_osc = df["osc"].iloc[-2]
        curr_osc = df["osc"].iloc[-1]
        details = {"aroon_osc": curr_osc}

        bull_cross = prev_osc < self.threshold and curr_osc >= self.threshold
        bear_cross = prev_osc > -self.threshold and curr_osc <= -self.threshold

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Stochastic Reversion
# ─────────────────────────────────────────────────────────────────────────────
class StochasticReversionStrategy(Strategy):
    """Buy when %K crosses above %D while both are in the oversold zone (<20),
    sell when %K crosses below %D while both are in the overbought zone (>80).

    The two-condition requirement (zone + crossover) reduces false signals
    vs. acting on the zone-touch alone.
    """

    name = "Stochastic_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, window=14, smooth=3, oversold=20, overbought=80):
        self.window = window
        self.smooth = smooth
        self.oversold = oversold
        self.overbought = overbought
        self.min_bars_required = window + smooth + 2

    def _decide(self, symbol, bars):
        k = ta.momentum.stoch(bars["high"], bars["low"], bars["close"],
                              window=self.window, smooth_window=self.smooth)
        d = ta.momentum.stoch_signal(bars["high"], bars["low"], bars["close"],
                                     window=self.window, smooth_window=self.smooth)

        df = pd.DataFrame({"k": k, "d": d}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"stoch_k": latest["k"], "stoch_d": latest["d"]}

        bull = (latest["k"] < self.oversold and latest["d"] < self.oversold
                and prev["k"] < prev["d"] and latest["k"] >= latest["d"])
        bear = (latest["k"] > self.overbought and latest["d"] > self.overbought
                and prev["k"] > prev["d"] and latest["k"] <= latest["d"])

        if bull:
            return Signal(BUY, self.name, details)
        if bear:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Williams %R
# ─────────────────────────────────────────────────────────────────────────────
class WilliamsRStrategy(Strategy):
    """Williams %R oscillates between -100 (most oversold) and 0 (most overbought).

    Buy signal: %R rises above -80 (escaping oversold territory).
    Sell signal: %R falls below -20 (escaping overbought territory).
    The escape-from-zone logic avoids chasing entries deep inside the zone.
    """

    name = "Williams_R"
    strategy_type = "mean_reversion"

    def __init__(self, lbp=14):
        self.lbp = lbp
        self.min_bars_required = lbp + 2

    def _decide(self, symbol, bars):
        wr = ta.momentum.williams_r(bars["high"], bars["low"], bars["close"], lbp=self.lbp)

        df = pd.DataFrame({"wr": wr}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_wr = df["wr"].iloc[-2]
        curr_wr = df["wr"].iloc[-1]
        details = {"williams_r": curr_wr}

        escaped_oversold = prev_wr <= -80 and curr_wr > -80
        escaped_overbought = prev_wr >= -20 and curr_wr < -20

        if escaped_oversold:
            return Signal(BUY, self.name, details)
        if escaped_overbought:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 13. CCI Reversion
# ─────────────────────────────────────────────────────────────────────────────
class CciReversionStrategy(Strategy):
    """CCI (Commodity Channel Index) measures how far price deviates from its
    statistical mean.  Typical range: -100 to +100; excursions beyond those
    levels suggest overextension and tend to revert.

    Buy: CCI crosses back above -100 (exits oversold).
    Sell: CCI crosses back below +100 (exits overbought).
    """

    name = "CCI_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, window=20):
        self.window = window
        self.min_bars_required = window + 2

    def _decide(self, symbol, bars):
        cci = ta.trend.cci(bars["high"], bars["low"], bars["close"], window=self.window)

        df = pd.DataFrame({"cci": cci}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_cci = df["cci"].iloc[-2]
        curr_cci = df["cci"].iloc[-1]
        details = {"cci": curr_cci}

        exit_oversold = prev_cci <= -100 and curr_cci > -100
        exit_overbought = prev_cci >= 100 and curr_cci < 100

        if exit_oversold:
            return Signal(BUY, self.name, details)
        if exit_overbought:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Keltner Channel Reversion
# ─────────────────────────────────────────────────────────────────────────────
class KeltnerReversionStrategy(Strategy):
    """Similar concept to Bollinger Bands but uses ATR instead of standard
    deviation for band width, making it more robust during volatility spikes.

    Buy when price closes below the lower Keltner channel (statistical undershot).
    Sell when price closes above the upper channel (overshot).
    """

    name = "Keltner_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, window=20, window_atr=10):
        self.window = window
        self.window_atr = window_atr
        self.min_bars_required = max(window, window_atr) + 1

    def _decide(self, symbol, bars):
        kc = ta.volatility.KeltnerChannel(
            bars["high"], bars["low"], bars["close"],
            window=self.window, window_atr=self.window_atr
        )
        upper = kc.keltner_channel_hband()
        lower = kc.keltner_channel_lband()

        df = pd.DataFrame({
            "close": bars["close"], "upper": upper, "lower": lower
        }).dropna()
        if df.empty:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        latest = df.iloc[-1]
        details = {"close": latest["close"], "upper": latest["upper"], "lower": latest["lower"]}

        if latest["close"] <= latest["lower"]:
            return Signal(BUY, self.name, details)
        if latest["close"] >= latest["upper"]:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Z-Score Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────
class ZscoreReversionStrategy(Strategy):
    """Z-score = (price - rolling_mean) / rolling_std.

    A z-score beyond ±2 means price is more than 2 standard deviations from
    its recent mean — statistically rare and likely to revert.

    Buy when z-score crosses above -2 from below (starts reverting upward).
    Sell when z-score crosses below +2 from above (starts reverting downward).
    """

    name = "Zscore_Reversion"
    strategy_type = "mean_reversion"

    def __init__(self, window=20, threshold=2.0):
        self.window = window
        self.threshold = threshold
        self.min_bars_required = window + 2

    def _decide(self, symbol, bars):
        close = bars["close"]
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        zscore = (close - mean) / std

        df = pd.DataFrame({"zscore": zscore}).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev_z = df["zscore"].iloc[-2]
        curr_z = df["zscore"].iloc[-1]
        details = {"zscore": curr_z}

        exit_oversold = prev_z <= -self.threshold and curr_z > -self.threshold
        exit_overbought = prev_z >= self.threshold and curr_z < self.threshold

        if exit_oversold:
            return Signal(BUY, self.name, details)
        if exit_overbought:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 16. ATR Channel Breakout
# ─────────────────────────────────────────────────────────────────────────────
class AtrChannelBreakoutStrategy(Strategy):
    """Draws a channel around a slow EMA using ATR width.
    Breakout above the upper channel = momentum buy.
    Breakdown below the lower channel = short/sell.

    Concept: combining a trend anchor (EMA) with a volatility-scaled band
    (ATR) means the channel adapts — calmer stocks get tighter breakout thresholds,
    wilder stocks need bigger moves to confirm a real breakout.
    """

    name = "ATR_Channel_Breakout"
    strategy_type = "trend"

    def __init__(self, ema_window=20, atr_window=14, multiplier=1.5):
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.multiplier = multiplier
        self.min_bars_required = max(ema_window, atr_window) + 2

    def _decide(self, symbol, bars):
        ema = ta.trend.ema_indicator(bars["close"], window=self.ema_window)
        atr = ta.volatility.average_true_range(bars["high"], bars["low"], bars["close"], window=self.atr_window)

        upper = ema + self.multiplier * atr
        lower = ema - self.multiplier * atr

        df = pd.DataFrame({
            "close": bars["close"], "upper": upper, "lower": lower
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"close": latest["close"], "upper": latest["upper"], "lower": latest["lower"]}

        broke_up = prev["close"] < prev["upper"] and latest["close"] >= latest["upper"]
        broke_dn = prev["close"] > prev["lower"] and latest["close"] <= latest["lower"]

        if broke_up:
            return Signal(BUY, self.name, details)
        if broke_dn:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Volatility Squeeze Breakout
# ─────────────────────────────────────────────────────────────────────────────
class VolatilitySqueezeStrategy(Strategy):
    """The "squeeze" occurs when Bollinger Bands narrow inside the Keltner Channel,
    meaning volatility has compressed. When BBs expand back outside the KC,
    a directional breakout (often explosive) begins.

    Buy if BBs break out of squeeze to the upside (momentum positive).
    Sell if BBs break out to the downside.
    """

    name = "Volatility_Squeeze"
    strategy_type = "trend"

    def __init__(self, bb_window=20, kc_window=20, kc_atr_window=10):
        self.bb_window = bb_window
        self.kc_window = kc_window
        self.kc_atr_window = kc_atr_window
        self.min_bars_required = max(bb_window, kc_window) + 2

    def _decide(self, symbol, bars):
        bb = ta.volatility.BollingerBands(bars["close"], window=self.bb_window)
        kc = ta.volatility.KeltnerChannel(
            bars["high"], bars["low"], bars["close"],
            window=self.kc_window, window_atr=self.kc_atr_window
        )

        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()

        # Squeeze = BB upper < KC upper AND BB lower > KC lower
        in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        momentum = ta.trend.sma_indicator(bars["close"], window=5) - ta.trend.sma_indicator(bars["close"], window=34)

        df = pd.DataFrame({
            "squeeze": in_squeeze, "momentum": momentum, "close": bars["close"]
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"in_squeeze": bool(latest["squeeze"]), "momentum": latest["momentum"]}

        # Fire when squeeze releases (was squeezed, now not)
        squeeze_released = prev["squeeze"] and not latest["squeeze"]

        if squeeze_released and latest["momentum"] > 0:
            return Signal(BUY, self.name, details)
        if squeeze_released and latest["momentum"] < 0:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 18. MACD + RSI Combo
# ─────────────────────────────────────────────────────────────────────────────
class MacdRsiComboStrategy(Strategy):
    """Two-factor confirmation: require BOTH a MACD crossover AND RSI in a
    neutral-to-favorable zone before entering.

    MACD tells us *momentum direction is changing*.
    RSI guard prevents buying into already-overbought conditions (RSI < 70)
    or shorting already-oversold (RSI > 30).
    Dual confirmation reduces false signals at the cost of fewer trades.
    """

    name = "MACD_RSI_Combo"
    strategy_type = "trend"

    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14,
                 rsi_overbought=70, rsi_oversold=30):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_bars_required = macd_slow + macd_signal + rsi_period + 2

    def _decide(self, symbol, bars):
        macd = ta.trend.MACD(bars["close"],
                             window_fast=self.macd_fast,
                             window_slow=self.macd_slow,
                             window_sign=self.macd_signal)
        rsi = ta.momentum.rsi(bars["close"], window=self.rsi_period)

        df = pd.DataFrame({
            "macd": macd.macd(),
            "signal": macd.macd_signal(),
            "rsi": rsi,
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"macd": latest["macd"], "signal": latest["signal"], "rsi": latest["rsi"]}

        macd_bull = prev["macd"] < prev["signal"] and latest["macd"] >= latest["signal"]
        macd_bear = prev["macd"] > prev["signal"] and latest["macd"] <= latest["signal"]

        if macd_bull and latest["rsi"] < self.rsi_overbought:
            return Signal(BUY, self.name, details)
        if macd_bear and latest["rsi"] > self.rsi_oversold:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 19. EMA Crossover + Volume Surge
# ─────────────────────────────────────────────────────────────────────────────
class EmaVolumeStrategy(Strategy):
    """EMA crossover gated by a volume surge above its rolling average.

    The core idea: a price crossover backed by above-average volume is more
    likely to be real institutional conviction than a low-volume fake-out.
    If volume is weak, skip the trade even if the crossover is clean.
    """

    name = "EMA_Volume"
    strategy_type = "trend"

    def __init__(self, fast=9, slow=21, vol_window=20, vol_multiplier=1.2):
        self.fast = fast
        self.slow = slow
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier
        self.min_bars_required = max(slow, vol_window) + 2

    def _decide(self, symbol, bars):
        fast_ema = ta.trend.ema_indicator(bars["close"], window=self.fast)
        slow_ema = ta.trend.ema_indicator(bars["close"], window=self.slow)
        avg_vol = bars["volume"].rolling(self.vol_window).mean()

        df = pd.DataFrame({
            "fast": fast_ema, "slow": slow_ema,
            "volume": bars["volume"], "avg_vol": avg_vol
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        high_volume = latest["volume"] >= latest["avg_vol"] * self.vol_multiplier
        details = {"fast_ema": latest["fast"], "slow_ema": latest["slow"],
                   "volume": latest["volume"], "avg_vol": latest["avg_vol"], "high_vol": high_volume}

        bull_cross = prev["fast"] < prev["slow"] and latest["fast"] >= latest["slow"]
        bear_cross = prev["fast"] > prev["slow"] and latest["fast"] <= latest["slow"]

        if bull_cross and high_volume:
            return Signal(BUY, self.name, details)
        if bear_cross and high_volume:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# 20. ADX + MACD Combo
# ─────────────────────────────────────────────────────────────────────────────
class AdxMacdComboStrategy(Strategy):
    """MACD crossover gated by ADX trend strength.

    MACD is good at detecting momentum direction changes, but it whipsaws badly
    in flat markets. Adding an ADX filter (only trade when ADX > threshold =
    trend is real) cuts the false signals in sideways conditions significantly.
    """

    name = "ADX_MACD_Combo"
    strategy_type = "trend"

    def __init__(self, adx_window=14, adx_threshold=20,
                 macd_fast=12, macd_slow=26, macd_signal=9):
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.min_bars_required = max(adx_window * 2, macd_slow + macd_signal) + 2

    def _decide(self, symbol, bars):
        adx = ta.trend.adx(bars["high"], bars["low"], bars["close"], window=self.adx_window)
        macd_obj = ta.trend.MACD(bars["close"],
                                 window_fast=self.macd_fast,
                                 window_slow=self.macd_slow,
                                 window_sign=self.macd_signal)

        df = pd.DataFrame({
            "adx": adx,
            "macd": macd_obj.macd(),
            "signal": macd_obj.macd_signal(),
        }).dropna()
        if len(df) < 2:
            return Signal(HOLD, self.name, {"reason": "not_enough_data"})

        prev, latest = df.iloc[-2], df.iloc[-1]
        details = {"adx": latest["adx"], "macd": latest["macd"], "signal": latest["signal"]}

        if latest["adx"] < self.adx_threshold:
            return Signal(HOLD, self.name, {**details, "reason": "no_trend"})

        bull_cross = prev["macd"] < prev["signal"] and latest["macd"] >= latest["signal"]
        bear_cross = prev["macd"] > prev["signal"] and latest["macd"] <= latest["signal"]

        if bull_cross:
            return Signal(BUY, self.name, details)
        if bear_cross:
            return Signal(SELL, self.name, details)
        return Signal(HOLD, self.name, details)


# ─────────────────────────────────────────────────────────────────────────────
# Registry: maps strategy name → (instance, timeframe)
# ─────────────────────────────────────────────────────────────────────────────
ALL_STRATEGIES = [
    (EmaCrossoverStrategy(),              "daily"),
    (TripleEmaCrossoverStrategy(),        "daily"),
    (DonchianBreakoutStrategy(),          "daily"),
    (AdxDiStrategy(),                     "daily"),
    (SupertrendStrategy(),                "daily"),
    (ParabolicSarStrategy(),              "daily"),
    (RocMomentumStrategy(),               "daily"),
    (TrixStrategy(),                      "daily"),
    (MomentumBreakoutStrategy(),          "daily"),
    (AroonOscillatorStrategy(),           "daily"),
    (StochasticReversionStrategy(),       "hourly"),
    (WilliamsRStrategy(),                 "hourly"),
    (CciReversionStrategy(),              "daily"),
    (KeltnerReversionStrategy(),          "daily"),
    (ZscoreReversionStrategy(),           "daily"),
    (AtrChannelBreakoutStrategy(),        "daily"),
    (VolatilitySqueezeStrategy(),         "daily"),
    (MacdRsiComboStrategy(),              "daily"),
    (EmaVolumeStrategy(),                 "daily"),
    (AdxMacdComboStrategy(),              "daily"),
]
