# src/strategies/momentum.py
# Momentum / Trend Following Strategy — rides strong moves.
#
# How it works:
# - Detects trends using EMA crossover (fast EMA > slow EMA = uptrend)
# - Confirms with RSI in healthy range (not overbought/oversold extremes)
# - Requires volume confirmation (real money behind the move)
# - Uses wider stops to give trends room to breathe
# - Trailing stop locks in profits as trend extends
#
# Best in: trending markets (up or down)
# Worst in: choppy/ranging markets (whipsaws kill it)
#
# Key difference from smart_scalp:
# - Wider TP/SL (lets winners run longer)
# - Trailing stop instead of fixed TP
# - EMA trend as primary signal (not RSI extremes)

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class MomentumStrategy(BaseStrategy):
    """Trend-following strategy using EMA crossover + RSI + volume."""

    @property
    def name(self) -> str:
        return "momentum"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Generate signal when a strong trend is confirmed."""
        if len(df) < 30:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest["close"]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        prev_ema_fast = prev.get("ema_fast")
        prev_ema_slow = prev.get("ema_slow")
        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        atr = latest.get("atr")

        if any(pd.isna(v) for v in [ema_fast, ema_slow, prev_ema_fast,
                                     prev_ema_slow, rsi, macd_hist, atr]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # Config
        rsi_min = config.get("rsi_min", 30)
        rsi_max = config.get("rsi_max", 70)
        trailing_stop_pct = config.get("trailing_stop_pct", 1.5) / 100

        reasons = []
        buy_score = 0
        sell_score = 0

        # Signal 1: EMA crossover (strongest trend signal)
        ema_bullish_cross = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
        ema_bearish_cross = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
        ema_bullish_trend = ema_fast > ema_slow
        ema_bearish_trend = ema_fast < ema_slow

        if ema_bullish_cross:
            buy_score += 2  # crossover is worth 2 points
            reasons.append("EMA bullish crossover (strong)")
        elif ema_bullish_trend:
            buy_score += 1
            reasons.append("EMA bullish trend")

        if ema_bearish_cross:
            sell_score += 2
            reasons.append("EMA bearish crossover (strong)")
        elif ema_bearish_trend:
            sell_score += 1
            reasons.append("EMA bearish trend")

        # Signal 2: RSI in healthy trend range (not exhausted)
        if rsi_min < rsi < 60:
            buy_score += 1
            reasons.append(f"RSI healthy for buy ({rsi:.0f})")
        elif 40 < rsi < rsi_max:
            sell_score += 1
            reasons.append(f"RSI healthy for sell ({rsi:.0f})")

        # Signal 3: MACD histogram confirms direction
        if macd_hist > 0:
            buy_score += 1
            reasons.append(f"MACD positive ({macd_hist:.4f})")
        elif macd_hist < 0:
            sell_score += 1
            reasons.append(f"MACD negative ({macd_hist:.4f})")

        # Signal 4: Volume confirmation
        if not pd.isna(vol_ratio) and vol_ratio >= 1.3:
            if buy_score > sell_score:
                buy_score += 1
            elif sell_score > buy_score:
                sell_score += 1
            reasons.append(f"volume confirms ({vol_ratio:.1f}x)")

        # Signal 5: Price above/below both EMAs (trend alignment)
        if close > ema_fast and close > ema_slow:
            buy_score += 1
            reasons.append("price above both EMAs")
        elif close < ema_fast and close < ema_slow:
            sell_score += 1
            reasons.append("price below both EMAs")

        # Need 4+ points for a trade (strong trend confirmation)
        min_score = 4

        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / 7, 1.0)
            # Wider stops for trend trades — ATR-based
            sl = close - (atr * 2.5)  # 2.5x ATR stop loss
            tp = close + (atr * 4.0)  # 4x ATR take profit (better R:R)
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=close,
                stop_loss=round(sl, 8),
                take_profit=round(tp, 8),
                strategy_name=self.name, reasons=reasons,
            )

        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / 7, 1.0)
            sl = close + (atr * 2.5)
            tp = close - (atr * 4.0)
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=close,
                stop_loss=round(sl, 8),
                take_profit=round(tp, 8),
                strategy_name=self.name, reasons=reasons,
            )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=reasons if reasons else ["no trend confirmed"])
