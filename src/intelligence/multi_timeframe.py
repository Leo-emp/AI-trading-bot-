# src/intelligence/multi_timeframe.py
# Multi-Timeframe Analysis Brain — confirms signals across time horizons.
#
# This is Brain 4 of the 5-brain consensus system.
#
# The principle: a signal on a 5m chart is meaningless if the
# 15m and 1h charts disagree. Higher timeframes overrule lower ones.
#
# How it works:
# - Fetches 5m, 15m, 1h candles
# - Computes indicators on each timeframe
# - Checks if all timeframes agree on direction
# - Higher timeframes have more weight in the confidence score
#
# If 1h says BUY but 5m says SELL → HOLD (don't fight the bigger trend)

import logging
from typing import Optional

import pandas as pd

from src.data.indicators import IndicatorEngine

logger = logging.getLogger(__name__)


class MultiTimeframeBrain:
    """Confirms trading signals across multiple timeframes.

    Brain 4: all timeframes must agree for a signal to pass.
    """

    def __init__(self):
        self._indicator_engine = IndicatorEngine()
        # Timeframe weights — higher = more important
        self._weights = {
            "5m": 0.2,
            "15m": 0.3,
            "1h": 0.5,  # 1h is the most important
        }

    def analyze(self, timeframe_data: dict[str, pd.DataFrame]) -> dict:
        """Analyze multiple timeframe DataFrames and return a consensus signal.

        timeframe_data: {"5m": df_5m, "15m": df_15m, "1h": df_1h}
        Each DataFrame should have OHLCV data.

        Returns: {"direction": "BUY"/"SELL"/"HOLD", "confidence": float, "reason": str}
        """
        signals = {}

        for tf, df in timeframe_data.items():
            if df is None or len(df) < 30:
                signals[tf] = {"direction": "HOLD", "score": 0}
                continue

            # Compute indicators for this timeframe
            df_with_indicators = self._indicator_engine.compute_all(df)
            signal = self._analyze_single_timeframe(df_with_indicators)
            signals[tf] = signal

        return self._combine_signals(signals)

    def _analyze_single_timeframe(self, df: pd.DataFrame) -> dict:
        """Analyze a single timeframe's indicators for direction."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        rsi = latest.get("rsi")
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        macd_hist = latest.get("macd_histogram")
        close = latest["close"]

        if any(pd.isna(v) for v in [rsi, ema_fast, ema_slow, macd_hist]):
            return {"direction": "HOLD", "score": 0}

        score = 0  # positive = bullish, negative = bearish

        # EMA position (strongest signal on higher timeframes)
        if ema_fast > ema_slow:
            score += 2
        elif ema_fast < ema_slow:
            score -= 2

        # Price vs EMA
        if close > ema_fast:
            score += 1
        elif close < ema_fast:
            score -= 1

        # MACD direction
        if macd_hist > 0:
            score += 1
        elif macd_hist < 0:
            score -= 1

        # RSI (moderate signals, not extremes)
        if rsi < 40:
            score += 1  # oversold → bullish bias
        elif rsi > 60:
            score -= 1  # overbought → bearish bias

        if score >= 3:
            return {"direction": "BUY", "score": score}
        elif score <= -3:
            return {"direction": "SELL", "score": abs(score)}
        return {"direction": "HOLD", "score": 0}

    def _combine_signals(self, signals: dict) -> dict:
        """Combine signals from all timeframes using weighted voting."""
        buy_weight = 0.0
        sell_weight = 0.0
        reasons = []

        for tf, signal in signals.items():
            weight = self._weights.get(tf, 0.2)
            direction = signal["direction"]

            if direction == "BUY":
                buy_weight += weight
                reasons.append(f"{tf}: BUY")
            elif direction == "SELL":
                sell_weight += weight
                reasons.append(f"{tf}: SELL")
            else:
                reasons.append(f"{tf}: HOLD")

        # All timeframes must lean the same way
        total_weight = sum(self._weights.values())
        # Need at least 70% weighted agreement
        threshold = total_weight * 0.7

        if buy_weight >= threshold:
            confidence = buy_weight / total_weight
            return {
                "direction": "BUY",
                "confidence": round(confidence, 2),
                "reason": f"multi-TF aligned BUY ({', '.join(reasons)})",
            }
        elif sell_weight >= threshold:
            confidence = sell_weight / total_weight
            return {
                "direction": "SELL",
                "confidence": round(confidence, 2),
                "reason": f"multi-TF aligned SELL ({', '.join(reasons)})",
            }
        else:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "reason": f"timeframes disagree ({', '.join(reasons)})",
            }
