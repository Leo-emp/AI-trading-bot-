# src/intelligence/market_regime.py
# Market Regime Detector — knows WHAT kind of market we're in.
#
# Markets have 4 regimes. Each regime needs a different strategy:
# - TRENDING_UP   → momentum strategy (ride the wave)
# - TRENDING_DOWN → momentum strategy (short the drop)
# - RANGING       → grid + mean reversion (buy low, sell high)
# - VOLATILE      → smart scalp with tight stops (quick in/out)
#
# Detection uses:
# - EMA slope: positive slope = uptrend, negative = downtrend
# - BB width: narrow = ranging, wide = volatile
# - ADX (via EMA separation): strong trend vs weak chop
# - RSI position: mid-range = no extreme, edges = possible reversal

import pandas as pd
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketRegime:
    """Current market regime classification."""
    regime: str          # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    confidence: float    # 0.0 to 1.0
    volatility: str      # LOW, NORMAL, HIGH
    trend_strength: float  # 0.0 (no trend) to 1.0 (very strong trend)
    reasons: list[str]


class MarketRegimeDetector:
    """Classifies the current market regime from indicator data.

    Called every cycle to determine which strategy should be active.
    The strategy selector uses this to pick the right approach.
    """

    def __init__(self):
        # BB width thresholds for volatility classification
        self._bb_narrow = 0.025   # below = low volatility (ranging)
        self._bb_wide = 0.055     # above = high volatility
        # EMA separation threshold for trend strength
        self._ema_trend_pct = 0.002  # 0.2% separation = trend exists

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Analyze recent price action and classify the market regime."""
        if len(df) < 30:
            return MarketRegime("RANGING", 0.3, "NORMAL", 0.0, ["insufficient data"])

        latest = df.iloc[-1]
        close = latest["close"]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        rsi = latest.get("rsi")
        bb_width = latest.get("bb_width")
        atr = latest.get("atr")

        if any(pd.isna(v) for v in [ema_fast, ema_slow, rsi, bb_width, atr]):
            return MarketRegime("RANGING", 0.3, "NORMAL", 0.0, ["indicators warming up"])

        reasons = []

        # --- Volatility classification ---
        if bb_width < self._bb_narrow:
            volatility = "LOW"
            reasons.append(f"low volatility (BB width: {bb_width:.3f})")
        elif bb_width > self._bb_wide:
            volatility = "HIGH"
            reasons.append(f"high volatility (BB width: {bb_width:.3f})")
        else:
            volatility = "NORMAL"
            reasons.append(f"normal volatility (BB width: {bb_width:.3f})")

        # --- Trend strength from EMA separation ---
        ema_sep = (ema_fast - ema_slow) / close if close > 0 else 0
        ema_sep_abs = abs(ema_sep)

        # Check trend over multiple candles for consistency
        trend_consistent = self._check_trend_consistency(df)

        if ema_sep > self._ema_trend_pct and trend_consistent > 0:
            trend_strength = min(ema_sep_abs / 0.01, 1.0)  # normalize
            reasons.append(f"EMA bullish (sep: {ema_sep*100:.3f}%)")

            if volatility == "HIGH":
                regime = "VOLATILE"
                confidence = 0.6
                reasons.append("trending up but volatile — scalp mode")
            else:
                regime = "TRENDING_UP"
                confidence = 0.5 + trend_strength * 0.3
                reasons.append(f"trend strength: {trend_strength:.2f}")

        elif ema_sep < -self._ema_trend_pct and trend_consistent < 0:
            trend_strength = min(ema_sep_abs / 0.01, 1.0)
            reasons.append(f"EMA bearish (sep: {ema_sep*100:.3f}%)")

            if volatility == "HIGH":
                regime = "VOLATILE"
                confidence = 0.6
                reasons.append("trending down but volatile — scalp mode")
            else:
                regime = "TRENDING_DOWN"
                confidence = 0.5 + trend_strength * 0.3
                reasons.append(f"trend strength: {trend_strength:.2f}")

        else:
            trend_strength = ema_sep_abs / 0.01
            if volatility == "HIGH":
                regime = "VOLATILE"
                confidence = 0.65
                reasons.append("no clear trend + high volatility")
            else:
                regime = "RANGING"
                confidence = 0.6
                reasons.append("no clear trend, EMAs close together")

        # RSI can boost confidence
        if regime in ("TRENDING_UP",) and 40 < rsi < 65:
            confidence = min(confidence + 0.1, 1.0)
            reasons.append(f"RSI confirms uptrend ({rsi:.0f})")
        elif regime in ("TRENDING_DOWN",) and 35 < rsi < 60:
            confidence = min(confidence + 0.1, 1.0)
            reasons.append(f"RSI confirms downtrend ({rsi:.0f})")

        logger.debug("Market regime: %s (confidence: %.2f)", regime, confidence)

        return MarketRegime(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            reasons=reasons,
        )

    def _check_trend_consistency(self, df: pd.DataFrame, lookback: int = 10) -> int:
        """Check if EMA trend has been consistent over last N candles.

        Returns: +1 (consistent uptrend), -1 (consistent downtrend), 0 (mixed)
        """
        if len(df) < lookback:
            return 0

        recent = df.tail(lookback)
        up_count = 0
        down_count = 0

        for _, row in recent.iterrows():
            ema_f = row.get("ema_fast")
            ema_s = row.get("ema_slow")
            if pd.isna(ema_f) or pd.isna(ema_s):
                continue
            if ema_f > ema_s:
                up_count += 1
            elif ema_f < ema_s:
                down_count += 1

        # 70%+ consistent = confirmed trend
        threshold = lookback * 0.7
        if up_count >= threshold:
            return 1
        elif down_count >= threshold:
            return -1
        return 0
