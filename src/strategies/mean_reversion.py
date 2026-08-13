# src/strategies/mean_reversion.py
# Mean Reversion Strategy — buys the dip, sells the rip.
#
# How it works:
# - Price always "reverts to the mean" (average price) eventually
# - When price hits extreme levels (Bollinger Band edges), it bounces
# - Uses RSI extremes to confirm the move is overextended
# - Volume spike on the extreme = capitulation = high-probability reversal
#
# Best in: ranging markets with clear support/resistance
# Worst in: strong trends (keeps buying a falling knife)
#
# Safety: only trades high-liquidity pairs (BTC, ETH) where
# mean reversion is most reliable.

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI extreme mean reversion strategy."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Generate signal when price hits Bollinger Band extremes + RSI extreme."""
        if len(df) < 30:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest["close"]
        rsi = latest.get("rsi")
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        bb_middle = latest.get("bb_middle")
        bb_width = latest.get("bb_width")
        vol_ratio = latest.get("volume_ratio")
        atr = latest.get("atr")
        macd_hist = latest.get("macd_histogram")
        prev_macd_hist = prev.get("macd_histogram")

        if any(pd.isna(v) for v in [rsi, bb_upper, bb_lower, bb_middle, atr]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # Config
        rsi_extreme_low = config.get("rsi_extreme_low", 25)
        rsi_extreme_high = config.get("rsi_extreme_high", 75)

        reasons = []
        buy_score = 0
        sell_score = 0

        # Skip if Bollinger Bands are too wide (strong trend, mean reversion fails)
        if not pd.isna(bb_width) and bb_width > 0.08:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=[f"BB too wide ({bb_width:.3f}), trend too strong"])

        # --- BUY signals: price at/below lower Bollinger Band ---
        if close <= bb_lower:
            buy_score += 2  # strong mean reversion signal
            reasons.append(f"price at lower BB ({close:.2f} <= {bb_lower:.2f})")

        # RSI extreme oversold
        if rsi <= rsi_extreme_low:
            buy_score += 2
            reasons.append(f"RSI extreme oversold ({rsi:.0f} <= {rsi_extreme_low})")
        elif rsi < 35:
            buy_score += 1
            reasons.append(f"RSI oversold ({rsi:.0f})")

        # Volume spike on dip = capitulation (smart money buying)
        if not pd.isna(vol_ratio) and vol_ratio >= 1.8 and buy_score > 0:
            buy_score += 1
            reasons.append(f"volume capitulation ({vol_ratio:.1f}x)")

        # MACD starting to turn up from negative
        if not pd.isna(prev_macd_hist) and prev_macd_hist < macd_hist and macd_hist < 0:
            buy_score += 1
            reasons.append("MACD turning up (momentum shift)")

        # --- SELL signals: price at/above upper Bollinger Band ---
        if close >= bb_upper:
            sell_score += 2
            reasons.append(f"price at upper BB ({close:.2f} >= {bb_upper:.2f})")

        if rsi >= rsi_extreme_high:
            sell_score += 2
            reasons.append(f"RSI extreme overbought ({rsi:.0f} >= {rsi_extreme_high})")
        elif rsi > 65:
            sell_score += 1
            reasons.append(f"RSI overbought ({rsi:.0f})")

        if not pd.isna(vol_ratio) and vol_ratio >= 1.8 and sell_score > 0:
            sell_score += 1
            reasons.append(f"volume blow-off ({vol_ratio:.1f}x)")

        if not pd.isna(prev_macd_hist) and prev_macd_hist > macd_hist and macd_hist > 0:
            sell_score += 1
            reasons.append("MACD turning down (momentum shift)")

        # Need 3+ points — BB touch alone is not enough
        min_score = 3

        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / 6, 1.0)
            # Mean reversion TP = middle Bollinger Band (the "mean")
            tp = bb_middle
            # SL = one ATR beyond the lower BB
            sl = bb_lower - atr
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=close,
                stop_loss=round(sl, 8),
                take_profit=round(tp, 8),
                strategy_name=self.name, reasons=reasons,
            )

        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / 6, 1.0)
            tp = bb_middle
            sl = bb_upper + atr
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=close,
                stop_loss=round(sl, 8),
                take_profit=round(tp, 8),
                strategy_name=self.name, reasons=reasons,
            )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=reasons if reasons else ["no extreme detected"])
