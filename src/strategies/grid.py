# src/strategies/grid.py
# Grid Trading Strategy — the sideways market earner.
#
# How it works:
# - Places buy orders at intervals BELOW current price
# - Places sell orders at intervals ABOVE current price
# - When price bounces between levels, it catches profits
# - ATR-based spacing adapts to current volatility
#
# Best in: ranging/sideways markets (60-70% of the time)
# Worst in: strong trends (one side keeps getting hit)
#
# Grid spacing is dynamic:
# - High volatility → wider grid (ATR * multiplier)
# - Low volatility → tighter grid (minimum spacing)

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class GridStrategy(BaseStrategy):
    """ATR-based dynamic grid trading strategy.

    Generates BUY signals when price drops to a grid level below,
    SELL signals when price rises to a grid level above.
    Grid spacing adapts to volatility using ATR.
    """

    @property
    def name(self) -> str:
        return "grid"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Check if price has hit a grid level."""
        if len(df) < 30:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = latest["close"]
        prev_close = prev["close"]
        atr = latest.get("atr")
        rsi = latest.get("rsi")
        bb_width = latest.get("bb_width")

        if pd.isna(atr) or pd.isna(rsi) or atr <= 0:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # Config
        levels = config.get("levels", 5)
        spacing_mult = config.get("spacing_atr_multiplier", 0.5)
        min_spacing_pct = config.get("min_spacing_pct", 0.3) / 100
        max_spacing_pct = config.get("max_spacing_pct", 2.0) / 100

        # Calculate grid spacing based on ATR
        spacing = atr * spacing_mult
        spacing_pct = spacing / close if close > 0 else 0

        # Clamp spacing to configured range
        spacing_pct = max(min_spacing_pct, min(spacing_pct, max_spacing_pct))
        spacing = close * spacing_pct

        # Only trade in ranging markets (BB width narrow = low volatility)
        # Skip grid in strong trends
        if not pd.isna(bb_width) and bb_width > 0.06:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=[f"BB too wide ({bb_width:.3f}), market trending"])

        reasons = []

        # BUY: price dropped through a grid level
        # Price crossed below a grid line from above
        price_drop = prev_close - close
        if price_drop >= spacing * 0.8:
            # Confirm with RSI — don't buy into freefall
            if rsi < 65:
                confidence = 0.55
                # Stronger signal if RSI is also oversold
                if rsi < 35:
                    confidence = 0.75
                    reasons.append(f"RSI oversold ({rsi:.0f})")
                reasons.append(f"price dropped through grid ({price_drop:.2f} >= {spacing:.2f})")
                reasons.append(f"grid spacing: {spacing_pct*100:.2f}% (ATR: {atr:.2f})")

                sl_pct = spacing_pct * 2  # stop at 2 grid levels below
                tp_pct = spacing_pct * 1.5  # TP at 1.5 grid levels above
                return StrategySignal(
                    direction="BUY", confidence=confidence,
                    entry_price=close,
                    stop_loss=round(close * (1 - sl_pct), 8),
                    take_profit=round(close * (1 + tp_pct), 8),
                    strategy_name=self.name, reasons=reasons,
                )

        # SELL: price rose through a grid level
        price_rise = close - prev_close
        if price_rise >= spacing * 0.8:
            if rsi > 35:
                confidence = 0.55
                if rsi > 65:
                    confidence = 0.75
                    reasons.append(f"RSI overbought ({rsi:.0f})")
                reasons.append(f"price rose through grid ({price_rise:.2f} >= {spacing:.2f})")
                reasons.append(f"grid spacing: {spacing_pct*100:.2f}% (ATR: {atr:.2f})")

                sl_pct = spacing_pct * 2
                tp_pct = spacing_pct * 1.5
                return StrategySignal(
                    direction="SELL", confidence=confidence,
                    entry_price=close,
                    stop_loss=round(close * (1 + sl_pct), 8),
                    take_profit=round(close * (1 - tp_pct), 8),
                    strategy_name=self.name, reasons=reasons,
                )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=["price between grid levels"])
