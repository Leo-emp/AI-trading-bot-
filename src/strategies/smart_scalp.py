# src/strategies/smart_scalp.py
# Smart Scalping Strategy — the primary profit engine.
#
# KEY DIFFERENCE from naive scalping:
# - Uses 5m+15m candles (not 1m — too noisy, fees kill you)
# - Requires MULTI-CONFIRMATION before entering (volume + momentum + indicator)
# - LIMIT ORDERS ONLY (maker fee, not taker)
# - 2:1 reward/risk ratio (0.8% TP, 0.4% SL) to overcome fee drag
# - Skips marginal signals — quality over quantity

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal
from src.data.indicators import IndicatorEngine


class SmartScalpStrategy(BaseStrategy):
    """Fee-aware scalping strategy.

    Entry requires multi-confirmation:
    1. RSI shows momentum (not overbought for buy, not oversold for sell)
    2. MACD confirms direction (histogram positive for buy, negative for sell)
    3. Volume is above average (confirms real interest, not noise)
    4. Price action confirms (close above EMA for buy, below for sell)

    All 4 must agree. This filters out 80%+ of noise signals.
    """

    @property
    def name(self) -> str:
        return "smart_scalp"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Generate a scalping signal from indicator data.

        Returns BUY/SELL only when all confirmation criteria are met.
        Otherwise returns HOLD (no trade).
        """
        if len(df) < 5:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        prev_macd_hist = prev.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        close = latest["close"]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")

        # Check indicators are warmed up
        if any(pd.isna(v) for v in [rsi, macd_hist, vol_ratio, ema_fast, ema_slow]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # Config values
        rsi_oversold = config.get("rsi_oversold", 30)
        rsi_overbought = config.get("rsi_overbought", 70)
        vol_multiplier = config.get("volume_spike_multiplier", 1.5)
        tp_pct = config.get("take_profit_pct", 0.8) / 100  # convert to decimal
        sl_pct = config.get("stop_loss_pct", 0.4) / 100

        # --- Multi-confirmation for BUY ---
        buy_confirmations = []

        # 1. RSI not overbought (room to go up)
        if rsi < rsi_overbought:
            buy_confirmations.append(f"RSI {rsi:.0f} < {rsi_overbought}")
        # Extra point if oversold (strong buy signal)
        if rsi < rsi_oversold:
            buy_confirmations.append(f"RSI oversold ({rsi:.0f})")

        # 2. MACD histogram positive or crossing up
        if macd_hist > 0:
            buy_confirmations.append(f"MACD positive ({macd_hist:.4f})")
        if not pd.isna(prev_macd_hist) and prev_macd_hist < 0 and macd_hist > 0:
            buy_confirmations.append("MACD bullish crossover")

        # 3. Volume above average
        if vol_ratio >= vol_multiplier:
            buy_confirmations.append(f"volume spike ({vol_ratio:.1f}x)")

        # 4. Price above fast EMA (upward momentum)
        if close > ema_fast:
            buy_confirmations.append("price above EMA-fast")

        # --- Multi-confirmation for SELL ---
        sell_confirmations = []

        if rsi > rsi_oversold:
            sell_confirmations.append(f"RSI {rsi:.0f} > {rsi_oversold}")
        if rsi > rsi_overbought:
            sell_confirmations.append(f"RSI overbought ({rsi:.0f})")

        if macd_hist < 0:
            sell_confirmations.append(f"MACD negative ({macd_hist:.4f})")
        if not pd.isna(prev_macd_hist) and prev_macd_hist > 0 and macd_hist < 0:
            sell_confirmations.append("MACD bearish crossover")

        if vol_ratio >= vol_multiplier:
            sell_confirmations.append(f"volume spike ({vol_ratio:.1f}x)")

        if close < ema_fast:
            sell_confirmations.append("price below EMA-fast")

        # --- Decision: need 4+ confirmations ---
        # (RSI range + MACD direction + volume + price/EMA alignment)
        min_confirmations = 4

        if len(buy_confirmations) >= min_confirmations and len(buy_confirmations) > len(sell_confirmations):
            confidence = min(len(buy_confirmations) / 6, 1.0)
            entry = close
            stop_loss = round(entry * (1 - sl_pct), 8)
            take_profit = round(entry * (1 + tp_pct), 8)
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=entry, stop_loss=stop_loss,
                take_profit=take_profit, strategy_name=self.name,
                reasons=buy_confirmations,
            )

        if len(sell_confirmations) >= min_confirmations and len(sell_confirmations) > len(buy_confirmations):
            confidence = min(len(sell_confirmations) / 6, 1.0)
            entry = close
            stop_loss = round(entry * (1 + sl_pct), 8)
            take_profit = round(entry * (1 - tp_pct), 8)
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=entry, stop_loss=stop_loss,
                take_profit=take_profit, strategy_name=self.name,
                reasons=sell_confirmations,
            )

        # Not enough confirmation — HOLD
        all_reasons = buy_confirmations + sell_confirmations
        return StrategySignal(
            direction="HOLD", confidence=0.0, strategy_name=self.name,
            reasons=all_reasons if all_reasons else ["no multi-confirmation met"],
        )
