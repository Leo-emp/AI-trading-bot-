# src/strategies/grid.py
# Volatility Squeeze Strategy -- catches regime transitions.
#
# EVIDENCE (Bollinger Band Squeeze, widely documented):
#   When BB width narrows significantly (low volatility), price is
#   compressing -- building potential energy for a big move. When bands
#   expand and price breaks out, a new trend is starting.
#   John Bollinger himself recommends this as a primary strategy.
#
# HOW IT WORKS:
#   1. DETECT SQUEEZE: min BB width in last 10 candles < 85% of BB width SMA
#      (market was recently compressed, coiling for a move)
#   2. DETECT EXPANSION: current BB width > previous AND approaching SMA
#      (bands opening up = volatility returning = breakout beginning)
#   3. TRADE DIRECTION: price above upper BB = long, below lower BB = short
#   4. VOLUME + MACD + candle direction must confirm
#
# WHEN IT FAILS:
#   - False squeezes that re-compress (volume filter reduces these)
#   - Breakout that immediately reverses (tight SL limits damage)
#   - Very low volatility environments where squeeze never breaks
#
# SL: 0.5% | TP: 1.5% | R:R = 3:1 | Cooldown: 24 candles (2 hrs)
#
# Class name kept as GridStrategy for backward compatibility.
# Internally this is a BB squeeze breakout strategy.

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class GridStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        # Per-pair cooldown — prevents BTC trade from blocking ETH signals
        self._last_trade_call: dict[str, int] = {}
        self._call_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        # Keep original name so config lookup and strategy selector still work
        return "grid"

    def evaluate(self, df: pd.DataFrame, config: dict,
                 pair: str = "default") -> StrategySignal:
        if len(df) < 20:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        # Per-pair call counter for cooldown tracking
        self._call_count[pair] = self._call_count.get(pair, 0) + 1

        # --- Cooldown: 24 candles (2 hours) between trades ---
        # Squeeze breakouts need time to develop, prevents re-entry
        cooldown = config.get("cooldown_candles", 24)
        last_trade = self._last_trade_call.get(pair, -999)
        if self._call_count[pair] - last_trade < cooldown:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["cooldown active"])

        # --- Pull all indicator values ---
        close = latest["close"]
        open_price = latest["open"]
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        bb_middle = latest.get("bb_middle")
        bb_width = latest.get("bb_width")
        bb_width_sma = latest.get("bb_width_sma")
        prev_bb_width = prev.get("bb_width")
        vol_ratio = latest.get("volume_ratio")
        macd_hist = latest.get("macd_histogram")

        # Skip if indicators haven't warmed up
        if any(pd.isna(v) for v in [bb_upper, bb_lower, bb_middle,
                                     bb_width, bb_width_sma, prev_bb_width]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # ====== STEP 1: SQUEEZE DETECTION ======
        # Check if BB width was recently compressed (below its own average).
        # A squeeze means the market is coiling -- low volatility before a move.
        # We check the minimum BB width in the last 10 candles vs the SMA.
        lookback = min(10, len(df))
        recent_bb = df["bb_width"].iloc[-lookback:]
        valid_widths = recent_bb.dropna()
        if len(valid_widths) < 3:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["not enough BB data"])

        min_recent_bb = valid_widths.min()
        # Squeeze = minimum BB width was at least 15% below the 20-period average
        squeeze_ratio = config.get("squeeze_ratio", 0.85)
        was_squeezed = min_recent_bb < bb_width_sma * squeeze_ratio

        if not was_squeezed:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["no recent squeeze"])

        # ====== STEP 2: EXPANSION DETECTION ======
        # BB width must be RISING now (breaking out of the squeeze).
        # Expanding bands = volatility returning = move starting.
        is_expanding = bb_width > prev_bb_width and bb_width > bb_width_sma * 0.9

        if not is_expanding:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["squeeze not expanding yet"])

        # --- SL/TP from config ---
        sl_pct = config.get("stop_loss_pct", 0.5) / 100  # 0.5%
        tp_pct = config.get("take_profit_pct", 1.5) / 100  # 1.5%

        # ====== STEP 3: DIRECTIONAL BREAKOUT ======
        # Now that we have squeeze + expansion, determine which direction
        buy_score = 0
        sell_score = 0
        buy_reasons = ["squeeze breakout"]
        sell_reasons = ["squeeze breakout"]

        # 1. Price position relative to Bollinger Bands (2 pts for clear break)
        # Price above upper BB = bullish breakout from squeeze
        if close > bb_upper:
            buy_score += 2
            buy_reasons.append("price above upper BB")
        elif close > (bb_upper + bb_middle) / 2:
            # Price in upper half of BB = leaning bullish
            buy_score += 1
            buy_reasons.append("price in upper BB zone")

        # Price below lower BB = bearish breakout from squeeze
        if close < bb_lower:
            sell_score += 2
            sell_reasons.append("price below lower BB")
        elif close < (bb_lower + bb_middle) / 2:
            sell_score += 1
            sell_reasons.append("price in lower BB zone")

        # 2. Candle direction confirms the breakout (1 point)
        if close > open_price:
            buy_score += 1
            buy_reasons.append("bullish candle")
        if close < open_price:
            sell_score += 1
            sell_reasons.append("bearish candle")

        # 3. Volume confirms the move is real (1 point)
        vol_min = config.get("volume_min", 1.2)
        if not pd.isna(vol_ratio) and vol_ratio >= vol_min:
            if buy_score > 0:
                buy_score += 1
                buy_reasons.append(f"volume {vol_ratio:.1f}x")
            if sell_score > 0:
                sell_score += 1
                sell_reasons.append(f"volume {vol_ratio:.1f}x")

        # 4. MACD momentum direction (1 point)
        if not pd.isna(macd_hist):
            if macd_hist > 0:
                buy_score += 1
                buy_reasons.append("MACD bullish")
            if macd_hist < 0:
                sell_score += 1
                sell_reasons.append("MACD bearish")

        # --- Need 3+ points: BB position (1-2) + direction confirmations ---
        min_score = 3

        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / 5, 1.0)
            self._last_trade_call[pair] = self._call_count[pair]
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 - sl_pct), 8),
                take_profit=round(close * (1 + tp_pct), 8),
                strategy_name=self.name, reasons=buy_reasons,
            )

        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / 5, 1.0)
            self._last_trade_call[pair] = self._call_count[pair]
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 + sl_pct), 8),
                take_profit=round(close * (1 - tp_pct), 8),
                strategy_name=self.name, reasons=sell_reasons,
            )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=["squeeze direction unclear"])
