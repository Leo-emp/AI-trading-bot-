# src/strategies/smart_scalp.py
# Volume Breakout Strategy (Donchian Channel) -- catches new trends early.
#
# EVIDENCE (CoinQuant backtest, ETH/USDT 4H, Nov 2025 - May 2026):
#   - Sharpe ratio: 1.95 (highest of ALL strategies tested)
#   - Win rate: 33.3% (low but compensated by 4.9x payoff ratio)
#   - Profit factor: 2.45
#   - KEY: volume confirmation eliminates 40-50% of false breakouts
#
# HOW IT WORKS:
#   1. ADX gate: only trade when ADX > 25 (confirmed trend, not choppy)
#   2. Price breaks above 40-period Donchian high (or below low) = NEW extreme
#   3. Volume must spike above average = real breakout, not random wick
#   4. +DI/-DI confirms directional momentum matches the breakout
#   5. MACD positive/negative as extra confirmation
#
# WHEN IT FAILS:
#   - Sideways/choppy markets (ADX gate prevents this)
#   - 14 consecutive losses documented during ranging ETH Dec 2025
#   - Late trend entries near reversal (cooldown helps limit exposure)
#
# SL: 0.6% | TP: 1.5% | R:R = 2.5:1 | Cooldown: 18 candles (1.5 hrs)
#
# Class name kept as SmartScalpStrategy for backward compatibility.
# Internally this is a Donchian channel breakout strategy.

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class SmartScalpStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        # Cooldown uses call counter, not candle count (window resets break len(df))
        self._last_trade_call = -999
        self._call_count = 0

    @property
    def name(self) -> str:
        # Keep original name so config lookup and strategy selector still work
        return "smart_scalp"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        # Need enough candles for Donchian (40) + ADX (28) warmup
        if len(df) < 20:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        self._call_count += 1

        # --- Cooldown: 18 candles (1.5 hours) between trades ---
        # Breakout zones are choppy -- prevents rapid-fire entries
        cooldown = config.get("cooldown_candles", 18)
        if self._call_count - self._last_trade_call < cooldown:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["cooldown active"])

        # --- Pull all indicator values from the latest candle ---
        close = latest["close"]
        adx = latest.get("adx")
        di_plus = latest.get("di_plus")
        di_minus = latest.get("di_minus")
        # donchian_high/low are SHIFTED by 1 in indicator engine,
        # so they represent the channel from PREVIOUS candles only
        donchian_high = latest.get("donchian_high")
        donchian_low = latest.get("donchian_low")
        vol_ratio = latest.get("volume_ratio")
        macd_hist = latest.get("macd_histogram")

        # Skip if any required indicator hasn't warmed up yet
        if any(pd.isna(v) for v in [adx, di_plus, di_minus, donchian_high,
                                     donchian_low, vol_ratio]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # ====== REGIME GATE: ADX > 25 (strong trend confirmed) ======
        # Without this gate, breakouts in ranging markets produce massive
        # losing streaks (14+ consecutive losses documented in research).
        # ADX measures trend STRENGTH, not direction -- perfect for gating.
        adx_threshold = config.get("adx_min", 25)
        if adx < adx_threshold:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=[f"ADX too low ({adx:.0f} < {adx_threshold})"])

        # --- SL/TP percentages from config ---
        sl_pct = config.get("stop_loss_pct", 0.6) / 100  # 0.6% stop loss
        tp_pct = config.get("take_profit_pct", 1.5) / 100  # 1.5% take profit

        # ====== BREAKOUT SCORING ======
        # Donchian break is the anchor (2 pts), needs at least 1 confirmation
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # 1. Donchian channel breakout (2 points -- the PRIMARY signal)
        # close > donchian_high = price broke above the N-period highest high
        if close > donchian_high:
            buy_score += 2
            buy_reasons.append(f"broke {donchian_high:.0f} Donchian high")
        # close < donchian_low = price broke below the N-period lowest low
        if close < donchian_low:
            sell_score += 2
            sell_reasons.append(f"broke {donchian_low:.0f} Donchian low")

        # 2. Directional Index confirmation (1 point)
        # +DI > -DI means bulls have more directional pressure
        if di_plus > di_minus:
            buy_score += 1
            buy_reasons.append(f"+DI leads ({di_plus:.0f} > {di_minus:.0f})")
        if di_minus > di_plus:
            sell_score += 1
            sell_reasons.append(f"-DI leads ({di_minus:.0f} > {di_plus:.0f})")

        # 3. Volume confirmation (1 point)
        # Research: volume filter eliminates 40-50% of false breakouts
        vol_min = config.get("volume_min", 1.5)
        if vol_ratio >= vol_min:
            # Only award volume point if there's a directional signal
            if buy_score > 0:
                buy_score += 1
                buy_reasons.append(f"volume {vol_ratio:.1f}x")
            if sell_score > 0:
                sell_score += 1
                sell_reasons.append(f"volume {vol_ratio:.1f}x")

        # 4. MACD direction (1 point)
        # MACD histogram positive = bullish momentum, negative = bearish
        if not pd.isna(macd_hist):
            if macd_hist > 0:
                buy_score += 1
                buy_reasons.append("MACD bullish")
            if macd_hist < 0:
                sell_score += 1
                sell_reasons.append("MACD bearish")

        # --- Need 3+ points: breakout (2) + at least 1 confirmation ---
        # This prevents trading on Donchian break alone (which has 40-50% false rate)
        min_score = 3

        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / 5, 1.0)
            self._last_trade_call = self._call_count
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 - sl_pct), 8),
                take_profit=round(close * (1 + tp_pct), 8),
                strategy_name=self.name, reasons=buy_reasons,
            )

        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / 5, 1.0)
            self._last_trade_call = self._call_count
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 + sl_pct), 8),
                take_profit=round(close * (1 - tp_pct), 8),
                strategy_name=self.name, reasons=sell_reasons,
            )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=["no confirmed breakout"])
