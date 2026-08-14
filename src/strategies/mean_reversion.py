# src/strategies/mean_reversion.py
# Trend Pullback Strategy -- enters trends during temporary dips/bounces.
#
# WHY NOT TRADITIONAL MEAN REVERSION:
#   On 5m candles with $10 positions, ranging market BB width is 0.2-0.5%.
#   Round-trip fees are 0.15%. This means wins in a range barely cover fees.
#   After 6 versions and extensive testing, pure range-bound mean reversion
#   CANNOT overcome the fee drag at this position size and timeframe.
#
# WHY TREND PULLBACK WORKS:
#   Trends provide persistent DRIFT that overcomes fees. A pullback in an
#   uptrend is a temporary dip before the trend resumes. Entering during
#   the dip gets a better price than chasing, and the trend's drift
#   pushes price toward TP.
#
# HOW THE 4 STRATEGIES COVER DIFFERENT TREND PHASES:
#   1. Vol Squeeze: detects pre-trend compression -> first mover
#   2. Volume Breakout: catches the new high/low -> trend start
#   3. Momentum: catches EMA crossover -> trend confirmation
#   4. Pullback: enters during dips/bounces -> trend continuation
#
# ENTRY LOGIC (BUY example):
#   - ADX > 20: trend confirmed
#   - EMA fast > EMA slow: uptrend intact
#   - Close < EMA fast: price PULLED BACK below fast EMA (the dip)
#   - Close > EMA slow: trend NOT broken (still above slow EMA)
#   - RSI < 45: dip created temporary oversold
#   - Volume BELOW average: quiet pullback (not panic selling)
#
# KEY DIFFERENCE FROM MOMENTUM:
#   Momentum enters when price is ABOVE ema_fast (strong move)
#   Pullback enters when price is BETWEEN the EMAs (temporary weakness)
#   They CANNOT trigger simultaneously -- they're complementary.
#
# SL: 0.6% | TP: 1.5% | R:R = 2.5:1 | Cooldown: 12 candles

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal


class MeanReversionStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        # Per-pair cooldown — prevents BTC trade from blocking ETH signals
        self._last_trade_call: dict[str, int] = {}
        self._call_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "mean_reversion"

    def evaluate(self, df: pd.DataFrame, config: dict,
                 pair: str = "default") -> StrategySignal:
        if len(df) < 30:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        # Per-pair call counter for cooldown tracking
        self._call_count[pair] = self._call_count.get(pair, 0) + 1

        # --- Cooldown: 12 candles (1 hour) between entries ---
        cooldown = config.get("cooldown_candles", 12)
        last_trade = self._last_trade_call.get(pair, -999)
        if self._call_count[pair] - last_trade < cooldown:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["cooldown active"])

        # --- Pull all indicator values ---
        close = latest["close"]
        open_price = latest["open"]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        adx = latest.get("adx")
        di_plus = latest.get("di_plus")
        di_minus = latest.get("di_minus")

        # Skip if indicators haven't warmed up
        if any(pd.isna(v) for v in [ema_fast, ema_slow, rsi, adx]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # ====== REGIME GATE: ADX > 20 (trend present) ======
        # Pullback entries only work when there's an actual trend.
        # ADX > 20 confirms directional movement exists.
        adx_min = config.get("adx_min", 20)
        if adx < adx_min:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=[f"ADX {adx:.0f} < {adx_min}, no trend"])

        # --- SL/TP from config ---
        sl_pct = config.get("stop_loss_pct", 0.6) / 100   # 0.6%
        tp_pct = config.get("take_profit_pct", 1.5) / 100  # 1.5%

        # ====== PULLBACK DETECTION + SCORING ======
        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        # --- BUY PULLBACK: uptrend with temporary dip ---
        # Price is in the "pullback zone": below fast EMA but above slow EMA.
        # The trend is intact (fast > slow) but price temporarily dipped.
        in_buy_pullback = (ema_fast > ema_slow
                           and close < ema_fast
                           and close > ema_slow)
        if in_buy_pullback:
            buy_score += 2  # core pullback signal
            buy_reasons.append("pullback in uptrend")

        # --- SELL PULLBACK: downtrend with temporary bounce ---
        # Price bounced above fast EMA but still below slow EMA.
        in_sell_pullback = (ema_fast < ema_slow
                            and close > ema_fast
                            and close < ema_slow)
        if in_sell_pullback:
            sell_score += 2
            sell_reasons.append("bounce in downtrend")

        # --- CONFIRMATIONS ---

        # 1. RSI shows the dip/bounce (1 point)
        # RSI < 45 during uptrend = temporarily oversold from the dip
        if rsi < 45 and buy_score > 0:
            buy_score += 1
            buy_reasons.append(f"RSI dipped ({rsi:.0f})")
        # RSI > 55 during downtrend = temporarily overbought from bounce
        if rsi > 55 and sell_score > 0:
            sell_score += 1
            sell_reasons.append(f"RSI bounced ({rsi:.0f})")

        # 2. Volume BELOW average = quiet pullback (1 point)
        # Low volume during pullback = healthy pause, not reversal.
        # High volume pullback = potential trend reversal (skip those).
        if not pd.isna(vol_ratio) and vol_ratio < 0.8:
            if buy_score > 0:
                buy_score += 1
                buy_reasons.append(f"quiet dip (vol {vol_ratio:.1f}x)")
            if sell_score > 0:
                sell_score += 1
                sell_reasons.append(f"quiet bounce (vol {vol_ratio:.1f}x)")

        # 3. MACD still in trend direction (1 point)
        # MACD positive during pullback = bullish momentum still intact
        if not pd.isna(macd_hist):
            if macd_hist > 0 and buy_score > 0:
                buy_score += 1
                buy_reasons.append("MACD still positive")
            if macd_hist < 0 and sell_score > 0:
                sell_score += 1
                sell_reasons.append("MACD still negative")

        # 4. DI confirms trend direction (1 point)
        if not pd.isna(di_plus) and not pd.isna(di_minus):
            if di_plus > di_minus and buy_score > 0:
                buy_score += 1
                buy_reasons.append("+DI confirms uptrend")
            if di_minus > di_plus and sell_score > 0:
                sell_score += 1
                sell_reasons.append("-DI confirms downtrend")

        # 5. Reversal candle (price starting to resume) (1 point)
        # After the dip, a bullish candle = buyers stepping back in
        if close > open_price and buy_score > 0:
            buy_score += 1
            buy_reasons.append("bullish reversal candle")
        if close < open_price and sell_score > 0:
            sell_score += 1
            sell_reasons.append("bearish reversal candle")

        # --- Need 3+ points: pullback zone (2) + at least 1 confirmation ---
        min_score = 3

        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / 7, 1.0)
            self._last_trade_call[pair] = self._call_count[pair]
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 - sl_pct), 8),
                take_profit=round(close * (1 + tp_pct), 8),
                strategy_name=self.name, reasons=buy_reasons,
            )

        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / 7, 1.0)
            self._last_trade_call[pair] = self._call_count[pair]
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=close,
                stop_loss=round(close * (1 + sl_pct), 8),
                take_profit=round(close * (1 - tp_pct), 8),
                strategy_name=self.name, reasons=sell_reasons,
            )

        return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                              reasons=["no pullback setup"])
