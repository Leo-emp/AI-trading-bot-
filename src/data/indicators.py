# src/data/indicators.py
# Computes technical indicators from OHLCV data using the `ta` library.
# Every indicator is a proven, widely-used tool for reading price action:
#
# - RSI: momentum oscillator — is the asset overbought or oversold?
# - MACD: trend direction and momentum — is a trend starting or ending?
# - Bollinger Bands: volatility envelope — is price at an extreme?
# - ATR: average true range — how volatile is the asset right now?
# - EMA: exponential moving average — smoothed price trend
# - Volume ratio: is current volume unusual compared to recent average?

from dataclasses import dataclass, field

import pandas as pd
import ta


@dataclass(frozen=True)
class TechnicalSignal:
    """Output of the technical analysis brain.

    direction: BUY, SELL, or HOLD
    confidence: 0.0 (no confidence) to 1.0 (very confident)
    reasons: human-readable list of why this signal was generated
    """
    direction: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


class IndicatorEngine:
    """Computes all technical indicators on an OHLCV DataFrame.

    Usage:
        engine = IndicatorEngine()
        df = engine.compute_all(ohlcv_df)   # adds indicator columns
        signal = engine.get_signal(df, strategy_config)  # BUY/SELL/HOLD
    """

    def __init__(self, rsi_period: int = 14, bb_period: int = 20,
                 bb_std: float = 2.0, atr_period: int = 14,
                 ema_fast: int = 9, ema_slow: int = 21,
                 volume_sma_period: int = 20):
        self._rsi_period = rsi_period
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._atr_period = atr_period
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._vol_sma_period = volume_sma_period

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicator columns to the DataFrame.

        Input must have columns: open, high, low, close, volume.
        Returns the same DataFrame with additional indicator columns.
        Does NOT modify the original DataFrame.
        """
        df = df.copy()

        # --- RSI: Relative Strength Index ---
        # Measures momentum: >70 = overbought, <30 = oversold
        df["rsi"] = ta.momentum.rsi(df["close"], window=self._rsi_period)

        # --- MACD: Moving Average Convergence Divergence ---
        # Shows trend direction and momentum changes
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # --- Bollinger Bands: volatility envelope around price ---
        # Price touching upper band = possibly overbought
        # Price touching lower band = possibly oversold
        bb = ta.volatility.BollingerBands(df["close"], window=self._bb_period, window_dev=self._bb_std)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        # Width shows volatility — narrow = squeeze, wide = expansion
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        # BB width moving average -- for squeeze detection
        # When bb_width < bb_width_sma, volatility is compressed (squeeze forming)
        # Squeeze -> expansion -> breakout is a proven pattern (John Bollinger)
        df["bb_width_sma"] = df["bb_width"].rolling(window=20).mean()

        # --- ATR: Average True Range ---
        # Measures volatility in price terms (used for grid spacing, stop sizes)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=self._atr_period)

        # --- EMAs: Exponential Moving Averages ---
        # Fast EMA crossing above slow = bullish, below = bearish
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=self._ema_fast)
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=self._ema_slow)

        # --- Volume analysis ---
        # Volume ratio: current volume / average volume
        # >1.5 = volume spike (something is happening)
        df["volume_sma"] = df["volume"].rolling(window=self._vol_sma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # --- ADX: Average Directional Index ---
        # Measures trend STRENGTH (not direction) on a 0-100 scale.
        # < 20 = weak/no trend (ranging market -- good for mean reversion)
        # 20-25 = emerging trend (transition zone)
        # > 25 = strong trend (good for breakout and momentum strategies)
        # > 50 = very strong trend (rare, ride it hard)
        # This is THE regime gate: each strategy only trades in its optimal ADX range.
        # Research: mean reversion profit factor 1.62 at ADX<20, -0.74 at ADX>30.
        adx_calc = ta.trend.ADXIndicator(
            df["high"], df["low"], df["close"], window=self._atr_period
        )
        df["adx"] = adx_calc.adx()
        # +DI measures bullish pressure, -DI measures bearish pressure
        # +DI > -DI = bulls winning, -DI > +DI = bears winning
        df["di_plus"] = adx_calc.adx_pos()
        df["di_minus"] = adx_calc.adx_neg()

        # --- Donchian Channel (40-period, shifted by 1) ---
        # Highest high / lowest low over last N candles.
        # SHIFTED by 1 period so donchian_high at candle i = max high from [i-40, i-1]
        # This way close > donchian_high means a genuine NEW high = breakout.
        # Different from BB (std dev envelope): Donchian tracks actual price extremes.
        # Evidence: Sharpe 1.95 on ETH/USDT (CoinQuant backtest, 2025-2026)
        donchian_period = 40  # ~3.3 hours at 5m candles
        df["donchian_high"] = df["high"].rolling(window=donchian_period).max().shift(1)
        df["donchian_low"] = df["low"].rolling(window=donchian_period).min().shift(1)
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2

        return df

    def get_signal(self, df: pd.DataFrame, config: dict) -> TechnicalSignal:
        """Generate a BUY/SELL/HOLD signal from computed indicators.

        This is Brain 1 of the multi-brain consensus engine.
        Uses multiple indicators for confirmation — no single
        indicator alone can trigger a trade.

        config keys: rsi_oversold, rsi_overbought, volume_spike_multiplier
        """
        if len(df) < 2:
            return TechnicalSignal("HOLD", 0.0, ["insufficient data"])

        # Use the latest complete candle (not the current incomplete one)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        reasons = []
        buy_score = 0  # positive signals
        sell_score = 0

        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        prev_macd_hist = prev.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        close = latest["close"]
        bb_lower = latest.get("bb_lower")
        bb_upper = latest.get("bb_upper")
        ema_f = latest.get("ema_fast")
        ema_s = latest.get("ema_slow")

        # Skip if indicators haven't warmed up yet
        if pd.isna(rsi) or pd.isna(macd_hist):
            return TechnicalSignal("HOLD", 0.0, ["indicators warming up"])

        # --- RSI signal ---
        oversold = config.get("rsi_oversold", 30)
        overbought = config.get("rsi_overbought", 70)
        if rsi < oversold:
            buy_score += 1
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > overbought:
            sell_score += 1
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # --- MACD crossover ---
        if not pd.isna(prev_macd_hist):
            if prev_macd_hist < 0 and macd_hist > 0:
                buy_score += 1
                reasons.append("MACD bullish crossover")
            elif prev_macd_hist > 0 and macd_hist < 0:
                sell_score += 1
                reasons.append("MACD bearish crossover")

        # --- Bollinger Band touch ---
        if not pd.isna(bb_lower) and close <= bb_lower:
            buy_score += 1
            reasons.append("price at lower Bollinger Band")
        elif not pd.isna(bb_upper) and close >= bb_upper:
            sell_score += 1
            reasons.append("price at upper Bollinger Band")

        # --- EMA crossover ---
        if not pd.isna(ema_f) and not pd.isna(ema_s):
            prev_ema_f = prev.get("ema_fast")
            prev_ema_s = prev.get("ema_slow")
            if not pd.isna(prev_ema_f) and not pd.isna(prev_ema_s):
                if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                    buy_score += 1
                    reasons.append("EMA bullish crossover")
                elif prev_ema_f >= prev_ema_s and ema_f < ema_s:
                    sell_score += 1
                    reasons.append("EMA bearish crossover")

        # --- Volume confirmation ---
        # High volume confirms the signal; low volume weakens it
        vol_multiplier = config.get("volume_spike_multiplier", 1.5)
        has_volume = not pd.isna(vol_ratio) and vol_ratio >= vol_multiplier
        if has_volume:
            reasons.append(f"volume spike ({vol_ratio:.1f}x)")

        # --- Combine scores into direction + confidence ---
        # Need at least 2 confirming indicators for a signal
        if buy_score >= 2 and buy_score > sell_score:
            confidence = min(buy_score / 4, 1.0)  # max 4 indicators
            if has_volume:
                confidence = min(confidence + 0.15, 1.0)
            return TechnicalSignal("BUY", confidence, reasons)
        elif sell_score >= 2 and sell_score > buy_score:
            confidence = min(sell_score / 4, 1.0)
            if has_volume:
                confidence = min(confidence + 0.15, 1.0)
            return TechnicalSignal("SELL", confidence, reasons)
        else:
            return TechnicalSignal("HOLD", 0.0, reasons if reasons else ["no confirmed signal"])
