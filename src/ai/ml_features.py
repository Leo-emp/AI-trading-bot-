# src/ai/ml_features.py
# Feature Engineering for ML Model (Phase 4).
#
# Transforms raw OHLCV data + indicators into feature vectors
# that XGBoost can learn from. Each row = one potential trade signal.
#
# Features are organized into groups:
# 1. Price-based (returns, momentum, trend)
# 2. Volume-based (spikes, trends, ratios)
# 3. Indicator-based (RSI, MACD, BB, ATR)
# 4. Market microstructure (spread, order book)
# 5. Regime context (regime type, confidence)
# 6. RAG-derived (historical similarity win rate)
#
# Labels: 1 = profitable trade, 0 = losing trade
# This is a binary classification problem.

import logging
import math
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# All feature names in order — ML model expects this exact order
FEATURE_NAMES = [
    # Price features (7)
    "return_5m", "return_15m", "return_1h",
    "momentum_5", "momentum_10", "momentum_20",
    "price_vs_ema_fast",

    # Volume features (4)
    "volume_ratio", "volume_trend_5", "volume_spike",
    "volume_consistency",

    # Indicator features (8)
    "rsi", "rsi_slope_5", "macd_histogram", "macd_signal_cross",
    "bb_position", "bb_width", "atr_pct", "adx",

    # Order book features (3)
    "ob_imbalance", "spread_pct", "whale_pressure",

    # Regime features (3)
    "regime_encoded", "regime_confidence", "regime_stability",

    # RAG features (3)
    "rag_win_rate", "rag_avg_pnl", "rag_similarity",

    # Time features (2)
    "hour_sin", "hour_cos",
]


class FeatureEngineer:
    """Transforms raw market data into ML-ready feature vectors.

    Call extract_features() with a DataFrame of OHLCV + indicators
    to get a feature vector that XGBoost can use for prediction.
    """

    def __init__(self):
        # Running stats for feature normalization (updated during training)
        self._feature_means: Optional[dict] = None
        self._feature_stds: Optional[dict] = None

    def extract_features(self, df: pd.DataFrame,
                         regime: str = "RANGING",
                         regime_confidence: float = 0.5,
                         ob_imbalance: float = 0.0,
                         spread_pct: float = 0.0,
                         whale_pressure: float = 0.0,
                         rag_stats: Optional[dict] = None,
                         hour: int = 12) -> list[float]:
        """Extract a feature vector from the latest market data.

        Args:
            df: DataFrame with OHLCV + indicator columns (needs 20+ rows)
            regime: current market regime string
            regime_confidence: confidence of regime detection
            ob_imbalance: order book imbalance [-1, 1]
            spread_pct: bid-ask spread as percentage
            whale_pressure: whale buy/sell pressure [-1, 1]
            rag_stats: dict from RAG memory (win_rate, avg_pnl, avg_similarity)
            hour: current hour UTC (for time-of-day features)

        Returns:
            list of floats — one value per feature in FEATURE_NAMES order
        """
        if len(df) < 20:
            logger.warning("Need at least 20 rows for features, got %d", len(df))
            return [0.0] * len(FEATURE_NAMES)

        latest = df.iloc[-1]
        closes = df["close"].values

        # --- Price features ---
        return_5m = self._pct_change(closes, 1)    # 1 candle = 5 min
        return_15m = self._pct_change(closes, 3)   # 3 candles = 15 min
        return_1h = self._pct_change(closes, 12)   # 12 candles = 1 hour
        momentum_5 = self._momentum(closes, 5)
        momentum_10 = self._momentum(closes, 10)
        momentum_20 = self._momentum(closes, 20)

        ema_fast = float(latest.get("ema_fast", closes[-1]))
        price_vs_ema = ((closes[-1] - ema_fast) / ema_fast * 100) if ema_fast > 0 else 0

        # --- Volume features ---
        volumes = df["volume"].values
        vol_ratio = float(latest.get("volume_ratio", 1.0))
        vol_trend_5 = self._trend(volumes, 5)
        vol_spike = 1.0 if vol_ratio > 2.0 else 0.0
        vol_consistency = self._consistency(volumes, 10)

        # --- Indicator features ---
        rsi = float(latest.get("rsi", 50.0))
        rsi_vals = df["rsi"].values if "rsi" in df.columns else np.full(len(df), 50.0)
        rsi_slope = self._slope(rsi_vals, 5)
        macd_hist = float(latest.get("macd_histogram", 0.0))

        # MACD signal crossover: 1 if just crossed above, -1 if below
        macd_cross = 0.0
        if "macd_histogram" in df.columns and len(df) >= 2:
            prev_hist = float(df.iloc[-2].get("macd_histogram", 0.0))
            if prev_hist <= 0 and macd_hist > 0:
                macd_cross = 1.0   # bullish cross
            elif prev_hist >= 0 and macd_hist < 0:
                macd_cross = -1.0  # bearish cross

        bb_pos = float(latest.get("bb_position", 0.5))
        bb_width = float(latest.get("bb_width", 0.0))
        atr = float(latest.get("atr", 0.0))
        atr_pct = (atr / closes[-1] * 100) if closes[-1] > 0 else 0
        adx = float(latest.get("adx", 25.0))

        # --- Regime features ---
        regime_map = {
            "TRENDING_UP": 1.0, "TRENDING_DOWN": -1.0,
            "RANGING": 0.0, "VOLATILE": 0.5, "CRASH": -1.0,
        }
        regime_encoded = regime_map.get(regime, 0.0)
        # Regime stability: how long have we been in this regime?
        # Approximate by checking if regime indicators are consistent
        regime_stability = min(regime_confidence * 1.5, 1.0)

        # --- RAG features (from historical pattern matching) ---
        rag = rag_stats or {}
        rag_win_rate = rag.get("win_rate", 0.5)
        rag_avg_pnl = rag.get("avg_pnl", 0.0)
        rag_similarity = rag.get("avg_similarity", 0.0)

        # --- Time features (cyclical encoding) ---
        # Sin/cos encoding preserves the circular nature of time
        # (23:00 is close to 00:00, not far apart)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # Assemble the feature vector in FEATURE_NAMES order
        features = [
            return_5m, return_15m, return_1h,
            momentum_5, momentum_10, momentum_20,
            price_vs_ema,
            vol_ratio, vol_trend_5, vol_spike, vol_consistency,
            rsi, rsi_slope, macd_hist, macd_cross,
            bb_pos, bb_width, atr_pct, adx,
            ob_imbalance, spread_pct, whale_pressure,
            regime_encoded, regime_confidence, regime_stability,
            rag_win_rate, rag_avg_pnl, rag_similarity,
            hour_sin, hour_cos,
        ]

        assert len(features) == len(FEATURE_NAMES), (
            f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}"
        )

        return features

    def build_training_dataset(self, trades: list[dict],
                                market_data: dict) -> tuple:
        """Build X (features) and y (labels) from historical trades.

        Args:
            trades: list of trade dicts with market context
            market_data: dict of DataFrames keyed by pair+timestamp

        Returns:
            (X, y) where X is numpy array of features, y is array of 0/1
        """
        X = []
        y = []

        for trade in trades:
            features = trade.get("features")
            if features and len(features) == len(FEATURE_NAMES):
                X.append(features)
                # Label: 1 if trade was profitable, 0 if not
                y.append(1 if trade.get("pnl", 0) > 0 else 0)

        if not X:
            logger.warning("No valid training samples found")
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def _pct_change(self, values: np.ndarray, periods: int) -> float:
        """Percentage change over N periods."""
        if len(values) <= periods or values[-periods - 1] == 0:
            return 0.0
        return ((values[-1] - values[-periods - 1]) / values[-periods - 1]) * 100

    def _momentum(self, values: np.ndarray, window: int) -> float:
        """Rate of change over a window — positive = up, negative = down."""
        if len(values) < window:
            return 0.0
        return self._pct_change(values, window)

    def _trend(self, values: np.ndarray, window: int) -> float:
        """Linear regression slope over a window, normalized."""
        if len(values) < window:
            return 0.0
        segment = values[-window:]
        x = np.arange(window, dtype=float)
        # Simple linear regression slope
        x_mean = x.mean()
        y_mean = segment.mean()
        if y_mean == 0:
            return 0.0
        slope = np.sum((x - x_mean) * (segment - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
        # Normalize by mean value
        return (slope / y_mean) * 100

    def _slope(self, values: np.ndarray, window: int) -> float:
        """Slope of an indicator over N periods."""
        if len(values) < window:
            return 0.0
        return float(values[-1] - values[-window])

    def _consistency(self, values: np.ndarray, window: int) -> float:
        """How consistent a series is (low std = consistent). 0 to 1."""
        if len(values) < window:
            return 0.5
        segment = values[-window:]
        mean = segment.mean()
        if mean == 0:
            return 0.5
        cv = segment.std() / mean  # coefficient of variation
        # Invert: high consistency = low CV
        return max(0.0, 1.0 - min(cv, 1.0))
