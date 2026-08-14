# src/ai/embeddings.py
# Market State Embedding Engine — turns market snapshots into vectors.
#
# Instead of heavyweight transformer models (too heavy for 1GB RAM),
# we use handcrafted numerical feature vectors from technical indicators.
# This is actually what professional quant funds use — structured features
# beat text embeddings for numerical financial data.
#
# Each market snapshot becomes a fixed-length vector:
# [RSI, MACD_hist, BB_width, volume_ratio, EMA_trend, ATR, regime_encoded,
#  price_change_5m, price_change_15m, price_change_1h, ...]
#
# These vectors go into ChromaDB for similarity search —
# "find me the 5 historical moments most similar to right now"

import math
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Total number of features in the market state vector
VECTOR_SIZE = 16


@dataclass
class MarketSnapshot:
    """A complete snapshot of market conditions at a point in time.

    This is the input to the embedding engine — raw market data
    gets converted into a normalized feature vector.
    """
    # Which trading pair (e.g. "BTC/USDT")
    pair: str
    # Current price in USDT
    price: float
    # Technical indicators from the indicator engine
    rsi: float = 50.0
    macd_histogram: float = 0.0
    bb_width: float = 0.0
    volume_ratio: float = 1.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    atr: float = 0.0
    # Price changes over different windows
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    price_change_1h: float = 0.0
    # Market regime (encoded as number)
    regime: str = "RANGING"
    regime_confidence: float = 0.5
    # Order book data
    ob_imbalance: float = 0.0
    spread_pct: float = 0.0

    # Extra metadata (not part of embedding, but stored alongside)
    timestamp: str = ""
    strategy: str = ""


# Map regime names to numbers for embedding
REGIME_MAP = {
    "TRENDING_UP": 1.0,
    "TRENDING_DOWN": -1.0,
    "RANGING": 0.0,
    "VOLATILE": 0.5,
    "CRASH": -1.0,
    "BULLISH": 0.8,
    "BEARISH": -0.8,
}


class EmbeddingEngine:
    """Converts market snapshots into fixed-length numerical vectors.

    Each feature is normalized to roughly [-1, 1] or [0, 1] range
    so that cosine similarity works properly in ChromaDB.
    Normalization uses domain knowledge (RSI is always 0-100,
    volume ratio centers around 1.0, etc.) rather than learned params.
    """

    def embed(self, snapshot: MarketSnapshot) -> list[float]:
        """Convert a MarketSnapshot into a VECTOR_SIZE-dimensional vector.

        Returns a list of floats ready for ChromaDB storage.
        Each dimension is normalized so similarity search works well.
        """
        vector = [
            # Feature 0: RSI normalized to [-1, 1] (50 = neutral)
            self._normalize_rsi(snapshot.rsi),

            # Feature 1: MACD histogram (clamped and scaled)
            self._clamp(snapshot.macd_histogram / 100.0, -1.0, 1.0),

            # Feature 2: Bollinger Band width (volatility indicator)
            # Typical range 0.01-0.10, normalize to [0, 1]
            self._clamp(snapshot.bb_width * 10.0, 0.0, 1.0),

            # Feature 3: Volume ratio (1.0 = average, >1.5 = spike)
            # Normalize: log scale so 0.5→-0.7, 1.0→0, 2.0→0.7, 4.0→1.4
            self._clamp(math.log(max(snapshot.volume_ratio, 0.01)), -2.0, 2.0) / 2.0,

            # Feature 4: EMA trend direction
            # Positive = bullish (fast > slow), negative = bearish
            self._ema_trend(snapshot.ema_fast, snapshot.ema_slow, snapshot.price),

            # Feature 5: ATR relative to price (volatility %)
            self._clamp(
                (snapshot.atr / max(snapshot.price, 1.0)) * 100.0,
                0.0, 5.0
            ) / 5.0,

            # Feature 6: Market regime encoded as number
            REGIME_MAP.get(snapshot.regime, 0.0),

            # Feature 7: Regime confidence [0, 1]
            self._clamp(snapshot.regime_confidence, 0.0, 1.0),

            # Feature 8: 5-minute price change (%)
            self._clamp(snapshot.price_change_5m, -5.0, 5.0) / 5.0,

            # Feature 9: 15-minute price change (%)
            self._clamp(snapshot.price_change_15m, -10.0, 10.0) / 10.0,

            # Feature 10: 1-hour price change (%)
            self._clamp(snapshot.price_change_1h, -15.0, 15.0) / 15.0,

            # Feature 11: Order book imbalance [-1, 1]
            # Positive = more bids (bullish), negative = more asks (bearish)
            self._clamp(snapshot.ob_imbalance, -1.0, 1.0),

            # Feature 12: Spread percentage [0, 1]
            # Lower is better (more liquid)
            self._clamp(snapshot.spread_pct * 100.0, 0.0, 1.0),

            # Feature 13: RSI extreme indicator
            # 1.0 if oversold (<30), -1.0 if overbought (>70), 0 otherwise
            self._rsi_extreme(snapshot.rsi),

            # Feature 14: Volume spike binary
            # 1.0 if volume > 2x average, 0 otherwise
            1.0 if snapshot.volume_ratio > 2.0 else 0.0,

            # Feature 15: Trend alignment (EMA + price change agree)
            # 1.0 if both bullish, -1.0 if both bearish, 0 if mixed
            self._trend_alignment(
                snapshot.ema_fast, snapshot.ema_slow, snapshot.price_change_5m
            ),
        ]

        # Safety: make sure vector is exactly VECTOR_SIZE
        assert len(vector) == VECTOR_SIZE, (
            f"Vector size mismatch: got {len(vector)}, expected {VECTOR_SIZE}"
        )

        return vector

    def similarity_score(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 to 1.0 (1.0 = identical market conditions).
        Used when ChromaDB isn't available as a fallback.
        """
        # Dot product
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        # Magnitudes
        mag_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
        mag_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
        # Cosine similarity: [-1, 1] → scale to [0, 1]
        return (dot / (mag_a * mag_b) + 1.0) / 2.0

    def _normalize_rsi(self, rsi: float) -> float:
        """RSI 0-100 → [-1, 1]. 50 maps to 0 (neutral)."""
        return (self._clamp(rsi, 0.0, 100.0) - 50.0) / 50.0

    def _ema_trend(self, fast: float, slow: float, price: float) -> float:
        """EMA crossover direction, normalized by price."""
        if price <= 0 or slow <= 0:
            return 0.0
        # Percentage difference between fast and slow EMA
        diff_pct = ((fast - slow) / price) * 100.0
        return self._clamp(diff_pct, -2.0, 2.0) / 2.0

    def _rsi_extreme(self, rsi: float) -> float:
        """Detect RSI extremes for mean reversion signals."""
        if rsi < 30:
            return 1.0   # oversold → potential buy
        elif rsi > 70:
            return -1.0  # overbought → potential sell
        return 0.0

    def _trend_alignment(self, ema_fast: float, ema_slow: float,
                         price_change: float) -> float:
        """Check if EMA trend and price momentum agree."""
        ema_bullish = ema_fast > ema_slow
        price_bullish = price_change > 0

        if ema_bullish and price_bullish:
            return 1.0   # both bullish — strong alignment
        elif not ema_bullish and not price_bullish:
            return -1.0  # both bearish — strong alignment
        return 0.0        # mixed signals — no alignment

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp a value to [min_val, max_val] range."""
        return max(min_val, min(max_val, value))
