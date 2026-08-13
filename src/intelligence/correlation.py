# src/intelligence/correlation.py
# Tracks price correlations between assets in real-time.
# Used by Brain 5 to prevent correlated positions:
# - If BTC is dumping, don't go long on altcoins
# - If ETH and SOL are moving together, don't hold both long
# - Max 60% exposure to any single pair

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class CorrelationTracker:
    """Tracks rolling price correlation between crypto assets.

    Maintains a sliding window of recent prices per pair and computes
    pairwise correlation coefficients. Also tracks BTC's short-term
    trend (since BTC leads the market).
    """

    def __init__(self, window: int = 50):
        # How many price points to keep per pair
        self._window = window
        # Recent prices per pair: {"BTC/USDT": deque([50000, 50050, ...])}
        self._prices: dict[str, deque] = {}

    def update(self, pair: str, price: float):
        """Record a new price for a pair."""
        if pair not in self._prices:
            self._prices[pair] = deque(maxlen=self._window)
        self._prices[pair].append(price)

    def is_correlated(self, pair_a: str, pair_b: str,
                      threshold: float = 0.7) -> bool:
        """Check if two pairs are strongly correlated.

        Uses Pearson correlation on recent returns.
        Returns True if |correlation| >= threshold.
        """
        returns_a = self._get_returns(pair_a)
        returns_b = self._get_returns(pair_b)

        if returns_a is None or returns_b is None:
            return False  # not enough data

        # Use the shorter length
        n = min(len(returns_a), len(returns_b))
        if n < 10:
            return False

        ra = list(returns_a)[-n:]
        rb = list(returns_b)[-n:]

        # Pearson correlation
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb)) / n
        std_a = (sum((a - mean_a) ** 2 for a in ra) / n) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in rb) / n) ** 0.5

        if std_a == 0 or std_b == 0:
            return False

        corr = cov / (std_a * std_b)
        return abs(corr) >= threshold

    def btc_trend(self) -> str:
        """Determine BTC's short-term trend.

        Returns "BULLISH", "BEARISH", or "NEUTRAL".
        Based on last 10 price changes.
        """
        btc_key = "BTC/USDT"
        if btc_key not in self._prices or len(self._prices[btc_key]) < 10:
            return "NEUTRAL"

        prices = list(self._prices[btc_key])[-10:]
        # Simple: compare first vs last
        change_pct = (prices[-1] - prices[0]) / prices[0] * 100

        if change_pct > 0.3:
            return "BULLISH"
        elif change_pct < -0.3:
            return "BEARISH"
        return "NEUTRAL"

    def get_brain_signal(self, pair: str) -> Optional[dict]:
        """Generate Brain 5 signal for cross-asset correlation.

        If holding an altcoin long and BTC is dumping → SELL signal.
        If BTC is bullish → supports BUY on altcoins.
        """
        btc = self.btc_trend()

        # BTC pairs always get HOLD from correlation brain
        if pair == "BTC/USDT":
            return {"direction": "HOLD", "confidence": 0.5}

        if btc == "BEARISH":
            # BTC dumping → don't long altcoins
            return {"direction": "SELL", "confidence": 0.6}
        elif btc == "BULLISH":
            # BTC rising → supports altcoin longs
            return {"direction": "BUY", "confidence": 0.55}

        return {"direction": "HOLD", "confidence": 0.5}

    def _get_returns(self, pair: str) -> Optional[list]:
        """Calculate percentage returns from price series."""
        if pair not in self._prices or len(self._prices[pair]) < 2:
            return None
        prices = list(self._prices[pair])
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
