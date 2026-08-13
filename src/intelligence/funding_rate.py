# src/intelligence/funding_rate.py
# Funding Rate Signal — free edge from Binance futures data.
#
# What is funding rate?
# - When longs pay shorts (positive funding), the market is overleveraged long
# - When shorts pay longs (negative funding), the market is overleveraged short
# - Extreme funding = market is about to reverse
#
# Why this is real edge:
# - It's not a lagging indicator (RSI, MACD look at past prices)
# - It shows POSITIONING — what traders are actually betting
# - Extreme funding → forced liquidations → price reversal
# - Free data from Binance API, updated every 8 hours
#
# How we use it:
# - Funding > +0.05% → market too long → bearish signal (expect dump)
# - Funding < -0.05% → market too short → bullish signal (expect squeeze)
# - Between -0.03% and +0.03% → neutral, no signal
# - This becomes an input to the 5-brain consensus system

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Funding rate thresholds (Binance default rate is ~0.01%)
EXTREME_POSITIVE = 0.05   # very overleveraged long → bearish
HIGH_POSITIVE = 0.03      # moderately overleveraged long
EXTREME_NEGATIVE = -0.05  # very overleveraged short → bullish
HIGH_NEGATIVE = -0.03     # moderately overleveraged short


class FundingRateSignal:
    """Analyzes Binance perpetual funding rates for contrarian signals.

    Extreme funding rates predict reversals because:
    1. High positive funding → longs paying expensive fees → they close → price drops
    2. High negative funding → shorts paying expensive fees → they close → price rises
    3. Liquidation cascades amplify these moves
    """

    def __init__(self):
        # Cache funding rates to avoid excessive API calls
        self._rates: dict[str, float] = {}
        self._last_fetch: dict[str, float] = {}

    async def fetch_funding_rate(self, exchange, pair: str) -> Optional[float]:
        """Fetch current funding rate from Binance futures.

        Returns funding rate as decimal (e.g., 0.0001 = 0.01%).
        Returns None if unavailable (spot-only pairs).
        """
        try:
            # Convert spot pair to futures symbol
            # BTC/USDT → BTC/USDT:USDT (ccxt perpetual format)
            perp_symbol = f"{pair}:USDT"

            # Fetch funding rate
            funding = await exchange.fetch_funding_rate(perp_symbol)
            if funding and "fundingRate" in funding:
                rate = float(funding["fundingRate"])
                # Convert to percentage for easier reading
                rate_pct = rate * 100
                self._rates[pair] = rate_pct
                logger.debug("Funding rate %s: %.4f%%", pair, rate_pct)
                return rate_pct
        except Exception as e:
            logger.debug("Funding rate unavailable for %s: %s", pair, e)

        return self._rates.get(pair)

    def analyze(self, pair: str, funding_rate_pct: Optional[float] = None) -> dict:
        """Analyze funding rate and return a trading signal.

        Args:
            pair: trading pair
            funding_rate_pct: funding rate as percentage (0.01 = 0.01%)

        Returns:
            {"direction": "BUY"/"SELL"/"HOLD",
             "confidence": 0.0-1.0,
             "reason": str}
        """
        rate = funding_rate_pct if funding_rate_pct is not None else self._rates.get(pair)

        if rate is None:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "reason": "No funding rate data available",
            }

        # Extreme positive funding → market overleveraged long → expect dump
        if rate >= EXTREME_POSITIVE:
            return {
                "direction": "SELL",
                "confidence": min(0.5 + (rate - EXTREME_POSITIVE) * 5, 0.85),
                "reason": f"Extreme positive funding ({rate:.3f}%) — longs overleveraged, expect squeeze down",
            }

        # High positive funding → leaning bearish
        if rate >= HIGH_POSITIVE:
            return {
                "direction": "SELL",
                "confidence": 0.4 + (rate - HIGH_POSITIVE) * 3,
                "reason": f"High positive funding ({rate:.3f}%) — market leaning overleveraged long",
            }

        # Extreme negative funding → market overleveraged short → expect squeeze up
        if rate <= EXTREME_NEGATIVE:
            return {
                "direction": "BUY",
                "confidence": min(0.5 + abs(rate - EXTREME_NEGATIVE) * 5, 0.85),
                "reason": f"Extreme negative funding ({rate:.3f}%) — shorts overleveraged, expect squeeze up",
            }

        # High negative funding → leaning bullish
        if rate <= HIGH_NEGATIVE:
            return {
                "direction": "BUY",
                "confidence": 0.4 + abs(rate - HIGH_NEGATIVE) * 3,
                "reason": f"High negative funding ({rate:.3f}%) — market leaning overleveraged short",
            }

        # Neutral funding → no signal
        return {
            "direction": "HOLD",
            "confidence": 0.0,
            "reason": f"Neutral funding ({rate:.3f}%) — no positioning extreme",
        }
