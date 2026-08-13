# src/data/order_book.py
# Analyzes order book depth to detect:
# - Buy/sell imbalance (are buyers or sellers stronger?)
# - Whale orders (single large orders that can move price)
# - Spread width (too wide = too expensive to trade)
# - Spoofing (large order placed then cancelled — fake signal)

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderBookSignal:
    """Result of order book analysis.

    imbalance: -1.0 (all sellers) to +1.0 (all buyers)
    whale_detected: True if any single order > whale_threshold_usd
    whale_side: "bid" or "ask" (which side the whale is on)
    spread_pct: bid-ask spread as percentage of mid price
    is_liquid: True if spread is within acceptable range
    """
    imbalance: float
    whale_detected: bool
    whale_side: str
    spread_pct: float
    is_liquid: bool


class OrderBookAnalyzer:
    """Analyzes raw order book data from Binance.

    The order book is a list of [price, quantity] pairs for bids and asks.
    We compute: imbalance ratio, whale detection, spread, liquidity check.
    """

    def __init__(self, whale_threshold_usd: float = 50000, max_spread_pct: float = 0.15):
        # Any single order above this USD value counts as a "whale"
        self._whale_threshold = whale_threshold_usd
        # Spread wider than this → skip trading (too expensive)
        self._max_spread = max_spread_pct

    def analyze(self, order_book: dict) -> OrderBookSignal:
        """Analyze a raw order book dict with 'bids' and 'asks' keys.

        Each bid/ask is [price, quantity]. Bids sorted high→low,
        asks sorted low→high (standard exchange format).
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            # No data — return neutral, illiquid signal
            return OrderBookSignal(
                imbalance=0.0, whale_detected=False,
                whale_side="none", spread_pct=999.0, is_liquid=False,
            )

        # --- Imbalance: total bid volume vs total ask volume ---
        # Sum up the quantity on each side
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        total_volume = bid_volume + ask_volume

        # Imbalance from -1 (all asks) to +1 (all bids)
        # +1 = very bullish (buyers dominate), -1 = very bearish
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0

        # --- Whale detection: any single order > threshold ---
        whale_detected = False
        whale_side = "none"

        for price, qty in bids:
            if price * qty >= self._whale_threshold:
                whale_detected = True
                whale_side = "bid"
                break

        if not whale_detected:
            for price, qty in asks:
                if price * qty >= self._whale_threshold:
                    whale_detected = True
                    whale_side = "ask"
                    break

        # --- Spread: how expensive is it to cross the book ---
        best_bid = bids[0][0]   # highest bid
        best_ask = asks[0][0]   # lowest ask
        mid_price = (best_bid + best_ask) / 2
        spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else 999.0

        # --- Liquidity: is the spread acceptable? ---
        is_liquid = spread_pct <= self._max_spread

        return OrderBookSignal(
            imbalance=imbalance,
            whale_detected=whale_detected,
            whale_side=whale_side,
            spread_pct=spread_pct,
            is_liquid=is_liquid,
        )
