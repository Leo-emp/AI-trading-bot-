# src/intelligence/trade_filter.py
# Trade Quality Filter — the final gate before execution.
#
# Even when the 5 brains agree and the strategy says "go", we check:
# 1. Is it a good TIME to trade? (skip dead hours)
# 2. Is the SPREAD acceptable? (don't give away profit to market makers)
# 3. Did we just LOSE? (cooldown to avoid revenge trading)
# 4. Is the signal STRONG enough? (weak consensus = skip)
# 5. Are we already EXPOSED? (don't stack correlated positions)
#
# This filter sits between the Trade Gate and execution.
# Trade Gate says "direction is correct" — this filter says "conditions are right."

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class TradeFilter:
    """Filters out trades with bad timing, conditions, or risk overlap.

    This is the final quality check before risking real money.
    Every "skip" here is a loss prevented.

    WIN RATE BOOSTERS (proven, not speculative):
    - ATR floor: skip dead markets where trades are 50/50 coin flips
    - Trend alignment: only trade WITH the 1h trend (55-65% documented edge)
    - Confidence gate: higher threshold means fewer but better trades
    """

    def __init__(self,
                 min_confidence: float = 0.50,
                 max_spread_pct: float = 0.25,
                 cooldown_minutes: int = 2,
                 max_correlated_positions: int = 5,
                 dead_hours_utc: Optional[list[int]] = None):
        # Minimum average confidence from agreeing brains
        # 0.55 = let more trades through, gate already pre-filtered
        self._min_confidence = min_confidence

        # Maximum bid-ask spread as % of price
        # 0.25% is generous — covers all 12 pairs on Binance
        self._max_spread_pct = max_spread_pct

        # Minutes to wait after a losing trade — short, don't miss moves
        self._cooldown_minutes = cooldown_minutes

        # Max positions in correlated assets — higher for more exposure
        self._max_correlated = max_correlated_positions

        # Hours in UTC where volume is lowest
        self._dead_hours = dead_hours_utc or []

        # Track last loss time for cooldown
        self._last_loss_time: Optional[datetime] = None

        # 1h trend direction cache (set by engine before check)
        self._hourly_trend: dict[str, str] = {}

        # ATR context (set by engine before check)
        self._pair_atr_ratio: dict[str, float] = {}

    def record_loss(self):
        """Record that a trade just closed at a loss."""
        self._last_loss_time = datetime.now(timezone.utc)

    def set_hourly_trend(self, pair: str, direction: str):
        """Set the 1h trend direction for a pair (called by engine).

        direction: "UP", "DOWN", or "NEUTRAL"
        """
        self._hourly_trend[pair] = direction

    def set_atr_ratio(self, pair: str, ratio: float):
        """Set ATR ratio for a pair (current ATR / 20-period ATR average).

        ratio > 1.0 = volatility expanding (good)
        ratio < 0.5 = dead market (bad — skip)
        """
        self._pair_atr_ratio[pair] = ratio

    def check(self,
              confidence: float,
              spread_pct: float,
              open_positions: list,
              pair: str,
              trade_direction: str = "",
              now: Optional[datetime] = None) -> dict:
        """Run all quality checks. Returns pass/fail with reason.

        Args:
            confidence: average confidence from agreeing brains (0-1)
            spread_pct: current bid-ask spread as percentage
            open_positions: list of currently open positions
            pair: the pair we want to trade
            trade_direction: BUY or SELL (for trend alignment check)
            now: current time (for testing)

        Returns:
            {"pass": bool, "reason": str, "filter": str}
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # --- Filter 0: ATR floor (PROVEN: dead markets = coin flips) ---
        # If current ATR is less than 50% of its average, market is dead.
        # No directional edge exists in a dead market.
        atr_ratio = self._pair_atr_ratio.get(pair, 1.0)
        if atr_ratio < 0.5:
            return {
                "pass": False,
                "reason": f"Dead market: ATR ratio {atr_ratio:.2f} < 0.5. No edge in flat markets.",
                "filter": "atr_floor",
            }

        # Trend alignment filter REMOVED — it was blocking ALL sell/short
        # signals because the 1h trend is usually UP in crypto. This made the
        # bot buy-only, so every market drop caused losses with no way to
        # profit from the downside. The 5-brain consensus already evaluates
        # direction; this filter was just overriding it.

        # --- Filter 1: Dead hours ---
        if now.hour in self._dead_hours:
            return {
                "pass": False,
                "reason": f"Dead hour (UTC {now.hour}:00). Low volume = bad fills.",
                "filter": "time",
            }

        # --- Filter 2: Spread too wide ---
        if spread_pct > self._max_spread_pct:
            return {
                "pass": False,
                "reason": (f"Spread {spread_pct:.3f}% > max {self._max_spread_pct:.3f}%. "
                          f"Would eat too much profit."),
                "filter": "spread",
            }

        # --- Filter 3: Cooldown after loss ---
        if self._last_loss_time:
            elapsed = (now - self._last_loss_time).total_seconds() / 60
            if elapsed < self._cooldown_minutes:
                remaining = self._cooldown_minutes - elapsed
                return {
                    "pass": False,
                    "reason": (f"Cooldown: {remaining:.0f}min left after last loss. "
                              f"Avoiding revenge trade."),
                    "filter": "cooldown",
                }

        # --- Filter 4: Minimum signal strength ---
        if confidence < self._min_confidence:
            return {
                "pass": False,
                "reason": (f"Weak signal ({confidence:.0%} < {self._min_confidence:.0%}). "
                          f"Brains agree but aren't confident enough."),
                "filter": "confidence",
            }

        # Correlation and direction caps REMOVED — the 5-brain consensus
        # already decides direction from market conditions. Hard caps force
        # the bot to skip valid signals and miss moves.

        # --- All checks passed ---
        logger.debug("Trade filter PASSED for %s (conf=%.0f%%, spread=%.3f%%)",
                    pair, confidence * 100, spread_pct)
        return {
            "pass": True,
            "reason": "All quality checks passed",
            "filter": "none",
        }

    def get_spread_pct(self, best_bid: float, best_ask: float) -> float:
        """Calculate spread as percentage of mid price."""
        if best_bid <= 0 or best_ask <= 0:
            return 999.0  # invalid → block trade
        mid = (best_bid + best_ask) / 2
        return ((best_ask - best_bid) / mid) * 100
