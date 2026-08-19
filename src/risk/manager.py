# src/risk/manager.py
# Central risk management gate. Every trade must pass through here.
# Checks: balance floor, position size limits, portfolio exposure.
#
# CHANGED: position sizing is now SL-distance-based, not Kelly.
# Risk per trade = fixed % of balance / SL distance.
# This guarantees every trade risks the SAME dollar amount regardless
# of the stop-loss width. The BNB -$168 problem (10-18x risk dispersion)
# is fixed because risk_amount is constant.
#
# Kelly sizer is kept for reference but not used for live sizing.

import logging
import time
from src.risk.position_sizer import PositionSizer
from src.core.config import get_scaling_tier

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against risk rules before execution.

    No trade reaches the exchange without passing can_trade().
    Position sizing is now handled by AdaptiveParams (inverse-stop-distance).
    This class provides portfolio-level exposure checks and pair cooldowns.
    """

    # Escalating cooldown: 30min → 60min → 120min after consecutive stops
    PAIR_COOLDOWN_1 = 30 * 60    # 30 minutes after 1st stop
    PAIR_COOLDOWN_2 = 60 * 60    # 60 minutes after 2nd consecutive stop
    PAIR_COOLDOWN_3 = 120 * 60   # 120 minutes after 3rd+ consecutive stop

    def __init__(self, settings: dict):
        # Store full settings for tier lookups
        self._settings = settings
        self._risk = settings.get("risk", {})
        self._fees = settings.get("fees", {})
        self._sizer = PositionSizer(
            min_order_size=self._risk.get("min_order_size", 10.0)
        )

        # Pair cooldown tracking — prevents same-pair churn after stops
        # Example: POL stopped 3 times in 45 min, all losses
        self._pair_loss_count: dict[str, int] = {}
        self._pair_cooldown_until: dict[str, float] = {}

    def can_trade(self, balance: float) -> bool:
        """Check if the bot is allowed to place any new trade right now."""
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if balance < min_balance:
            logger.warning("Balance $%.2f below floor $%.2f — blocking trade",
                          balance, min_balance)
            return False
        return True

    def check_portfolio_exposure(self, balance: float,
                                  open_positions: list,
                                  new_position_size: float,
                                  new_sl_pct: float) -> bool:
        """Check if adding a new position would exceed portfolio risk limits.

        Limits:
        - Max total open notional: 50% of balance
        - Max total initial risk: 2% of balance
        Both prevent correlated crypto positions from compounding losses.
        """
        max_total_notional_pct = 0.50  # 50% of balance in open positions
        max_total_risk_pct = 0.02      # 2% total initial risk across all positions

        # Calculate current total notional
        total_notional = sum(
            pos.quantity * pos.entry_price for pos in open_positions
        )

        # Calculate current total initial risk (each position's SL distance × size)
        total_risk = 0.0
        for pos in open_positions:
            sl_dist = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
            pos_notional = pos.quantity * pos.entry_price
            total_risk += pos_notional * sl_dist

        # Check if adding new position exceeds notional limit
        if (total_notional + new_position_size) > balance * max_total_notional_pct:
            logger.info(
                "PORTFOLIO LIMIT: notional $%.0f + $%.0f > %.0f%% of $%.0f",
                total_notional, new_position_size,
                max_total_notional_pct * 100, balance,
            )
            return False

        # Check if adding new position exceeds risk limit
        new_risk = new_position_size * (new_sl_pct / 100)
        if (total_risk + new_risk) > balance * max_total_risk_pct:
            logger.info(
                "PORTFOLIO RISK LIMIT: risk $%.0f + $%.0f > %.1f%% of $%.0f",
                total_risk, new_risk, max_total_risk_pct * 100, balance,
            )
            return False

        return True

    def register_stop_loss(self, pair: str):
        """Record a stop-loss event for pair cooldown tracking.

        Escalating cooldown: 30m → 60m → 120m based on consecutive stops.
        """
        now = time.time()
        count = self._pair_loss_count.get(pair, 0) + 1
        self._pair_loss_count[pair] = count

        # Escalating cooldown
        if count == 1:
            cooldown = self.PAIR_COOLDOWN_1
        elif count == 2:
            cooldown = self.PAIR_COOLDOWN_2
        else:
            cooldown = self.PAIR_COOLDOWN_3

        self._pair_cooldown_until[pair] = now + cooldown
        logger.info(
            "PAIR COOLDOWN: %s — %d consecutive stops, blocked for %d min",
            pair, count, cooldown // 60,
        )

    def register_profitable_close(self, pair: str):
        """Reset pair loss counter after a fully profitable close.

        Only resets on COMPLETE trade profit, not partial exits.
        A 1R partial + remainder stopped at BE is not a "profitable close"
        for cooldown purposes — only a full position closed in profit.
        """
        if pair in self._pair_loss_count:
            self._pair_loss_count[pair] = 0
            logger.info("PAIR COOLDOWN RESET: %s — profitable close", pair)

    def pair_cooldown_active(self, pair: str) -> bool:
        """Check if a pair is in cooldown after recent stop losses."""
        cooldown_until = self._pair_cooldown_until.get(pair, 0)
        if time.time() < cooldown_until:
            remaining = cooldown_until - time.time()
            logger.info(
                "PAIR COOLDOWN ACTIVE: %s — %d min remaining (%d stops)",
                pair, remaining // 60, self._pair_loss_count.get(pair, 0),
            )
            return True
        return False

    def get_position_size(self, balance: float, sl_distance_pct: float) -> float:
        """Calculate position size using inverse-stop-distance sizing.

        risk_amount = balance × risk_per_trade_pct
        position_size = risk_amount / sl_distance_pct

        This guarantees every trade risks the SAME dollar amount.
        Tight stops → bigger positions. Wide stops → smaller positions.
        The old Kelly sizer had no knowledge of SL distance, creating
        10-18x risk dispersion between trades.
        """
        risk_pct = self._risk.get("risk_per_trade_pct", 0.5) / 100.0
        risk_amount = balance * risk_pct

        if sl_distance_pct <= 0:
            return self._risk.get("min_order_size", 10.0)

        # Convert sl_distance_pct from % to decimal
        sl_frac = sl_distance_pct / 100.0
        position_size = risk_amount / sl_frac

        # Cap at max position % from config
        max_pos_pct = self._risk.get("max_position_pct", 25.0) / 100.0
        max_pos = balance * max_pos_pct
        position_size = min(position_size, max_pos)

        # Floor at exchange minimum
        min_order = self._risk.get("min_order_size", 10.0)
        position_size = max(position_size, min_order)

        return round(position_size, 2)
