# src/risk/manager.py
# Central risk management gate. Every trade must pass through here.
# Checks: balance floor, position size limits, fee profitability.
# Also provides Kelly-based position sizing.
#
# FIXED: Constructor now takes (settings) only — no db dependency.
# Engine handles daily trade counting separately.

import logging
from src.risk.position_sizer import PositionSizer
from src.core.config import get_scaling_tier

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against risk rules before execution.

    No trade reaches the exchange without passing can_trade() and
    having its position size calculated here.
    """

    def __init__(self, settings: dict):
        # Store full settings for tier lookups
        self._settings = settings
        self._risk = settings.get("risk", {})
        self._fees = settings.get("fees", {})
        self._sizer = PositionSizer(
            min_order_size=self._risk.get("min_order_size", 10.0)
        )

    def can_trade(self, balance: float) -> bool:
        """Check if the bot is allowed to place any new trade right now.

        Returns True if balance is above the floor, False otherwise.
        """
        # Check balance floor — don't trade if we're nearly wiped out
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if balance < min_balance:
            logger.warning("Balance $%.2f below floor $%.2f — blocking trade",
                          balance, min_balance)
            return False
        return True

    def get_position_size(self, balance: float, win_rate: float,
                          avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using quarter-Kelly.

        Args:
            balance: current USDT balance
            win_rate: rolling win rate (0-1)
            avg_win: average winning trade as fraction (e.g. 0.008)
            avg_loss: average losing trade as fraction (e.g. 0.004)

        Returns position size in USDT.
        """
        # Scale avg_win/avg_loss to dollar amounts for Kelly formula
        # These come from the performance tracker as fractions
        avg_win_usd = avg_win * balance
        avg_loss_usd = avg_loss * balance

        size = self._sizer.kelly_size(win_rate, avg_win_usd, avg_loss_usd, balance)

        # Cap at max position % from scaling tier
        tier = get_scaling_tier(balance, self._settings)
        max_pct = tier.get("max_risk_pct", 0.5) / 100
        # Use the broader max_position_pct from risk config
        max_pos_pct = self._risk.get("max_position_pct", 25.0) / 100
        max_pos = balance * max_pos_pct
        size = min(size, max_pos)

        # Floor at exchange minimum
        min_order = self._risk.get("min_order_size", 10.0)
        size = max(size, min_order)

        return round(size, 2)
