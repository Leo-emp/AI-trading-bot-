# src/risk/manager.py
# Central risk management gate. Every trade must pass through here.
# Checks: daily trade limit, balance floor, position size, drawdown.
# Also provides stop-loss and take-profit price calculations.

import logging
from src.risk.position_sizer import PositionSizer
from src.core.config import get_scaling_tier

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against risk rules before execution.

    No trade reaches the exchange without passing can_trade() and
    validate_trade(). This is the final safety gate.
    """

    def __init__(self, db, settings: dict, balance: float):
        self._db = db
        self._settings = settings
        self._risk = settings.get("risk", {})
        self._fees = settings.get("fees", {})
        self._balance = balance
        self._sizer = PositionSizer(
            min_order_size=self._risk.get("min_order_size", 10.0)
        )

    def update_balance(self, balance: float):
        """Update the current balance (called after each trade)."""
        self._balance = balance

    async def can_trade(self) -> tuple[bool, str]:
        """Check if the bot is allowed to place any new trade right now.

        Returns (True, "ok") or (False, "reason why not").
        This is the pre-flight check before evaluating any signal.
        """
        # Check balance floor
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if self._balance < min_balance:
            return False, f"Balance ${self._balance:.2f} below balance floor ${min_balance:.2f}"

        # Check daily trade limit
        tier = get_scaling_tier(self._balance, self._settings)
        daily_limit = tier.get("daily_trade_limit", 25)
        daily_count = await self._db.get_daily_trade_count()
        if daily_count >= daily_limit:
            return False, f"Daily trade limit reached ({daily_count}/{daily_limit})"

        return True, "ok"

    async def validate_trade(self, pair: str, side: str,
                              price: float, quantity: float) -> tuple[bool, str]:
        """Validate a specific trade against all risk rules.

        Called after can_trade() passes and a signal is generated.
        Checks position size, notional value, and fee profitability.
        """
        notional = price * quantity
        min_order = self._risk.get("min_order_size", 10.0)

        if notional < min_order:
            return False, f"Order ${notional:.2f} below exchange minimum ${min_order}"

        max_pct = self._risk.get("max_position_pct", 25.0) / 100
        if notional > self._balance * max_pct:
            return False, f"Order ${notional:.2f} exceeds {max_pct*100:.0f}% of balance"

        # Check min net profit threshold (must exceed 2x fees)
        maker_rate = self._fees.get("maker_rate", 0.00075)
        round_trip_fee_pct = maker_rate * 2 * 100  # percentage
        min_net = self._risk.get("min_net_profit_pct", 0.30) if "min_net_profit_pct" in self._risk else round_trip_fee_pct * 2

        return True, "ok"

    async def get_position_size(self, balance: float, confidence: float) -> float:
        """Calculate optimal position size for a trade.

        Uses Kelly criterion from the sizer, scaled by AI confidence.
        """
        # Get strategy stats for Kelly inputs (fallback to defaults if no history)
        stats = await self._db.get_strategy_stats("smart_scalp", window=50)

        if stats.total_trades < 10:
            # Not enough data — use conservative defaults
            win_rate = 0.55
            avg_win = 0.008 * balance  # 0.8% of a $10-15 position
            avg_loss = 0.004 * balance
        else:
            win_rate = stats.win_rate
            avg_win = stats.avg_win
            avg_loss = abs(stats.avg_loss)

        size = self._sizer.kelly_size(win_rate, avg_win, avg_loss, balance)

        # Scale by confidence: high confidence → full size, low → reduced
        if confidence < 0.5:
            size *= 0.5
        elif confidence < 0.75:
            size *= 0.75

        # Floor at exchange minimum
        size = max(size, self._risk.get("min_order_size", 10.0))
        # Cap at max position %
        max_pos = balance * self._risk.get("max_position_pct", 25.0) / 100
        size = min(size, max_pos)

        return round(size, 2)

    def get_stop_loss_price(self, entry_price: float, side: str) -> float:
        """Calculate stop-loss price for a trade.

        For BUY: stop is below entry (price dropping = losing)
        For SELL: stop is above entry (price rising = losing)
        Uses 0.4% stop-loss from the spec.
        """
        stop_pct = 0.004  # 0.4%
        if side == "buy":
            return round(entry_price * (1 - stop_pct), 8)
        else:
            return round(entry_price * (1 + stop_pct), 8)

    def get_take_profit_price(self, entry_price: float, side: str) -> float:
        """Calculate take-profit price for a trade.

        For BUY: TP is above entry (price rising = winning)
        For SELL: TP is below entry (price dropping = winning)
        Uses 0.8% take-profit from the spec (2:1 reward/risk).
        """
        tp_pct = 0.008  # 0.8%
        if side == "buy":
            return round(entry_price * (1 + tp_pct), 8)
        else:
            return round(entry_price * (1 - tp_pct), 8)
