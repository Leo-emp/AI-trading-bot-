# src/risk/position_sizer.py
# Position sizing using the Kelly Criterion with a safety factor.
#
# Kelly Criterion: the mathematically optimal fraction of your bankroll
# to bet on each trade, given your win rate and average win/loss sizes.
# We use QUARTER-Kelly (0.25 safety factor) because:
# - Full Kelly is too aggressive for real trading
# - Quarter-Kelly gives ~75% of the growth with much less risk
# - It's the standard for professional systematic trading

from src.core.config import get_scaling_tier


class PositionSizer:
    """Calculates position sizes using Kelly Criterion.

    Respects exchange minimums (Binance $10) and portfolio percentage
    limits from the auto-scaling tier system.
    """

    def __init__(self, min_order_size: float = 10.0, safety_factor: float = 0.25):
        # Binance minimum order size in USDT
        self._min_order = min_order_size
        # Quarter-Kelly: multiply Kelly fraction by this
        self._safety = safety_factor

    def kelly_size(self, win_rate: float, avg_win: float,
                   avg_loss: float, balance: float) -> float:
        """Calculate optimal position size using quarter-Kelly.

        Formula: f* = (p * W - (1-p) * L) / W
        Where p = win rate, W = avg win, L = avg loss (positive number)

        Returns position size in USDT, floored at min_order_size.
        """
        # Ensure avg_loss is positive for the formula
        avg_loss = abs(avg_loss) if avg_loss != 0 else 0.01

        if avg_win <= 0:
            return self._min_order

        # Kelly fraction: what % of bankroll to risk
        # This is the edge divided by the odds
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        if kelly_fraction <= 0:
            # Negative edge — no profitable bet exists
            # Return minimum to allow the system to keep learning
            return self._min_order

        # Apply safety factor (quarter-Kelly)
        safe_fraction = kelly_fraction * self._safety

        # Convert fraction to dollar amount
        position_size = balance * safe_fraction

        # Cap at 25% of balance (spec max per-trade)
        max_position = balance * 0.25
        position_size = min(position_size, max_position)

        # Floor at exchange minimum (takes priority over cap — Binance requires $10)
        position_size = max(position_size, self._min_order)

        return round(position_size, 2)

    def get_tier(self, balance: float, settings: dict) -> dict:
        """Get the risk scaling tier for the current balance.

        Returns a dict with: max_risk_pct, max_positions, daily_trade_limit.
        Higher balances get slightly looser limits.
        """
        return get_scaling_tier(balance, settings)
