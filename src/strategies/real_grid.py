# src/strategies/real_grid.py
# Real Grid Engine — harvests volatility in ranging markets.
#
# Unlike directional strategies that predict price movement,
# a grid bot profits from EVERY bounce within a range.
# No prediction needed — just volatility.
#
# HOW IT WORKS:
# 1. Define a price range (from Bollinger Bands)
# 2. Place N evenly-spaced levels within that range
# 3. When price drops to a level → buy
# 4. When price rises to the next level → sell (profit = level spacing - fees)
# 5. Repeat. Every bounce = profit.
#
# SAFETY:
# - If price exits range by >1 ATR → close everything (trend starting)
# - Max exposure capped at 15% of balance
# - Grid pauses if BB width > 5.5% (too volatile)

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """One price level in the grid."""
    price: float
    # True = we bought at this level, waiting to sell at level above
    is_holding: bool = False
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_fee: float = 0.0


@dataclass
class GridState:
    """Full state of a grid for one trading pair."""
    pair: str
    lower_bound: float          # bottom of grid range
    upper_bound: float          # top of grid range
    levels: list[GridLevel] = field(default_factory=list)
    size_per_level: float = 0.0  # USDT per grid level
    fee_rate: float = 0.00075
    is_active: bool = True
    total_pnl: float = 0.0      # running P&L from completed grid fills
    total_fills: int = 0         # how many buy+sell cycles completed
    reserved_balance: float = 0.0  # USDT reserved from main balance


class GridCalculator:
    """Calculates grid levels from market data.

    Uses Bollinger Bands as the natural volatility range.
    Grid levels are evenly spaced between lower and upper BB.
    """

    def __init__(self,
                 num_levels: int = 10,
                 max_grid_exposure_pct: float = 15.0,
                 max_bb_width_pct: float = 5.5,
                 exit_atr_multiplier: float = 1.0,
                 fee_rate: float = 0.00075):
        self._num_levels = num_levels
        self._max_exposure_pct = max_grid_exposure_pct / 100
        self._max_bb_width = max_bb_width_pct / 100
        self._exit_atr_mult = exit_atr_multiplier
        self._fee_rate = fee_rate

    def can_activate(self, bb_width: float, regime: str) -> tuple[bool, str]:
        """Check if conditions allow grid activation."""
        if regime not in ("RANGING",):
            return False, f"regime is {regime}, need RANGING"
        if bb_width > self._max_bb_width:
            return False, f"BB width {bb_width:.4f} > max {self._max_bb_width:.4f}"
        if bb_width <= 0:
            return False, "invalid BB width"
        return True, "conditions met"

    def calculate_levels(self, bb_lower: float, bb_upper: float,
                         current_price: float,
                         balance: float) -> GridState | None:
        """Calculate grid levels between BB lower and upper.

        Returns a GridState ready to activate, or None if invalid.
        """
        if bb_lower >= bb_upper or bb_lower <= 0:
            logger.warning("Invalid BB range: %.2f - %.2f", bb_lower, bb_upper)
            return None

        if current_price < bb_lower or current_price > bb_upper:
            logger.warning("Price %.2f outside BB range %.2f-%.2f",
                          current_price, bb_lower, bb_upper)
            return None

        # Calculate how much balance to allocate to the grid
        total_grid_allocation = balance * self._max_exposure_pct
        size_per_level = total_grid_allocation / self._num_levels

        # Minimum viable level size (Binance minimum + fees)
        if size_per_level < 10.0:
            logger.info("Grid level size $%.2f too small (min $10)", size_per_level)
            return None

        # Create evenly spaced levels
        spacing = (bb_upper - bb_lower) / (self._num_levels + 1)
        levels = []
        for i in range(1, self._num_levels + 1):
            level_price = bb_lower + spacing * i
            levels.append(GridLevel(price=round(level_price, 8)))

        return GridState(
            pair="",  # set by caller
            lower_bound=bb_lower,
            upper_bound=bb_upper,
            levels=levels,
            size_per_level=round(size_per_level, 2),
            fee_rate=self._fee_rate,
            reserved_balance=round(total_grid_allocation, 2),
        )

    def should_close_grid(self, current_price: float, grid: GridState,
                          atr: float) -> tuple[bool, str]:
        """Check if price has exited the grid range (trend breakout)."""
        exit_distance = atr * self._exit_atr_mult

        if current_price > grid.upper_bound + exit_distance:
            return True, f"price {current_price:.2f} broke above range + ATR"
        if current_price < grid.lower_bound - exit_distance:
            return True, f"price {current_price:.2f} broke below range - ATR"

        return False, ""

    def get_grid_spacing_pct(self, grid: GridState) -> float:
        """Return the profit potential per grid fill as a percentage."""
        if not grid.levels or len(grid.levels) < 2:
            return 0.0
        spacing = grid.levels[1].price - grid.levels[0].price
        mid_price = (grid.upper_bound + grid.lower_bound) / 2
        return (spacing / mid_price) * 100 if mid_price > 0 else 0.0

    def get_profit_per_fill(self, grid: GridState) -> float:
        """Estimated profit per grid fill after fees."""
        if not grid.levels or len(grid.levels) < 2:
            return 0.0
        spacing = grid.levels[1].price - grid.levels[0].price
        mid_price = (grid.upper_bound + grid.lower_bound) / 2
        spacing_pct = spacing / mid_price if mid_price > 0 else 0
        # Profit = spacing% × position_size - round-trip fees
        gross = grid.size_per_level * spacing_pct
        fees = grid.size_per_level * grid.fee_rate * 2  # entry + exit
        return gross - fees
