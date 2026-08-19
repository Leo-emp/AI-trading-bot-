# src/execution/grid_executor.py
# Grid Executor — manages the grid lifecycle independently.
#
# RISK MANAGEMENT:
# - Per-grid cash isolation: each grid's capital is separate
# - Inventory scaling: position size shrinks as more levels fill
#   (reduces directional exposure when most vulnerable)
# - Max loss protection: force-closes grid if unrealized loss
#   exceeds threshold (before ATR breakout)
# - Fully loaded detection: signals when all levels are filled
#   and price is below the range (maximum pain, no upside)

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.strategies.real_grid import GridState, GridLevel, GridCalculator

logger = logging.getLogger(__name__)

# Scale position size down as more levels fill.
# At 0 filled: 100% size. At 4 filled: 40% size.
# Reduces total directional exposure when we're most vulnerable.
INVENTORY_SCALE = [1.0, 0.85, 0.70, 0.55, 0.40]


@dataclass
class GridFill:
    """Record of one completed grid fill (buy + sell cycle)."""
    pair: str
    buy_price: float
    sell_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    timestamp: float


class GridExecutor:
    """Manages grid trading positions for all pairs.

    Completely independent from the directional paper trader.
    Reserves balance on activation, returns it on deactivation.
    """

    def __init__(self, fee_rate: float = 0.00075,
                 max_loss_pct: float = 40.0):
        self._fee_rate = fee_rate
        self._max_loss_pct = max_loss_pct
        self._grids: dict[str, GridState] = {}  # pair -> active grid
        self._fill_history: list[GridFill] = []
        self._calculator = GridCalculator(fee_rate=fee_rate)
        # Balance reserved from paper trader (total across all grids)
        self._reserved_balance: float = 0.0
        # Per-grid cash pools — each grid's capital is isolated
        self._grid_cash: dict[str, float] = {}

    @property
    def active_pairs(self) -> list[str]:
        return [p for p, g in self._grids.items() if g.is_active]

    @property
    def total_pnl(self) -> float:
        return sum(g.total_pnl for g in self._grids.values())

    @property
    def total_fills(self) -> int:
        return sum(g.total_fills for g in self._grids.values())

    @property
    def total_grid_cash(self) -> float:
        """Total unspent cash across all active grids."""
        return sum(self._grid_cash.get(p, 0.0) for p in self.active_pairs)

    def get_grid(self, pair: str) -> Optional[GridState]:
        return self._grids.get(pair)

    def get_open_exposure(self) -> float:
        """Total USDT currently held in grid positions."""
        total = 0.0
        for grid in self._grids.values():
            if not grid.is_active:
                continue
            for level in grid.levels:
                if level.is_holding:
                    total += level.quantity * level.entry_price
        return total

    def activate_grid(self, pair: str, grid_state: GridState,
                      paper_balance: float) -> tuple[bool, str]:
        """Activate a grid for a pair. Reserves balance from paper trader.

        Returns (success, message). Caller must deduct reserved_balance
        from paper trader if success is True.
        """
        if pair in self._grids and self._grids[pair].is_active:
            return False, f"grid already active for {pair}"

        if grid_state.reserved_balance > paper_balance * 0.5:
            return False, f"grid needs ${grid_state.reserved_balance:.0f} but only ${paper_balance:.0f} available"

        grid_state.pair = pair
        grid_state.is_active = True
        grid_state.activated_at = time.time()
        self._grids[pair] = grid_state
        self._reserved_balance += grid_state.reserved_balance
        # Each grid gets its own isolated cash pool
        self._grid_cash[pair] = grid_state.reserved_balance

        logger.info(
            "GRID ACTIVATED | %s | range $%.2f-$%.2f | %d levels | "
            "$%.2f per level | total $%.2f reserved",
            pair, grid_state.lower_bound, grid_state.upper_bound,
            len(grid_state.levels), grid_state.size_per_level,
            grid_state.reserved_balance,
        )
        return True, "grid activated"

    def update(self, pair: str, current_price: float,
               candle_high: float, candle_low: float) -> list[GridFill]:
        """Process price update for an active grid. Returns completed fills."""
        grid = self._grids.get(pair)
        if grid is None or not grid.is_active:
            return []

        pair_cash = self._grid_cash.get(pair, 0.0)
        filled_count = sum(1 for l in grid.levels if l.is_holding)
        fills = []

        for i, level in enumerate(grid.levels):
            if level.is_holding:
                # Check if we should sell at the next level above
                sell_target = self._get_sell_target(grid, i)
                if sell_target is not None and candle_high >= sell_target:
                    fill = self._execute_sell(grid, pair, level, sell_target)
                    if fill:
                        fills.append(fill)
                        filled_count -= 1

            else:
                # Check if we should buy at this level
                if candle_low <= level.price and current_price >= level.price * 0.995:
                    # Inventory-scaled position size
                    scale = self._get_inventory_scale(filled_count)
                    adjusted_size = grid.size_per_level * scale
                    pair_cash = self._grid_cash.get(pair, 0.0)

                    if pair_cash >= adjusted_size:
                        self._execute_buy(grid, pair, level, adjusted_size)
                        filled_count += 1

        return fills

    def _get_inventory_scale(self, filled_count: int) -> float:
        """Scale factor based on how many levels are already filled."""
        if filled_count < len(INVENTORY_SCALE):
            return INVENTORY_SCALE[filled_count]
        return INVENTORY_SCALE[-1]

    def _get_sell_target(self, grid: GridState, level_index: int) -> Optional[float]:
        """Sell target = the price of the next level above.
        If this is the highest level, sell at upper_bound.
        """
        if level_index + 1 < len(grid.levels):
            return grid.levels[level_index + 1].price
        return grid.upper_bound

    def _execute_buy(self, grid: GridState, pair: str,
                     level: GridLevel, position_size: float):
        """Execute a grid buy at this level with inventory-scaled size."""
        buy_price = level.price
        quantity = position_size / buy_price
        fee = position_size * self._fee_rate

        # Deduct from this grid's isolated cash pool
        self._grid_cash[pair] -= (position_size + fee)

        level.is_holding = True
        level.entry_price = buy_price
        level.quantity = quantity
        level.entry_fee = fee

        logger.info(
            "GRID BUY | %s @ $%.2f | qty %.6f | size $%.0f | fee $%.4f | grid_cash $%.2f",
            grid.pair, buy_price, quantity, position_size, fee,
            self._grid_cash[pair],
        )

    def _execute_sell(self, grid: GridState, pair: str,
                      level: GridLevel,
                      sell_price: float) -> Optional[GridFill]:
        """Execute a grid sell — complete one buy+sell cycle."""
        if not level.is_holding or level.quantity <= 0:
            return None

        exit_notional = level.quantity * sell_price
        exit_fee = exit_notional * self._fee_rate
        gross_pnl = (sell_price - level.entry_price) * level.quantity
        net_pnl = gross_pnl - level.entry_fee - exit_fee

        # Return notional + profit to this grid's cash pool
        self._grid_cash[pair] += exit_notional - exit_fee

        fill = GridFill(
            pair=grid.pair,
            buy_price=level.entry_price,
            sell_price=sell_price,
            quantity=level.quantity,
            gross_pnl=gross_pnl,
            fees=level.entry_fee + exit_fee,
            net_pnl=net_pnl,
            timestamp=time.time(),
        )

        # Reset level for next cycle
        level.is_holding = False
        level.entry_price = 0.0
        level.quantity = 0.0
        level.entry_fee = 0.0

        # Update grid totals
        grid.total_pnl += net_pnl
        grid.total_fills += 1

        self._fill_history.append(fill)

        logger.info(
            "GRID SELL | %s @ $%.2f (bought $%.2f) | net P&L $%.4f | "
            "total fills %d | grid P&L $%.4f",
            grid.pair, sell_price, fill.buy_price, net_pnl,
            grid.total_fills, grid.total_pnl,
        )
        return fill

    def deactivate_grid(self, pair: str,
                        current_price: float) -> tuple[float, str]:
        """Deactivate a grid and close all open positions at market.

        Returns (net_balance_change, reason).
        The caller adds this to the paper trader balance.
        """
        grid = self._grids.get(pair)
        if grid is None or not grid.is_active:
            return 0.0, "no active grid"

        # Close any open positions at current price (may be losses)
        close_pnl = 0.0
        for level in grid.levels:
            if level.is_holding:
                exit_notional = level.quantity * current_price
                exit_fee = exit_notional * self._fee_rate
                gross = (current_price - level.entry_price) * level.quantity
                net = gross - level.entry_fee - exit_fee
                close_pnl += net
                self._grid_cash[pair] += exit_notional - exit_fee

                level.is_holding = False
                level.quantity = 0.0

                logger.info(
                    "GRID CLOSE | %s @ $%.2f (entry $%.2f) | P&L $%.4f",
                    pair, current_price, level.entry_price, net,
                )

        grid.total_pnl += close_pnl
        grid.is_active = False

        # Return this grid's remaining cash to paper trader
        balance_return = self._grid_cash.pop(pair, 0.0)
        self._reserved_balance -= grid.reserved_balance

        summary = (
            f"grid closed: {grid.total_fills} fills, "
            f"P&L ${grid.total_pnl:.2f}, returning ${balance_return:.2f}"
        )
        logger.info("GRID DEACTIVATED | %s | %s", pair, summary)

        return balance_return, summary

    def get_unrealized_pnl(self, pair: str, current_price: float) -> float:
        """Calculate unrealized P&L on open grid positions."""
        grid = self._grids.get(pair)
        if grid is None or not grid.is_active:
            return 0.0

        pnl = 0.0
        for level in grid.levels:
            if level.is_holding:
                pnl += (current_price - level.entry_price) * level.quantity
        return pnl

    def get_loss_pct(self, pair: str, current_price: float) -> float:
        """Unrealized loss as % of reserved balance. 0 if in profit."""
        grid = self._grids.get(pair)
        if not grid or not grid.is_active or grid.reserved_balance <= 0:
            return 0.0
        unrealized = self.get_unrealized_pnl(pair, current_price)
        if unrealized >= 0:
            return 0.0
        return abs(unrealized) / grid.reserved_balance * 100

    def should_force_close(self, pair: str, current_price: float) -> tuple[bool, str]:
        """Check if grid should be force-closed due to excessive loss."""
        loss_pct = self.get_loss_pct(pair, current_price)
        if loss_pct >= self._max_loss_pct:
            return True, f"unrealized loss {loss_pct:.1f}% >= {self._max_loss_pct:.0f}% max"
        return False, ""

    def is_fully_loaded(self, pair: str) -> bool:
        """True if all grid levels are filled (maximum directional exposure)."""
        grid = self._grids.get(pair)
        if not grid or not grid.is_active:
            return False
        return all(l.is_holding for l in grid.levels)

    def get_status(self, pair: str) -> dict:
        """Get grid status for logging/Telegram."""
        grid = self._grids.get(pair)
        if grid is None:
            return {"active": False}

        holding = sum(1 for l in grid.levels if l.is_holding)
        return {
            "active": grid.is_active,
            "pair": pair,
            "range": f"${grid.lower_bound:.2f}-${grid.upper_bound:.2f}",
            "levels": len(grid.levels),
            "holding": holding,
            "fills": grid.total_fills,
            "pnl": round(grid.total_pnl, 4),
            "grid_cash": round(self._grid_cash.get(pair, 0.0), 2),
        }
