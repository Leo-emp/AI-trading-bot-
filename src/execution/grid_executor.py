# src/execution/grid_executor.py
# Grid Executor — manages the grid lifecycle independently.
#
# This is separate from the directional paper trader.
# The grid executor:
# 1. Reserves balance from the paper trader on activation
# 2. Manages its own positions internally (no max_positions conflict)
# 3. Returns balance + P&L on deactivation
#
# Each grid fill (buy at level N, sell at level N+1) is a micro-trade.
# Fees are simulated identically to the paper trader.

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.strategies.real_grid import GridState, GridLevel, GridCalculator

logger = logging.getLogger(__name__)


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

    def __init__(self, fee_rate: float = 0.00075):
        self._fee_rate = fee_rate
        self._grids: dict[str, GridState] = {}  # pair -> active grid
        self._fill_history: list[GridFill] = []
        self._calculator = GridCalculator(fee_rate=fee_rate)
        # Balance reserved from paper trader (returned on deactivation)
        self._reserved_balance: float = 0.0
        # Available balance within the grid allocation
        self._grid_cash: float = 0.0

    @property
    def active_pairs(self) -> list[str]:
        return [p for p, g in self._grids.items() if g.is_active]

    @property
    def total_pnl(self) -> float:
        return sum(g.total_pnl for g in self._grids.values())

    @property
    def total_fills(self) -> int:
        return sum(g.total_fills for g in self._grids.values())

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
        self._grids[pair] = grid_state
        self._reserved_balance += grid_state.reserved_balance
        self._grid_cash += grid_state.reserved_balance

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

        fills = []

        for i, level in enumerate(grid.levels):
            if level.is_holding:
                # We have a position at this level — check if we should sell
                # Sell when price rises to the NEXT level above
                sell_target = self._get_sell_target(grid, i)
                if sell_target is not None and candle_high >= sell_target:
                    fill = self._execute_sell(grid, level, sell_target)
                    if fill:
                        fills.append(fill)

            else:
                # No position at this level — check if we should buy
                # Buy when price drops to this level (or through it)
                if candle_low <= level.price and current_price >= level.price * 0.995:
                    # Price touched this level — buy
                    if self._grid_cash >= grid.size_per_level:
                        self._execute_buy(grid, level)

        return fills

    def _get_sell_target(self, grid: GridState, level_index: int) -> Optional[float]:
        """Get the sell target for a position at level_index.

        Sell target = the price of the next level above.
        If this is the highest level, sell at upper_bound.
        """
        if level_index + 1 < len(grid.levels):
            return grid.levels[level_index + 1].price
        return grid.upper_bound

    def _execute_buy(self, grid: GridState, level: GridLevel):
        """Execute a grid buy at this level."""
        buy_price = level.price
        quantity = grid.size_per_level / buy_price
        fee = grid.size_per_level * self._fee_rate

        # Deduct from grid cash
        self._grid_cash -= (grid.size_per_level + fee)

        level.is_holding = True
        level.entry_price = buy_price
        level.quantity = quantity
        level.entry_fee = fee

        logger.info(
            "GRID BUY | %s @ $%.2f | qty %.6f | fee $%.4f | grid_cash $%.2f",
            grid.pair, buy_price, quantity, fee, self._grid_cash,
        )

    def _execute_sell(self, grid: GridState, level: GridLevel,
                      sell_price: float) -> Optional[GridFill]:
        """Execute a grid sell — complete one buy+sell cycle."""
        if not level.is_holding or level.quantity <= 0:
            return None

        exit_notional = level.quantity * sell_price
        exit_fee = exit_notional * self._fee_rate
        gross_pnl = (sell_price - level.entry_price) * level.quantity
        net_pnl = gross_pnl - level.entry_fee - exit_fee

        # Return notional + profit to grid cash
        self._grid_cash += exit_notional - exit_fee

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
                self._grid_cash += exit_notional - exit_fee

                level.is_holding = False
                level.quantity = 0.0

                logger.info(
                    "GRID CLOSE | %s @ $%.2f (entry $%.2f) | P&L $%.4f",
                    pair, current_price, level.entry_price, net,
                )

        grid.total_pnl += close_pnl
        grid.is_active = False

        # Calculate what to return to paper trader
        # = remaining grid cash (original allocation + accumulated P&L)
        balance_return = self._grid_cash
        self._reserved_balance -= grid.reserved_balance
        self._grid_cash = max(0.0, self._grid_cash - grid.reserved_balance)

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
            "grid_cash": round(self._grid_cash, 2),
        }
