# src/execution/partial_exit.py
# Partial Profit Taking — take some profit early, let the rest ride.
#
# The problem with all-or-nothing exits:
# - Exit too early → miss big moves
# - Exit too late → give back profit on reversal
#
# Solution: split exit into stages
# - Take 50% at first target (+0.5%) → lock in guaranteed profit
# - Let remaining 50% ride with trailing stop → capture big moves
#
# Real example on $25 position:
# - Price moves +0.5%: take $12.50 off → profit $0.06 locked
# - Price continues to +1.5%: remaining $12.50 exits → profit $0.19
# - Total: $0.25 (vs $0.10 with fixed TP at +0.8%)
#
# If price reverses after partial:
# - Already took $0.06 profit
# - Remaining $12.50 hits break-even stop → $0 loss
# - Total: $0.06 profit (vs $0 or loss with all-or-nothing)
#
# This is what professional traders and profitable bots do.

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PartialExitLevel:
    """One exit level: take X% off at Y% profit."""
    profit_pct: float    # trigger: take profit when price moves this %
    exit_pct: float      # how much of remaining position to close (0-1)
    triggered: bool = False


@dataclass
class PartialExitState:
    """Tracks partial exit progress for one position."""
    pair: str
    side: str
    entry_price: float
    original_size: float
    remaining_size: float
    levels: list[PartialExitLevel]
    total_partial_pnl: float = 0.0


class PartialExitManager:
    """Manages staged profit taking across multiple positions.

    Default levels:
    - At +0.5% profit: close 50% of position
    - Remaining 50% rides with smart exit / trailing stop

    The partial exit is in ADDITION to the smart exit system.
    Smart exit manages the stop loss. Partial exit manages the take profit.
    """

    def __init__(self, levels: Optional[list[dict]] = None):
        if levels is None:
            # Default: take half off at +0.5%
            levels = [
                {"profit_pct": 0.5, "exit_pct": 0.5},
            ]

        self._default_levels = [
            PartialExitLevel(
                profit_pct=l["profit_pct"] / 100,
                exit_pct=l["exit_pct"],
            )
            for l in levels
        ]
        self._positions: dict[str, PartialExitState] = {}

    def register(self, pos_id: str, pair: str, side: str,
                 entry_price: float, position_size: float):
        """Register a position for partial profit taking."""
        # Deep copy the default levels for this position
        levels = [
            PartialExitLevel(
                profit_pct=l.profit_pct,
                exit_pct=l.exit_pct,
            )
            for l in self._default_levels
        ]

        self._positions[pos_id] = PartialExitState(
            pair=pair,
            side=side,
            entry_price=entry_price,
            original_size=position_size,
            remaining_size=position_size,
            levels=levels,
        )

    def check(self, pos_id: str, current_price: float) -> Optional[dict]:
        """Check if any partial exit level is triggered.

        Returns None if no action needed.
        Returns {"action": "partial_exit", "size": float, "pnl": float}
        if a partial exit should be executed.
        """
        state = self._positions.get(pos_id)
        if state is None:
            return None

        entry = state.entry_price

        # Calculate current profit percentage
        if state.side == "buy":
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry

        # Check each level
        for level in state.levels:
            if level.triggered:
                continue

            if profit_pct >= level.profit_pct:
                # Trigger this partial exit
                level.triggered = True

                # Calculate how much to close
                exit_size = state.remaining_size * level.exit_pct
                state.remaining_size -= exit_size

                # Calculate P&L on the partial
                pnl = exit_size * profit_pct

                # Account for fees (maker 0.075% each side = 0.15% round trip)
                fee = exit_size * 0.0015
                net_pnl = pnl - fee

                state.total_partial_pnl += net_pnl

                logger.info(
                    "PARTIAL EXIT: %s — took %.0f%% off at +%.2f%% profit. "
                    "Closed $%.2f, P&L $%.4f. Remaining: $%.2f",
                    state.pair, level.exit_pct * 100, profit_pct * 100,
                    exit_size, net_pnl, state.remaining_size,
                )

                return {
                    "action": "partial_exit",
                    "size": exit_size,
                    "pnl": net_pnl,
                    "remaining": state.remaining_size,
                    "profit_pct": profit_pct * 100,
                    "level_pct": level.exit_pct * 100,
                }

        return None

    def get_remaining_size(self, pos_id: str) -> float:
        """Get remaining position size after partial exits."""
        state = self._positions.get(pos_id)
        return state.remaining_size if state else 0.0

    def get_total_partial_pnl(self, pos_id: str) -> float:
        """Get total P&L from all partial exits for this position."""
        state = self._positions.get(pos_id)
        return state.total_partial_pnl if state else 0.0

    def remove(self, pos_id: str):
        """Remove a position (fully closed)."""
        self._positions.pop(pos_id, None)
