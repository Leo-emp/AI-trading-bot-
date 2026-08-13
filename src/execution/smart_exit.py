# src/execution/smart_exit.py
# Smart Exit Manager — maximizes profit, minimizes losses.
#
# The problem with fixed take-profit:
# - BTC moves +2% -> you exit at +0.8% -> missed $0.30 on a $25 position
# - Over 100 trades, that's $30 left on the table
#
# Smart exit solution:
# - Phase 1 (0% to +0.2%): Normal SL. If price reverses, take small loss.
# - Phase 2 (+0.2%): Move SL to entry. Trade is now RISK-FREE.
# - Phase 3 (+0.5%): Trailing stop activates. Locks in profit.
# - Phase 4 (+0.8% original TP): DON'T exit. Tighten trailing distance
#   from 0.3% to 0.15%. Let the winner run with a tight leash.
# - Exit only when trailing stop is hit (could be at +1%, +2%, +5%)
#
# This means:
# - Losses stay small (0.4% max, or $0 after break-even)
# - Wins can be MUCH bigger (no cap)
# - Average win grows while average loss stays the same or shrinks

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExitState:
    """Tracks the exit logic state for one position."""
    pair: str
    side: str
    entry_price: float
    original_sl: float
    original_tp: float
    current_sl: float
    highest_price: float    # for buy positions
    lowest_price: float     # for sell positions
    phase: int = 1          # 1=normal, 2=breakeven, 3=trailing, 4=tight trailing
    is_closed: bool = False
    close_reason: str = ""
    close_price: float = 0.0


class SmartExitManager:
    """Manages intelligent exits that let winners run and cut losers short.

    Replaces fixed TP with dynamic trailing that tightens as profit grows.
    """

    def __init__(self,
                 breakeven_pct: float = 0.2,
                 trail_activate_pct: float = 0.5,
                 trail_distance_pct: float = 0.3,
                 tight_trail_pct: float = 0.15,
                 tight_activate_pct: float = 0.8):
        self._breakeven_pct = breakeven_pct / 100
        self._trail_activate_pct = trail_activate_pct / 100
        self._trail_distance_pct = trail_distance_pct / 100
        self._tight_trail_pct = tight_trail_pct / 100
        self._tight_activate_pct = tight_activate_pct / 100
        self._positions: dict[str, ExitState] = {}

    def register(self, pos_id: str, pair: str, side: str,
                 entry: float, sl: float, tp: float):
        """Register a new position for smart exit management."""
        self._positions[pos_id] = ExitState(
            pair=pair, side=side,
            entry_price=entry,
            original_sl=sl, original_tp=tp,
            current_sl=sl,
            highest_price=entry,
            lowest_price=entry,
        )

    def update(self, pos_id: str, current_price: float) -> Optional[ExitState]:
        """Update position with latest price. Returns state (check is_closed)."""
        state = self._positions.get(pos_id)
        if state is None or state.is_closed:
            return state

        if state.side == "buy":
            self._update_buy(state, current_price)
        else:
            self._update_sell(state, current_price)

        return state

    def _update_buy(self, state: ExitState, price: float):
        """Smart exit logic for a long position."""
        # Track highest price
        if price > state.highest_price:
            state.highest_price = price

        entry = state.entry_price
        profit_pct = (price - entry) / entry

        # Check if stop loss hit
        if price <= state.current_sl:
            state.is_closed = True
            state.close_price = state.current_sl
            state.close_reason = f"phase{state.phase}_stop"
            logger.info("Smart exit: %s closed at phase %d stop (%.2f)",
                       state.pair, state.phase, state.current_sl)
            return

        # Phase transitions based on how much profit we have
        if state.phase == 1 and profit_pct >= self._breakeven_pct:
            # Move to phase 2: break-even
            state.phase = 2
            state.current_sl = entry
            logger.info("Smart exit: %s -> phase 2 (break-even at %.2f)",
                       state.pair, entry)

        if state.phase == 2 and profit_pct >= self._trail_activate_pct:
            # Move to phase 3: trailing stop
            state.phase = 3
            new_sl = price * (1 - self._trail_distance_pct)
            state.current_sl = max(state.current_sl, new_sl)
            logger.info("Smart exit: %s -> phase 3 (trailing at %.2f)",
                       state.pair, state.current_sl)

        if state.phase == 3 and profit_pct >= self._tight_activate_pct:
            # Move to phase 4: tight trailing (lock in more profit)
            state.phase = 4
            new_sl = price * (1 - self._tight_trail_pct)
            state.current_sl = max(state.current_sl, new_sl)
            logger.info("Smart exit: %s -> phase 4 (tight trail at %.2f)",
                       state.pair, state.current_sl)

        # Update trailing stop in phases 3 and 4
        if state.phase == 3:
            new_sl = state.highest_price * (1 - self._trail_distance_pct)
            state.current_sl = max(state.current_sl, new_sl)
        elif state.phase == 4:
            new_sl = state.highest_price * (1 - self._tight_trail_pct)
            state.current_sl = max(state.current_sl, new_sl)

    def _update_sell(self, state: ExitState, price: float):
        """Smart exit logic for a short position."""
        if price < state.lowest_price:
            state.lowest_price = price

        entry = state.entry_price
        profit_pct = (entry - price) / entry

        # Check stop loss
        if price >= state.current_sl:
            state.is_closed = True
            state.close_price = state.current_sl
            state.close_reason = f"phase{state.phase}_stop"
            return

        if state.phase == 1 and profit_pct >= self._breakeven_pct:
            state.phase = 2
            state.current_sl = entry

        if state.phase == 2 and profit_pct >= self._trail_activate_pct:
            state.phase = 3
            new_sl = price * (1 + self._trail_distance_pct)
            state.current_sl = min(state.current_sl, new_sl)

        if state.phase == 3 and profit_pct >= self._tight_activate_pct:
            state.phase = 4
            new_sl = price * (1 + self._tight_trail_pct)
            state.current_sl = min(state.current_sl, new_sl)

        if state.phase == 3:
            new_sl = state.lowest_price * (1 + self._trail_distance_pct)
            state.current_sl = min(state.current_sl, new_sl)
        elif state.phase == 4:
            new_sl = state.lowest_price * (1 + self._tight_trail_pct)
            state.current_sl = min(state.current_sl, new_sl)

    def remove(self, pos_id: str):
        self._positions.pop(pos_id, None)

    def get_current_sl(self, pos_id: str) -> Optional[float]:
        state = self._positions.get(pos_id)
        return state.current_sl if state else None

    def get_phase(self, pos_id: str) -> int:
        state = self._positions.get(pos_id)
        return state.phase if state else 0
