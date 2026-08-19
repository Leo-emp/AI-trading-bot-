# src/execution/smart_exit.py
# Smart Exit Manager — let winners run, cut losers at 1R.
#
# PROFIT-FIRST EXIT SYSTEM:
# Old system: 50% partial at 1R, BE at 1.5R → avg win 0.5R, avg loss 1R = LOSING
# New system: 25% partial at 2R, BE at 1R → avg win 2-4R, avg loss 1R = WINNING
#
# Phase flow:
# - Phase 1 (0 to +1R): Normal SL. If price reverses, take the 1R loss.
# - Phase 2 (+1R): Move SL to entry + 0.25R. Trade is risk-free.
#   +0.25R buffer (not exact BE) survives crypto noise without stopping out.
# - Phase 3 (+2.5R): Trailing stop activates at 1.0R behind price.
#   25% partial already taken at 2R — remaining 75% rides the trend.
# - Phase 4 (+4R): Tighten trailing to 0.75R behind. Lock in big profit.
#
# Why +0.25R not +0.10R: crypto swings are wider than forex.
# 0.10R buffer got stopped constantly on normal retracement.

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
    # Per-position dynamic thresholds (R-based, set at registration)
    breakeven_pct: float = 0.2
    trail_activate_pct: float = 0.5
    trail_distance_pct: float = 0.3
    tight_trail_pct: float = 0.15
    tight_activate_pct: float = 0.8
    # Initial risk distance: abs(entry - original_sl)
    # Used for BE protection: SL moves to entry + 0.10 * initial_risk
    initial_risk: float = 0.0


class SmartExitManager:
    """Manages intelligent exits that let winners run to 3R-5R.

    BE at +0.25R (at 1R profit) makes trades risk-free early.
    Wide trailing (1.0R behind) gives crypto moves room to breathe.
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
                 entry: float, sl: float, tp: float,
                 breakeven_pct: float = None,
                 trail_activate_pct: float = None,
                 tight_activate_pct: float = None,
                 trail_distance_pct: float = None,
                 tight_trail_pct: float = None):
        """Register a new position for smart exit management.

        Calculates initial_risk = abs(entry - sl) for R-based BE protection.
        """
        # Calculate initial risk distance (1R in price terms)
        initial_risk = abs(entry - sl)

        state = ExitState(
            pair=pair, side=side,
            entry_price=entry,
            original_sl=sl, original_tp=tp,
            current_sl=sl,
            highest_price=entry,
            lowest_price=entry,
            initial_risk=initial_risk,
        )
        # Apply per-position dynamic thresholds if provided
        if breakeven_pct is not None:
            state.breakeven_pct = breakeven_pct
        if trail_activate_pct is not None:
            state.trail_activate_pct = trail_activate_pct
        if tight_activate_pct is not None:
            state.tight_activate_pct = tight_activate_pct
        if trail_distance_pct is not None:
            state.trail_distance_pct = trail_distance_pct
        if tight_trail_pct is not None:
            state.tight_trail_pct = tight_trail_pct

        self._positions[pos_id] = state

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
        """Smart exit logic for a long position.

        BE at +0.10R (not exact entry) prevents normal retracement stops.
        """
        # Track highest price
        if price > state.highest_price:
            state.highest_price = price

        entry = state.entry_price
        profit_pct = (price - entry) / entry

        # Per-position thresholds (R-based, set at registration)
        be_pct = state.breakeven_pct / 100
        trail_pct = state.trail_activate_pct / 100
        tight_pct = state.tight_activate_pct / 100
        trail_dist = state.trail_distance_pct / 100
        tight_dist = state.tight_trail_pct / 100

        # Check if stop loss hit
        if price <= state.current_sl:
            state.is_closed = True
            state.close_price = state.current_sl
            state.close_reason = f"phase{state.phase}_stop"
            logger.info("Smart exit: %s closed at phase %d stop (%.2f)",
                       state.pair, state.phase, state.current_sl)
            return

        # Phase transitions based on how much profit we have
        if state.phase == 1 and profit_pct >= be_pct:
            state.phase = 2
            # BE PROTECTION: move SL to entry + 0.25R
            # 0.25R buffer survives crypto volatility (0.10R was too tight)
            be_buffer = state.initial_risk * 0.25
            new_sl = entry + be_buffer
            state.current_sl = new_sl
            logger.info("Smart exit: %s -> phase 2 (BE at %.2f, +0.25R buffer)",
                       state.pair, new_sl)

        if state.phase == 2 and profit_pct >= trail_pct:
            state.phase = 3
            new_sl = price * (1 - trail_dist)
            state.current_sl = max(state.current_sl, new_sl)
            logger.info("Smart exit: %s -> phase 3 (trailing at %.2f)",
                       state.pair, state.current_sl)

        if state.phase == 3 and profit_pct >= tight_pct:
            state.phase = 4
            new_sl = price * (1 - tight_dist)
            state.current_sl = max(state.current_sl, new_sl)
            logger.info("Smart exit: %s -> phase 4 (tight trail at %.2f)",
                       state.pair, state.current_sl)

        # Update trailing stop in phases 3 and 4
        if state.phase == 3:
            new_sl = state.highest_price * (1 - trail_dist)
            state.current_sl = max(state.current_sl, new_sl)
        elif state.phase == 4:
            new_sl = state.highest_price * (1 - tight_dist)
            state.current_sl = max(state.current_sl, new_sl)

    def _update_sell(self, state: ExitState, price: float):
        """Smart exit logic for a short position.

        BE at -0.10R (below entry) for shorts.
        """
        if price < state.lowest_price:
            state.lowest_price = price

        entry = state.entry_price
        profit_pct = (entry - price) / entry

        # Per-position thresholds
        be_pct = state.breakeven_pct / 100
        trail_pct = state.trail_activate_pct / 100
        tight_pct = state.tight_activate_pct / 100
        trail_dist = state.trail_distance_pct / 100
        tight_dist = state.tight_trail_pct / 100

        # Check stop loss
        if price >= state.current_sl:
            state.is_closed = True
            state.close_price = state.current_sl
            state.close_reason = f"phase{state.phase}_stop"
            return

        if state.phase == 1 and profit_pct >= be_pct:
            state.phase = 2
            # BE PROTECTION for shorts: move SL to entry - 0.25R
            be_buffer = state.initial_risk * 0.25
            new_sl = entry - be_buffer
            state.current_sl = new_sl
            logger.info("Smart exit: %s -> phase 2 (BE at %.2f, -0.25R buffer)",
                       state.pair, new_sl)

        if state.phase == 2 and profit_pct >= trail_pct:
            state.phase = 3
            new_sl = price * (1 + trail_dist)
            state.current_sl = min(state.current_sl, new_sl)

        if state.phase == 3 and profit_pct >= tight_pct:
            state.phase = 4
            new_sl = price * (1 + tight_dist)
            state.current_sl = min(state.current_sl, new_sl)

        if state.phase == 3:
            new_sl = state.lowest_price * (1 + trail_dist)
            state.current_sl = min(state.current_sl, new_sl)
        elif state.phase == 4:
            new_sl = state.lowest_price * (1 + tight_dist)
            state.current_sl = min(state.current_sl, new_sl)

    def remove(self, pos_id: str):
        self._positions.pop(pos_id, None)

    def get_current_sl(self, pos_id: str) -> Optional[float]:
        state = self._positions.get(pos_id)
        return state.current_sl if state else None

    def get_phase(self, pos_id: str) -> int:
        state = self._positions.get(pos_id)
        return state.phase if state else 0
