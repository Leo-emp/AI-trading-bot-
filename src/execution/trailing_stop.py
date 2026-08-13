# src/execution/trailing_stop.py
# Trailing Stop Manager — locks in profits as price moves favorably.
#
# The problem with fixed take-profit:
# - Price hits TP → you exit with 0.8% profit
# - Price continues up another 3% → you missed out
#
# Trailing stop solution:
# - Set initial SL as normal
# - Once price moves X% in your favor, activate trailing stop
# - Trailing stop follows price at a fixed distance
# - If price reverses, you exit with profits locked in
# - If price keeps going, trailing stop keeps moving up
#
# This lets winners run while still protecting profits.

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """State for a single trailing stop."""
    pair: str
    side: str                    # "buy" or "sell"
    entry_price: float
    initial_stop: float          # original stop loss
    current_stop: float          # current trailing stop level
    activation_price: float      # price at which trailing activates
    trail_distance_pct: float    # trailing distance as %
    is_activated: bool = False   # whether trailing has started
    highest_price: float = 0.0   # highest price seen (for buy)
    lowest_price: float = 999999.0  # lowest price seen (for sell)


class TrailingStopManager:
    """Manages trailing stops for all open positions.

    Activation rules:
    - Trailing stop activates after price moves X% in your favor
    - Once activated, the stop follows at a fixed % distance
    - The stop only moves in the favorable direction (never backwards)

    Example for a BUY at $50,000:
    - Activation at +0.5% → $50,250
    - Trail distance: 0.3%
    - Price hits $50,400 → stop moves to $50,400 * (1 - 0.003) = $50,249
    - Price hits $50,600 → stop moves to $50,600 * (1 - 0.003) = $50,448
    - Price drops to $50,448 → STOP HIT, exit with profit
    """

    def __init__(self, activation_pct: float = 0.5,
                 trail_distance_pct: float = 0.3):
        # Price must move this % in your favor to activate trailing
        self._activation_pct = activation_pct / 100
        # Trailing stop distance from highest/lowest price
        self._trail_distance_pct = trail_distance_pct / 100
        # Active trailing stops: position_id → TrailingStopState
        self._stops: dict[str, TrailingStopState] = {}

    def register(self, position_id: str, pair: str, side: str,
                 entry_price: float, initial_stop: float):
        """Register a new position for trailing stop management."""
        if side == "buy":
            activation = entry_price * (1 + self._activation_pct)
        else:
            activation = entry_price * (1 - self._activation_pct)

        self._stops[position_id] = TrailingStopState(
            pair=pair,
            side=side,
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=initial_stop,  # start at initial SL
            activation_price=activation,
            trail_distance_pct=self._trail_distance_pct,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        logger.debug("Trailing stop registered for %s %s @ %.2f, activates at %.2f",
                     side, pair, entry_price, activation)

    def update(self, position_id: str, current_price: float) -> Optional[float]:
        """Update trailing stop with new price. Returns current stop level.

        Returns None if position not found.
        """
        state = self._stops.get(position_id)
        if state is None:
            return None

        if state.side == "buy":
            return self._update_buy(state, current_price)
        else:
            return self._update_sell(state, current_price)

    def _update_buy(self, state: TrailingStopState, price: float) -> float:
        """Update trailing stop for a long position."""
        # Track highest price seen
        if price > state.highest_price:
            state.highest_price = price

        # Check if trailing should activate
        if not state.is_activated:
            if price >= state.activation_price:
                state.is_activated = True
                # Set trailing stop
                new_stop = price * (1 - state.trail_distance_pct)
                state.current_stop = max(state.current_stop, new_stop)
                logger.info("Trailing stop ACTIVATED for %s at %.2f, stop: %.2f",
                           state.pair, price, state.current_stop)
        else:
            # Trailing is active — move stop up if price made new high
            new_stop = state.highest_price * (1 - state.trail_distance_pct)
            if new_stop > state.current_stop:
                state.current_stop = new_stop
                logger.debug("Trailing stop moved up to %.2f (price: %.2f)",
                           state.current_stop, price)

        return state.current_stop

    def _update_sell(self, state: TrailingStopState, price: float) -> float:
        """Update trailing stop for a short position."""
        if price < state.lowest_price:
            state.lowest_price = price

        if not state.is_activated:
            if price <= state.activation_price:
                state.is_activated = True
                new_stop = price * (1 + state.trail_distance_pct)
                state.current_stop = min(state.current_stop, new_stop)
                logger.info("Trailing stop ACTIVATED for short %s at %.2f, stop: %.2f",
                           state.pair, price, state.current_stop)
        else:
            new_stop = state.lowest_price * (1 + state.trail_distance_pct)
            if new_stop < state.current_stop:
                state.current_stop = new_stop
                logger.debug("Trailing stop moved down to %.2f (price: %.2f)",
                           state.current_stop, price)

        return state.current_stop

    def remove(self, position_id: str):
        """Remove a trailing stop (position closed)."""
        self._stops.pop(position_id, None)

    def get_stop(self, position_id: str) -> Optional[float]:
        """Get current stop level for a position."""
        state = self._stops.get(position_id)
        return state.current_stop if state else None

    def is_activated(self, position_id: str) -> bool:
        """Check if trailing stop has been activated."""
        state = self._stops.get(position_id)
        return state.is_activated if state else False
