# src/execution/adaptive_sizer.py
# Adaptive Position Sizer — bets bigger when winning, smaller when losing.
#
# The idea: when the bot is on a winning streak, market conditions are
# favorable and the strategies are working. Increase size to capture more.
# When losing, conditions are against us. Reduce size to preserve capital.
#
# This is NOT martingale (doubling after losses — that blows up accounts).
# This is the OPPOSITE: scale UP winners, scale DOWN losers.
#
# Win streak:  +10% per consecutive win (capped at +50%)
# Loss streak: -20% per consecutive loss (capped at -60%)
# After 1 win following losses: reset to normal (not immediate full size)

import logging

logger = logging.getLogger(__name__)


class AdaptiveSizer:
    """Adjusts position size based on recent win/loss streaks.

    Works on top of the Kelly position sizer — this is a multiplier
    applied AFTER Kelly calculates the base position size.
    """

    def __init__(self,
                 win_bonus_pct: float = 10.0,
                 max_win_bonus_pct: float = 50.0,
                 loss_reduction_pct: float = 20.0,
                 max_loss_reduction_pct: float = 60.0):
        # How much to increase per consecutive win
        self._win_bonus = win_bonus_pct / 100
        self._max_win_bonus = max_win_bonus_pct / 100
        # How much to decrease per consecutive loss
        self._loss_reduction = loss_reduction_pct / 100
        self._max_loss_reduction = max_loss_reduction_pct / 100

        self._consecutive_wins = 0
        self._consecutive_losses = 0

    def record_win(self):
        """Record a winning trade."""
        self._consecutive_wins += 1
        self._consecutive_losses = 0
        logger.info("AdaptiveSizer: win streak %d, multiplier: %.2f",
                    self._consecutive_wins, self.get_multiplier())

    def record_loss(self):
        """Record a losing trade."""
        self._consecutive_losses += 1
        self._consecutive_wins = 0
        logger.info("AdaptiveSizer: loss streak %d, multiplier: %.2f",
                    self._consecutive_losses, self.get_multiplier())

    def get_multiplier(self) -> float:
        """Get the current position size multiplier.

        Returns: 0.4 to 1.5
        - 1.0 = normal size
        - >1.0 = winning streak, bet bigger
        - <1.0 = losing streak, bet smaller
        """
        if self._consecutive_wins > 0:
            bonus = min(
                self._consecutive_wins * self._win_bonus,
                self._max_win_bonus,
            )
            return 1.0 + bonus

        if self._consecutive_losses > 0:
            reduction = min(
                self._consecutive_losses * self._loss_reduction,
                self._max_loss_reduction,
            )
            return 1.0 - reduction

        return 1.0

    def adjust_size(self, base_size: float, min_size: float = 10.0) -> float:
        """Apply the multiplier to a base position size.

        Never goes below min_size (Binance $10 minimum).
        """
        adjusted = base_size * self.get_multiplier()
        return max(adjusted, min_size)

    def reset(self):
        """Reset streaks (e.g., on regime change)."""
        self._consecutive_wins = 0
        self._consecutive_losses = 0
