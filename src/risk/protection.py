# src/risk/protection.py
# 5-layer autonomous protection system.
# Each layer acts independently — faster layers override slower ones.
#
# Layer 1: Per-trade (stop-loss, trailing stop) — handled by executor
# Layer 2: Session (consecutive losses, daily drawdown)
# Layer 3: Portfolio (weekly/monthly drawdown) — future: needs DB snapshots
# Layer 4: Black swan (flash crash, balance floor)
# Layer 5: Infrastructure (reconnection, restart) — handled by watchdog
#
# FIXED: Constructor takes individual params (not db+settings).
# Method is check() not check_all_layers(). Sync, not async.
# Tracks consecutive losses and daily P&L internally.

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtectionAction:
    """What the protection system instructs the engine to do.

    Actions ranked by severity:
    CONTINUE → REDUCE_SIZE → PAUSE → DEFENSE_ONLY → SHUTDOWN → EMERGENCY_SELL
    """
    action: str         # CONTINUE, REDUCE_SIZE, PAUSE, DEFENSE_ONLY, SHUTDOWN
    reason: str
    duration_minutes: int = 0
    size_multiplier: float = 1.0


class ProtectionSystem:
    """Multi-layer autonomous protection.

    Called by the engine on every cycle. Returns the most severe
    protection action from all layers. The engine must obey.
    """

    def __init__(self,
                 max_consecutive_losses: int = 5,
                 daily_drawdown_limit: float = 5.0,
                 weekly_drawdown_limit: float = 10.0,
                 monthly_drawdown_limit: float = 20.0,
                 balance_floor_pct: float = 50.0):
        # How many losses in a row before pausing
        self._max_consec_losses = max_consecutive_losses
        # Daily drawdown % that triggers shutdown
        self._daily_dd_limit = daily_drawdown_limit
        # Weekly/monthly limits (tracked when DB snapshots available)
        self._weekly_dd_limit = weekly_drawdown_limit
        self._monthly_dd_limit = monthly_drawdown_limit
        # If balance drops below this % of initial, shutdown
        self._balance_floor_pct = balance_floor_pct

        # Internal tracking (no DB dependency)
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._daily_start_balance = 0.0
        self._current_day = ""
        self._reduce_threshold = max(max_consecutive_losses - 2, 2)

    def record_trade_result(self, pnl: float):
        """Record a trade result for consecutive loss tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._daily_pnl += pnl

    def check(self, current_balance: float, initial_balance: float) -> ProtectionAction:
        """Run all protection layers and return the most severe action.

        Args:
            current_balance: current USDT balance
            initial_balance: starting balance (for floor calculation)
        """
        # Reset daily tracking at day boundary
        import datetime as dt
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._current_day = today
            self._daily_pnl = 0.0
            self._daily_start_balance = current_balance

        # --- Layer 4: Balance floor (most critical) ---
        floor = initial_balance * (self._balance_floor_pct / 100)
        if current_balance < floor:
            logger.critical("LAYER 4: Balance $%.2f below floor $%.2f — SHUTDOWN",
                          current_balance, floor)
            return ProtectionAction(
                action="SHUTDOWN",
                reason=f"Balance ${current_balance:.2f} below {self._balance_floor_pct:.0f}% floor (${floor:.2f})",
            )

        # --- Layer 2: Daily drawdown ---
        if self._daily_start_balance > 0 and self._daily_pnl < 0:
            daily_dd_pct = abs(self._daily_pnl) / self._daily_start_balance * 100

            if daily_dd_pct >= self._daily_dd_limit:
                logger.critical("LAYER 2: Daily drawdown %.1f%% >= %.1f%% — SHUTDOWN",
                              daily_dd_pct, self._daily_dd_limit)
                return ProtectionAction(
                    action="SHUTDOWN",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% exceeded {self._daily_dd_limit}% limit",
                )

            # Defense mode at 60% of daily limit
            defense_threshold = self._daily_dd_limit * 0.6
            if daily_dd_pct >= defense_threshold:
                logger.warning("LAYER 2: Daily DD %.1f%% — defense mode", daily_dd_pct)
                return ProtectionAction(
                    action="REDUCE_SIZE",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% — reducing position sizes",
                    size_multiplier=0.5,
                )

        # --- Layer 2: Consecutive losses ---
        if self._consecutive_losses >= self._max_consec_losses:
            logger.warning("LAYER 2: %d consecutive losses — PAUSE",
                          self._consecutive_losses)
            return ProtectionAction(
                action="PAUSE",
                reason=f"{self._consecutive_losses} consecutive losses — cooling off",
                duration_minutes=30,
            )

        if self._consecutive_losses >= self._reduce_threshold:
            logger.info("LAYER 2: %d consecutive losses — reducing size",
                       self._consecutive_losses)
            return ProtectionAction(
                action="REDUCE_SIZE",
                reason=f"{self._consecutive_losses} consecutive losses — smaller positions",
                size_multiplier=0.5,
            )

        # --- All clear ---
        return ProtectionAction(action="CONTINUE", reason="all protection layers clear")

    def reset_losses(self):
        """Reset consecutive loss counter (e.g. after regime change)."""
        self._consecutive_losses = 0
