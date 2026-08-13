# src/risk/protection.py
# 5-layer autonomous protection system.
# Each layer acts independently — faster layers override slower ones.
# The system checks all layers on every trading cycle and returns
# the most severe action needed.
#
# Layer 1: Per-trade (stop-loss, trailing stop) — handled by executor
# Layer 2: Session (consecutive losses, daily drawdown)
# Layer 3: Portfolio (weekly/monthly drawdown)
# Layer 4: Black swan (flash crash, API errors, balance floor)
# Layer 5: Infrastructure (reconnection, restart) — handled by watchdog

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtectionAction:
    """What the protection system instructs the engine to do.

    Actions ranked by severity:
    CONTINUE → REDUCE_SIZE → PAUSE → DEFENSE_ONLY → SHUTDOWN → EMERGENCY_SELL
    """
    action: str         # CONTINUE, REDUCE_SIZE, PAUSE, DEFENSE_ONLY, SHUTDOWN, EMERGENCY_SELL
    reason: str         # human-readable explanation
    duration_minutes: int = 0  # how long to pause/reduce (0 = until reset)
    size_multiplier: float = 1.0  # for REDUCE_SIZE: multiply position by this


class ProtectionSystem:
    """Multi-layer autonomous protection.

    Called by the engine on every cycle. Returns the most severe
    protection action from all layers. The engine must obey.
    """

    def __init__(self, db, settings: dict):
        self._db = db
        self._settings = settings
        self._protection = settings.get("protection", {})
        self._risk = settings.get("risk", {})

    async def check_all_layers(self, balance: float, daily_pnl: float) -> ProtectionAction:
        """Run all protection layers and return the most severe action.

        Checks layers 2-4 (layer 1 is per-trade, layer 5 is infra).
        Returns the highest-severity action found.
        """
        # Start with the most severe checks first

        # --- Layer 4: Black swan / balance floor ---
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if balance < min_balance:
            logger.critical("LAYER 4: Balance $%.2f below floor $%.2f — SHUTDOWN", balance, min_balance)
            return ProtectionAction(
                action="SHUTDOWN",
                reason=f"Balance ${balance:.2f} below minimum floor ${min_balance:.2f}",
            )

        # --- Layer 2: Daily drawdown checks ---
        # Calculate daily drawdown as percentage
        # daily_pnl is negative when losing money
        starting_balance = balance - daily_pnl  # what we started the day with
        if starting_balance > 0 and daily_pnl < 0:
            daily_dd_pct = abs(daily_pnl) / starting_balance * 100

            shutdown_threshold = self._risk.get("max_daily_drawdown_pct", 5.0)
            if daily_dd_pct >= shutdown_threshold:
                logger.critical("LAYER 2: Daily drawdown %.1f%% >= %.1f%% — SHUTDOWN", daily_dd_pct, shutdown_threshold)
                return ProtectionAction(
                    action="SHUTDOWN",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% exceeded {shutdown_threshold}% limit",
                )

            defense_threshold = self._protection.get("defense_mode_drawdown_pct", 3.0)
            if daily_dd_pct >= defense_threshold:
                logger.warning("LAYER 2: Daily drawdown %.1f%% — switching to defense-only", daily_dd_pct)
                return ProtectionAction(
                    action="DEFENSE_ONLY",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% — defense-only mode",
                )

        # --- Layer 2: Consecutive loss checks ---
        consecutive_losses = await self._db.get_consecutive_losses()

        pause_threshold = self._protection.get("pause_after_losses", 5)
        if consecutive_losses >= pause_threshold:
            pause_minutes = self._risk.get("consecutive_loss_pause_minutes", 30)
            logger.warning("LAYER 2: %d consecutive losses — pausing %d min", consecutive_losses, pause_minutes)
            return ProtectionAction(
                action="PAUSE",
                reason=f"{consecutive_losses} consecutive losses — cooling off",
                duration_minutes=pause_minutes,
            )

        reduce_threshold = self._protection.get("reduce_size_after_losses", 3)
        if consecutive_losses >= reduce_threshold:
            logger.info("LAYER 2: %d consecutive losses — reducing size 50%%", consecutive_losses)
            return ProtectionAction(
                action="REDUCE_SIZE",
                reason=f"{consecutive_losses} consecutive losses — reducing position sizes",
                size_multiplier=0.5,
            )

        # --- Layer 3: Weekly/monthly drawdown ---
        weekly_dd = await self._db.get_weekly_drawdown()
        weekly_threshold = self._protection.get("weekly_drawdown_reduce_pct", 10.0)
        if weekly_dd >= weekly_threshold:
            logger.warning("LAYER 3: Weekly drawdown %.1f%% — reducing sizes", weekly_dd)
            return ProtectionAction(
                action="REDUCE_SIZE",
                reason=f"Weekly drawdown {weekly_dd:.1f}% — protective size reduction",
                size_multiplier=0.5,
            )

        monthly_dd = await self._db.get_monthly_drawdown()
        monthly_threshold = self._protection.get("monthly_drawdown_emergency_pct", 15.0)
        if monthly_dd >= monthly_threshold:
            logger.critical("LAYER 3: Monthly drawdown %.1f%% — EMERGENCY paper-only", monthly_dd)
            return ProtectionAction(
                action="SHUTDOWN",
                reason=f"Monthly drawdown {monthly_dd:.1f}% — emergency shutdown, switch to paper trading",
            )

        # --- All clear ---
        return ProtectionAction(action="CONTINUE", reason="all protection layers clear")
