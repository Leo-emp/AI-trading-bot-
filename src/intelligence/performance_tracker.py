# src/intelligence/performance_tracker.py
# Strategy Performance Tracker — learns which strategies are winning.
#
# Tracks every trade by strategy and maintains rolling statistics:
# - Win rate (over last N trades)
# - Average profit per trade
# - Average loss per trade
# - Profit factor (gross profit / gross loss)
# - Consecutive wins/losses
#
# Auto-disables strategies that are consistently losing.
# Re-enables them when market conditions change (regime shift).
#
# This is how the bot "learns" without ML — pure performance data.

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Rolling performance metrics for one strategy."""
    strategy_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    recent_pnls: list[float] = field(default_factory=list)
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    is_disabled: bool = False
    disabled_reason: str = ""


class PerformanceTracker:
    """Tracks and ranks strategy performance over time.

    Every closed trade is recorded here. The strategy selector
    uses this data to pick the best strategy for current conditions.
    """

    def __init__(self, window: int = 30,
                 min_win_rate: float = 0.40,
                 max_consecutive_losses: int = 5):
        # Rolling window for recent performance
        self._window = window
        # Disable strategy if win rate drops below this
        self._min_win_rate = min_win_rate
        # Disable after this many losses in a row
        self._max_consecutive_losses = max_consecutive_losses
        # Performance per strategy
        self._performance: dict[str, StrategyPerformance] = {}

    def record_trade(self, strategy_name: str, pnl: float):
        """Record a completed trade's P&L for a strategy."""
        perf = self._get_or_create(strategy_name)
        perf.total_trades += 1
        perf.recent_pnls.append(pnl)

        # Keep only the rolling window
        if len(perf.recent_pnls) > self._window:
            perf.recent_pnls = perf.recent_pnls[-self._window:]

        if pnl > 0:
            perf.wins += 1
            perf.total_profit += pnl
            perf.consecutive_wins += 1
            perf.consecutive_losses = 0
        else:
            perf.losses += 1
            perf.total_loss += abs(pnl)
            perf.consecutive_losses += 1
            perf.consecutive_wins = 0

        # Auto-disable check
        self._check_auto_disable(perf)

        logger.info(
            "Strategy %s: trade #%d, P&L: $%.4f, win rate: %.1f%%, streak: %s",
            strategy_name, perf.total_trades, pnl,
            self.get_win_rate(strategy_name) * 100,
            f"+{perf.consecutive_wins}" if perf.consecutive_wins > 0
            else f"-{perf.consecutive_losses}",
        )

    def get_win_rate(self, strategy_name: str) -> float:
        """Get rolling win rate for a strategy."""
        perf = self._performance.get(strategy_name)
        if not perf or not perf.recent_pnls:
            return 0.5  # assume 50% if no data yet

        wins = sum(1 for p in perf.recent_pnls if p > 0)
        return wins / len(perf.recent_pnls)

    def get_avg_pnl(self, strategy_name: str) -> float:
        """Get average P&L per trade for a strategy."""
        perf = self._performance.get(strategy_name)
        if not perf or not perf.recent_pnls:
            return 0.0
        return sum(perf.recent_pnls) / len(perf.recent_pnls)

    def get_profit_factor(self, strategy_name: str) -> float:
        """Get profit factor (gross profit / gross loss)."""
        perf = self._performance.get(strategy_name)
        if not perf or perf.total_loss == 0:
            return 1.0
        return perf.total_profit / perf.total_loss

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is currently enabled."""
        perf = self._performance.get(strategy_name)
        if not perf:
            return True  # new strategies are enabled by default
        return not perf.is_disabled

    def get_rankings(self) -> list[tuple[str, float]]:
        """Rank strategies by recent performance. Higher = better."""
        rankings = []
        for name, perf in self._performance.items():
            if perf.total_trades < 3:
                score = 0.5  # neutral score for new strategies
            else:
                win_rate = self.get_win_rate(name)
                avg_pnl = self.get_avg_pnl(name)
                # Score = win rate * 60% + normalized avg P&L * 40%
                pnl_score = min(max(avg_pnl * 100 + 0.5, 0), 1)  # normalize
                score = win_rate * 0.6 + pnl_score * 0.4
            rankings.append((name, score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def reset_strategy(self, strategy_name: str):
        """Re-enable a disabled strategy (e.g., after regime change)."""
        perf = self._performance.get(strategy_name)
        if perf:
            perf.is_disabled = False
            perf.disabled_reason = ""
            perf.consecutive_losses = 0
            logger.info("Strategy %s re-enabled", strategy_name)

    def reset_all(self):
        """Re-enable all strategies (e.g., after major regime change)."""
        for name in self._performance:
            self.reset_strategy(name)

    def _get_or_create(self, strategy_name: str) -> StrategyPerformance:
        """Get or create performance record for a strategy."""
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )
        return self._performance[strategy_name]

    def _check_auto_disable(self, perf: StrategyPerformance):
        """Disable strategy if it's consistently losing."""
        if perf.total_trades < 5:
            return  # need minimum trades before disabling

        win_rate = self.get_win_rate(perf.strategy_name)

        # Disable if win rate too low
        if win_rate < self._min_win_rate:
            perf.is_disabled = True
            perf.disabled_reason = f"win rate {win_rate*100:.1f}% < {self._min_win_rate*100:.0f}%"
            logger.warning("Strategy %s DISABLED: %s", perf.strategy_name, perf.disabled_reason)

        # Disable on consecutive loss streak
        if perf.consecutive_losses >= self._max_consecutive_losses:
            perf.is_disabled = True
            perf.disabled_reason = f"{perf.consecutive_losses} consecutive losses"
            logger.warning("Strategy %s DISABLED: %s", perf.strategy_name, perf.disabled_reason)
