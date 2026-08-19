# src/intelligence/strategy_selector.py
# Strategy Selector — picks the RIGHT strategy for the RIGHT market.
#
# This is what makes the bot intelligent. Instead of running one
# strategy all the time, it:
# 1. Detects the market regime (trending, ranging, volatile)
# 2. Checks each strategy's recent performance
# 3. Picks the best strategy for current conditions
# 4. Falls back to the safest option when uncertain
#
# Regime -> Strategy mapping:
# - TRENDING_UP   -> momentum (ride the trend)
# - TRENDING_DOWN -> momentum (short the trend)
# - RANGING       -> grid + mean_reversion (buy low, sell high)
# - VOLATILE      -> smart_scalp (quick in/out with tight stops)

import logging
from typing import Optional

from src.intelligence.market_regime import MarketRegime
from src.intelligence.performance_tracker import PerformanceTracker
from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Which strategies work best in which regime (evidence-based mapping)
# Each strategy covers a DIFFERENT phase of the trend cycle:
# 1. grid (vol squeeze) = pre-trend compression -> first mover
# 2. smart_scalp (breakout) = new high/low -> trend start
# 3. momentum = EMA crossover -> trend confirmation
# 4. mean_reversion (pullback) = dip in trend -> continuation
REGIME_STRATEGY_MAP = {
    "TRENDING_UP": ["momentum", "smart_scalp", "mean_reversion"],
    "TRENDING_DOWN": ["momentum", "smart_scalp", "mean_reversion"],
    "RANGING": ["grid", "mean_reversion", "momentum"],  # squeeze + pullback + emerging trend
    "VOLATILE": ["smart_scalp", "grid"],               # breakout + squeeze (big moves)
}


class StrategySelector:
    """Selects the best strategy based on market regime + performance.

    Called every trading cycle to determine which strategy to use.
    """

    def __init__(self, strategies: dict[str, BaseStrategy],
                 performance_tracker: PerformanceTracker):
        self._strategies = strategies
        self._tracker = performance_tracker
        self._last_regime: Optional[str] = None

    def select(self, regime: MarketRegime) -> Optional[BaseStrategy]:
        """Pick the best strategy for the current market regime.

        Returns the strategy instance, or None if all strategies
        are disabled (extreme safety mode).
        """
        # Check for regime change — re-enable strategies on regime shift
        if self._last_regime and self._last_regime != regime.regime:
            logger.info("Regime changed: %s -> %s, re-enabling strategies",
                       self._last_regime, regime.regime)
            self._tracker.reset_all()
        self._last_regime = regime.regime

        # Get preferred strategies for this regime
        preferred = REGIME_STRATEGY_MAP.get(regime.regime, ["smart_scalp"])

        # Filter: must exist, be enabled in config, and not disabled by performance
        candidates = []
        for name in preferred:
            strategy = self._strategies.get(name)
            if strategy is None:
                continue
            if not self._tracker.is_strategy_enabled(name):
                logger.debug("Strategy %s disabled by performance tracker", name)
                continue
            candidates.append(strategy)

        if not candidates:
            # All preferred strategies disabled — fall back to smart_scalp
            fallback = self._strategies.get("smart_scalp")
            if fallback and self._tracker.is_strategy_enabled("smart_scalp"):
                logger.warning("All preferred strategies disabled, falling back to smart_scalp")
                return fallback
            logger.warning("ALL strategies disabled — no trading this cycle")
            return None

        # If only one candidate, use it
        if len(candidates) == 1:
            logger.info("Selected strategy: %s (only candidate for %s)",
                       candidates[0].name, regime.regime)
            return candidates[0]

        # Multiple candidates — pick by performance ranking
        best = None
        best_score = -1
        for strategy in candidates:
            # Combine win rate + avg P&L for ranking
            win_rate = self._tracker.get_win_rate(strategy.name)
            avg_pnl = self._tracker.get_avg_pnl(strategy.name)
            # Weighted score: 70% win rate, 30% profitability
            score = win_rate * 0.7 + (1.0 if avg_pnl > 0 else 0.0) * 0.3
            if score > best_score:
                best_score = score
                best = strategy

        if best:
            logger.info("Selected strategy: %s (score: %.2f) for regime %s",
                       best.name, best_score, regime.regime)
        return best

    def get_all_suitable(self, regime: MarketRegime) -> list[BaseStrategy]:
        """Get ALL suitable strategies for the regime (for multi-strategy mode)."""
        preferred = REGIME_STRATEGY_MAP.get(regime.regime, ["smart_scalp"])
        result = []
        for name in preferred:
            strategy = self._strategies.get(name)
            if strategy and self._tracker.is_strategy_enabled(name):
                result.append(strategy)
        return result
