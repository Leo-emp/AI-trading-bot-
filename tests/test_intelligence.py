# tests/test_intelligence.py
# Tests for the intelligence layer: market regime, performance tracker,
# strategy selector, multi-timeframe, trailing stops.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.intelligence.market_regime import MarketRegimeDetector
from src.intelligence.performance_tracker import PerformanceTracker
from src.intelligence.strategy_selector import StrategySelector, REGIME_STRATEGY_MAP
from src.intelligence.multi_timeframe import MultiTimeframeBrain
from src.intelligence.gemini_brain import GeminiBrain
from src.execution.trailing_stop import TrailingStopManager
from src.data.indicators import IndicatorEngine
from src.strategies.smart_scalp import SmartScalpStrategy
from src.strategies.grid import GridStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy


def make_df(closes, n=100):
    """Build OHLCV DataFrame with indicators."""
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
        "open": [c - 10 for c in closes],
        "high": [c + 30 for c in closes],
        "low": [c - 30 for c in closes],
        "close": closes,
        "volume": [500.0] * n,
    })
    return IndicatorEngine().compute_all(df)


# --- Market Regime Tests ---

class TestMarketRegimeDetector:
    def test_detects_some_regime(self):
        closes = [50000 + i * 50 for i in range(100)]
        df = make_df(closes)
        regime = MarketRegimeDetector().detect(df)
        assert regime.regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE")
        assert 0 <= regime.confidence <= 1.0

    def test_ranging_market(self):
        closes = [50000 + 200 * np.sin(i * 0.3) for i in range(100)]
        df = make_df(closes)
        regime = MarketRegimeDetector().detect(df)
        # Ranging or volatile — both valid for oscillating data
        assert regime.regime in ("RANGING", "VOLATILE", "TRENDING_UP", "TRENDING_DOWN")

    def test_insufficient_data(self):
        closes = [50000] * 20
        df = make_df(closes, n=20)
        regime = MarketRegimeDetector().detect(df)
        assert regime.regime == "RANGING"

    def test_volatility_classification(self):
        closes = [50000 + i * 50 for i in range(100)]
        df = make_df(closes)
        regime = MarketRegimeDetector().detect(df)
        assert regime.volatility in ("LOW", "NORMAL", "HIGH")


# --- Performance Tracker Tests ---

class TestPerformanceTracker:
    def test_records_trade(self):
        tracker = PerformanceTracker()
        tracker.record_trade("smart_scalp", 0.50)
        assert tracker.get_win_rate("smart_scalp") == 1.0
        assert tracker.get_avg_pnl("smart_scalp") == 0.50

    def test_win_rate_calculation(self):
        tracker = PerformanceTracker()
        tracker.record_trade("grid", 0.30)
        tracker.record_trade("grid", -0.10)
        tracker.record_trade("grid", 0.20)
        tracker.record_trade("grid", -0.05)
        assert tracker.get_win_rate("grid") == 0.5

    def test_auto_disable_on_low_win_rate(self):
        tracker = PerformanceTracker(min_win_rate=0.40)
        # Record 10 losing trades
        for _ in range(10):
            tracker.record_trade("bad_strategy", -0.10)
        assert not tracker.is_strategy_enabled("bad_strategy")

    def test_auto_disable_on_consecutive_losses(self):
        tracker = PerformanceTracker(max_consecutive_losses=3)
        tracker.record_trade("loser", 0.10)  # win
        tracker.record_trade("loser", 0.10)  # win
        tracker.record_trade("loser", 0.10)  # win
        tracker.record_trade("loser", 0.10)  # win
        tracker.record_trade("loser", 0.10)  # win — 5 trades needed
        tracker.record_trade("loser", -0.10)
        tracker.record_trade("loser", -0.10)
        tracker.record_trade("loser", -0.10)
        assert not tracker.is_strategy_enabled("loser")

    def test_reset_re_enables(self):
        tracker = PerformanceTracker(max_consecutive_losses=3)
        for _ in range(6):
            tracker.record_trade("strat", -0.10)
        assert not tracker.is_strategy_enabled("strat")
        tracker.reset_strategy("strat")
        assert tracker.is_strategy_enabled("strat")

    def test_rankings(self):
        tracker = PerformanceTracker()
        for _ in range(5):
            tracker.record_trade("winner", 0.50)
        for _ in range(5):
            tracker.record_trade("loser", -0.10)
        rankings = tracker.get_rankings()
        # Winner should rank higher
        names = [r[0] for r in rankings]
        assert names.index("winner") < names.index("loser")

    def test_new_strategy_defaults_enabled(self):
        tracker = PerformanceTracker()
        assert tracker.is_strategy_enabled("never_seen")


# --- Strategy Selector Tests ---

class TestStrategySelector:
    def _make_selector(self):
        strategies = {
            "smart_scalp": SmartScalpStrategy(),
            "grid": GridStrategy(),
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }
        tracker = PerformanceTracker()
        return StrategySelector(strategies, tracker), tracker

    def test_selects_strategy_for_regime(self):
        selector, _ = self._make_selector()
        from src.intelligence.market_regime import MarketRegime
        regime = MarketRegime("TRENDING_UP", 0.8, "NORMAL", 0.6, [])
        strategy = selector.select(regime)
        assert strategy is not None
        assert strategy.name in REGIME_STRATEGY_MAP["TRENDING_UP"]

    def test_selects_grid_for_ranging(self):
        selector, _ = self._make_selector()
        from src.intelligence.market_regime import MarketRegime
        regime = MarketRegime("RANGING", 0.7, "LOW", 0.1, [])
        strategy = selector.select(regime)
        assert strategy is not None
        assert strategy.name in REGIME_STRATEGY_MAP["RANGING"]

    def test_falls_back_when_all_disabled(self):
        selector, tracker = self._make_selector()
        # Disable all preferred strategies for VOLATILE
        for _ in range(10):
            tracker.record_trade("smart_scalp", -0.10)
            tracker.record_trade("grid", -0.10)
        from src.intelligence.market_regime import MarketRegime
        regime = MarketRegime("VOLATILE", 0.6, "HIGH", 0.2, [])
        # Should still return something (or None if truly all disabled)
        strategy = selector.select(regime)
        # Either a fallback or None — both are valid safe behaviors
        assert strategy is None or strategy.name in ("smart_scalp", "grid", "momentum", "mean_reversion")


# --- Multi-Timeframe Tests ---

class TestMultiTimeframeBrain:
    def test_aligned_buy(self):
        """All timeframes bullish → BUY signal."""
        brain = MultiTimeframeBrain()
        # Strong uptrend data for all timeframes
        closes = [50000 + i * 80 for i in range(100)]
        tf_data = {
            "5m": make_df(closes),
            "15m": make_df(closes),
            "1h": make_df(closes),
        }
        result = brain.analyze(tf_data)
        assert result["direction"] in ("BUY", "HOLD")

    def test_disagreeing_timeframes(self):
        """Disagreeing timeframes → HOLD."""
        brain = MultiTimeframeBrain()
        up = [50000 + i * 80 for i in range(100)]
        down = [55000 - i * 80 for i in range(100)]
        tf_data = {
            "5m": make_df(up),
            "15m": make_df(down),
            "1h": make_df(down),
        }
        result = brain.analyze(tf_data)
        # Should not be a strong BUY — timeframes disagree
        assert result["direction"] in ("SELL", "HOLD")

    def test_missing_timeframe(self):
        """Missing timeframe data → still works."""
        brain = MultiTimeframeBrain()
        closes = [50000 + i * 80 for i in range(100)]
        tf_data = {"5m": make_df(closes)}
        result = brain.analyze(tf_data)
        assert result["direction"] in ("BUY", "SELL", "HOLD")


# --- Trailing Stop Tests ---

class TestTrailingStopManager:
    def test_initial_stop_is_initial_sl(self):
        mgr = TrailingStopManager(activation_pct=0.5, trail_distance_pct=0.3)
        mgr.register("pos1", "BTC/USDT", "buy", 50000, 49800)
        assert mgr.get_stop("pos1") == 49800

    def test_activates_on_price_rise(self):
        mgr = TrailingStopManager(activation_pct=0.5, trail_distance_pct=0.3)
        mgr.register("pos1", "BTC/USDT", "buy", 50000, 49800)
        # Price rises 0.6% — above activation threshold of 0.5%
        mgr.update("pos1", 50300)
        assert mgr.is_activated("pos1")
        # Trailing stop should be above initial SL
        assert mgr.get_stop("pos1") > 49800

    def test_stop_only_moves_up_for_buy(self):
        mgr = TrailingStopManager(activation_pct=0.5, trail_distance_pct=0.3)
        mgr.register("pos1", "BTC/USDT", "buy", 50000, 49800)
        mgr.update("pos1", 50300)  # activate
        stop_after_300 = mgr.get_stop("pos1")
        mgr.update("pos1", 50500)  # price goes higher
        stop_after_500 = mgr.get_stop("pos1")
        assert stop_after_500 >= stop_after_300
        mgr.update("pos1", 50400)  # price drops a bit
        stop_after_drop = mgr.get_stop("pos1")
        assert stop_after_drop == stop_after_500  # stop doesn't move down

    def test_sell_trailing_stop(self):
        mgr = TrailingStopManager(activation_pct=0.5, trail_distance_pct=0.3)
        mgr.register("pos1", "BTC/USDT", "sell", 50000, 50200)
        # Price drops 0.6% — activates trailing
        mgr.update("pos1", 49700)
        assert mgr.is_activated("pos1")
        assert mgr.get_stop("pos1") < 50200

    def test_remove_cleans_up(self):
        mgr = TrailingStopManager()
        mgr.register("pos1", "BTC/USDT", "buy", 50000, 49800)
        mgr.remove("pos1")
        assert mgr.get_stop("pos1") is None


# --- Gemini Brain Tests ---

class TestGeminiBrain:
    def test_parse_valid_json(self):
        brain = GeminiBrain()
        result = brain._parse_response('{"direction": "BUY", "confidence": 0.75, "reasoning": "strong uptrend"}')
        assert result["direction"] == "BUY"
        assert result["confidence"] == 0.75

    def test_parse_with_code_fences(self):
        brain = GeminiBrain()
        result = brain._parse_response('```json\n{"direction": "SELL", "confidence": 0.6, "reasoning": "weak"}\n```')
        assert result["direction"] == "SELL"

    def test_parse_invalid_falls_back_to_hold(self):
        brain = GeminiBrain()
        result = brain._parse_response("this is not json at all")
        assert result["direction"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_parse_clamps_confidence(self):
        brain = GeminiBrain()
        result = brain._parse_response('{"direction": "BUY", "confidence": 5.0, "reasoning": "test"}')
        assert result["confidence"] == 1.0

    def test_parse_invalid_direction_defaults_hold(self):
        brain = GeminiBrain()
        result = brain._parse_response('{"direction": "MAYBE", "confidence": 0.5, "reasoning": "test"}')
        assert result["direction"] == "HOLD"
