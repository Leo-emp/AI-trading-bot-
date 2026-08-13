# tests/test_strategies_all.py
# Tests for all 4 strategies: grid, momentum, mean_reversion, smart_scalp.

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.strategies.grid import GridStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.smart_scalp import SmartScalpStrategy
from src.data.indicators import IndicatorEngine


def make_df_with_indicators(closes: list[float], volumes: list[float] = None) -> pd.DataFrame:
    """Build a DataFrame with OHLCV + indicators from a close price series."""
    n = len(closes)
    if volumes is None:
        volumes = [500.0] * n

    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
        "open": [c - 10 for c in closes],
        "high": [c + 30 for c in closes],
        "low": [c - 30 for c in closes],
        "close": closes,
        "volume": volumes,
    })
    engine = IndicatorEngine()
    return engine.compute_all(df)


def make_ranging_data(n=100, center=50000, amplitude=200):
    """Price oscillating around a center — ideal for grid/mean_reversion."""
    closes = [center + amplitude * np.sin(i * 0.3) for i in range(n)]
    return closes


def make_trending_up_data(n=100, start=50000, step=50):
    """Steadily rising prices — ideal for momentum."""
    return [start + step * i + np.random.randn() * 10 for i in range(n)]


def make_trending_down_data(n=100, start=55000, step=50):
    """Steadily falling prices."""
    return [start - step * i + np.random.randn() * 10 for i in range(n)]


class TestGridStrategy:
    def test_returns_strategy_signal(self):
        df = make_df_with_indicators(make_ranging_data())
        strategy = GridStrategy()
        config = {"levels": 5, "spacing_atr_multiplier": 0.5,
                  "min_spacing_pct": 0.3, "max_spacing_pct": 2.0}
        signal = strategy.evaluate(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "grid"

    def test_hold_when_insufficient_data(self):
        # 20 candles — enough for indicators but not for strategy (needs 30)
        df = make_df_with_indicators([50000] * 20)
        strategy = GridStrategy()
        signal = strategy.evaluate(df, {})
        assert signal.direction == "HOLD"

    def test_name_property(self):
        assert GridStrategy().name == "grid"


class TestMomentumStrategy:
    def test_returns_strategy_signal(self):
        df = make_df_with_indicators(make_trending_up_data())
        strategy = MomentumStrategy()
        config = {"ema_fast": 9, "ema_slow": 21, "rsi_min": 30,
                  "rsi_max": 70, "trailing_stop_pct": 1.5}
        signal = strategy.evaluate(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "momentum"

    def test_hold_when_insufficient_data(self):
        df = make_df_with_indicators([50000] * 20)
        strategy = MomentumStrategy()
        signal = strategy.evaluate(df, {})
        assert signal.direction == "HOLD"

    def test_name_property(self):
        assert MomentumStrategy().name == "momentum"

    def test_sets_sl_tp_on_signal(self):
        # Strong uptrend should trigger a signal with SL/TP
        df = make_df_with_indicators(make_trending_up_data(n=100, step=80))
        strategy = MomentumStrategy()
        config = {"rsi_min": 30, "rsi_max": 70, "trailing_stop_pct": 1.5}
        signal = strategy.evaluate(df, config)
        if signal.direction != "HOLD":
            assert signal.stop_loss > 0
            assert signal.take_profit > 0


class TestMeanReversionStrategy:
    def test_returns_strategy_signal(self):
        df = make_df_with_indicators(make_ranging_data())
        strategy = MeanReversionStrategy()
        config = {"rsi_extreme_low": 25, "rsi_extreme_high": 75,
                  "bollinger_period": 20, "bollinger_std": 2.0}
        signal = strategy.evaluate(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "mean_reversion"

    def test_hold_when_insufficient_data(self):
        df = make_df_with_indicators([50000] * 20)
        strategy = MeanReversionStrategy()
        signal = strategy.evaluate(df, {})
        assert signal.direction == "HOLD"

    def test_name_property(self):
        assert MeanReversionStrategy().name == "mean_reversion"


class TestSmartScalpStrategy:
    def test_still_works(self):
        """Verify existing smart_scalp strategy still functions."""
        df = make_df_with_indicators(make_ranging_data())
        strategy = SmartScalpStrategy()
        config = {"rsi_oversold": 30, "rsi_overbought": 70,
                  "volume_spike_multiplier": 1.5,
                  "take_profit_pct": 0.8, "stop_loss_pct": 0.4}
        signal = strategy.evaluate(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "smart_scalp"
