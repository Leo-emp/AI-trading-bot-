# tests/test_strategies.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.strategies.smart_scalp import SmartScalpStrategy
from src.strategies.base import StrategySignal
from src.data.indicators import IndicatorEngine


def make_bullish_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate data with a clear uptrend for testing BUY signals."""
    np.random.seed(42)
    base = 50000
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    # Steady uptrend with noise
    closes = [base + i * 20 + np.random.randn() * 30 for i in range(n)]
    # Make last few candles show oversold RSI by adding a dip then recovery
    for i in range(n - 10, n - 3):
        closes[i] = closes[i] - 500  # dip
    for i in range(n - 3, n):
        closes[i] = closes[i - 1] + 100  # recovery
    volumes = [500 + np.random.rand() * 200 for _ in range(n)]
    # Volume spike on last candle
    volumes[-1] = 3000
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [c - 10 for c in closes],
        "high": [c + 50 for c in closes],
        "low": [c - 60 for c in closes],
        "close": closes,
        "volume": volumes,
    })


class TestSmartScalp:

    def test_returns_strategy_signal(self):
        """evaluate() should return a StrategySignal."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        df = make_bullish_ohlcv()
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        assert isinstance(signal, StrategySignal)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "smart_scalp"

    def test_stop_loss_and_take_profit_are_set(self):
        """When signal is BUY/SELL, SL and TP must be set."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        df = make_bullish_ohlcv()
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        if signal.direction != "HOLD":
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            if signal.direction == "BUY":
                assert signal.stop_loss < signal.entry_price
                assert signal.take_profit > signal.entry_price

    def test_hold_when_no_confirmation(self):
        """With random flat data, should mostly HOLD (no multi-confirmation)."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        np.random.seed(99)
        # Flat random walk — no clear signal
        n = 100
        closes = [50000 + np.random.randn() * 10 for _ in range(n)]
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
            "open": closes, "high": [c+5 for c in closes],
            "low": [c-5 for c in closes], "close": closes,
            "volume": [500] * n,
        })
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        assert signal.direction == "HOLD"
