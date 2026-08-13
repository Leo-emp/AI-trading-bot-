# tests/test_backtest.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from backtest.engine import BacktestEngine, BacktestResult
from src.strategies.smart_scalp import SmartScalpStrategy


def make_trending_data(n: int = 500) -> pd.DataFrame:
    """Generate 500 candles with alternating trends for backtesting."""
    np.random.seed(42)
    base = 50000.0
    closes = [base]
    for i in range(1, n):
        # Alternate between up-trend and down-trend every 50 candles
        cycle = (i // 50) % 2
        drift = 10 if cycle == 0 else -10
        closes.append(closes[-1] + drift + np.random.randn() * 50)

    volumes = [500 + np.random.rand() * 500 for _ in range(n)]
    # Add some volume spikes
    for i in range(0, n, 20):
        volumes[i] = 3000

    return pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
        "open": [c - np.random.rand() * 30 for c in closes],
        "high": [c + abs(np.random.randn()) * 60 for c in closes],
        "low": [c - abs(np.random.randn()) * 60 for c in closes],
        "close": closes,
        "volume": volumes,
    })


class TestBacktestEngine:

    def test_returns_backtest_result(self):
        """Should return a BacktestResult with all metrics."""
        engine = BacktestEngine(initial_balance=50.0, maker_fee_rate=0.00075)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1.0
        assert result.total_fees >= 0

    def test_fees_are_always_subtracted(self):
        """Net P&L should always be less than gross P&L (fees exist)."""
        engine = BacktestEngine(initial_balance=100.0, maker_fee_rate=0.00075)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        if result.total_trades > 0:
            assert result.total_fees > 0

    def test_max_drawdown_is_calculated(self):
        """Max drawdown should be >= 0."""
        engine = BacktestEngine(initial_balance=50.0)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        assert result.max_drawdown_pct >= 0
