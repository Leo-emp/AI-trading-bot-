# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.data.indicators import IndicatorEngine, TechnicalSignal


def make_ohlcv(n: int = 100, base_price: float = 50000.0,
               volatility: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Creates a random walk with specified volatility.
    Returns DataFrame matching Binance OHLCV format.
    """
    np.random.seed(42)  # reproducible
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    closes = [base_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + np.random.randn() * volatility)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [c - np.random.rand() * 50 for c in closes],
        "high": [c + abs(np.random.randn()) * 80 for c in closes],
        "low": [c - abs(np.random.randn()) * 80 for c in closes],
        "close": closes,
        "volume": [np.random.rand() * 1000 + 100 for _ in range(n)],
    })
    return df


class TestIndicatorEngine:

    def test_compute_all_adds_expected_columns(self):
        """compute_all should add all indicator columns to the DataFrame."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        result = engine.compute_all(df)

        expected_cols = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr", "ema_fast", "ema_slow", "volume_sma", "volume_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_bounded_0_100(self):
        """RSI should always be between 0 and 100."""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        rsi_valid = result["rsi"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_bollinger_bands_order(self):
        """Upper band > middle > lower, always."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        result = engine.compute_all(df)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_volume_ratio_flags_spikes(self):
        """Volume 2x the SMA should give volume_ratio >= 2.0."""
        engine = IndicatorEngine()
        df = make_ohlcv(50)
        # Inject a volume spike at the last row
        df.loc[df.index[-1], "volume"] = 999999
        result = engine.compute_all(df)
        assert result.iloc[-1]["volume_ratio"] > 2.0

    def test_get_signal_returns_valid_direction(self):
        """Signal direction should be BUY, SELL, or HOLD."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
        }
        signal = engine.get_signal(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.reasons, list)
