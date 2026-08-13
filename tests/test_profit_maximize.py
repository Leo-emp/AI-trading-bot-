# tests/test_profit_maximize.py
# Tests for partial exits and funding rate signal.

import pytest
from src.execution.partial_exit import PartialExitManager
from src.intelligence.funding_rate import FundingRateSignal


class TestPartialExit:
    """Test partial profit taking."""

    def test_no_exit_before_target(self):
        """Should not trigger before profit target."""
        mgr = PartialExitManager()
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=25.0)
        result = mgr.check("pos1", 100.30)  # +0.3% < 0.5%
        assert result is None

    def test_partial_exit_at_target(self):
        """Should take 50% off at +0.5% profit."""
        mgr = PartialExitManager()
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=25.0)
        result = mgr.check("pos1", 100.55)  # +0.55% > 0.5%
        assert result is not None
        assert result["action"] == "partial_exit"
        # Should close 50% of $25 = $12.50
        assert abs(result["size"] - 12.50) < 0.01
        assert result["pnl"] > 0  # should be profitable

    def test_remaining_size_after_partial(self):
        """Remaining size should be reduced after partial exit."""
        mgr = PartialExitManager()
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=25.0)
        mgr.check("pos1", 100.55)  # trigger partial
        remaining = mgr.get_remaining_size("pos1")
        assert abs(remaining - 12.50) < 0.01

    def test_partial_only_triggers_once(self):
        """Same level should not trigger twice."""
        mgr = PartialExitManager()
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=25.0)
        result1 = mgr.check("pos1", 100.55)  # first trigger
        assert result1 is not None
        result2 = mgr.check("pos1", 100.60)  # same level again
        assert result2 is None  # already triggered

    def test_sell_side_partial(self):
        """Partial exit should work for short positions too."""
        mgr = PartialExitManager()
        mgr.register("pos1", "ETHUSDT", "sell", entry_price=100.0, position_size=25.0)
        result = mgr.check("pos1", 99.40)  # price dropped 0.6% = profit for short
        assert result is not None
        assert result["pnl"] > 0

    def test_custom_levels(self):
        """Should support custom exit levels."""
        mgr = PartialExitManager(levels=[
            {"profit_pct": 0.3, "exit_pct": 0.25},  # take 25% at +0.3%
            {"profit_pct": 0.8, "exit_pct": 0.50},  # take 50% of remaining at +0.8%
        ])
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=20.0)

        # First level: 25% of $20 = $5
        result1 = mgr.check("pos1", 100.35)
        assert result1 is not None
        assert abs(result1["size"] - 5.0) < 0.01
        assert mgr.get_remaining_size("pos1") == pytest.approx(15.0, abs=0.01)

        # Second level: 50% of remaining $15 = $7.50
        result2 = mgr.check("pos1", 100.85)
        assert result2 is not None
        assert abs(result2["size"] - 7.50) < 0.01
        assert mgr.get_remaining_size("pos1") == pytest.approx(7.50, abs=0.01)

    def test_total_partial_pnl_accumulates(self):
        """Total partial P&L should accumulate across levels."""
        mgr = PartialExitManager(levels=[
            {"profit_pct": 0.3, "exit_pct": 0.50},
            {"profit_pct": 0.8, "exit_pct": 0.50},
        ])
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=20.0)
        mgr.check("pos1", 100.35)  # trigger level 1
        mgr.check("pos1", 100.85)  # trigger level 2
        total = mgr.get_total_partial_pnl("pos1")
        assert total > 0

    def test_remove_cleans_up(self):
        """Remove should clean up position data."""
        mgr = PartialExitManager()
        mgr.register("pos1", "BTCUSDT", "buy", entry_price=100.0, position_size=25.0)
        mgr.remove("pos1")
        assert mgr.get_remaining_size("pos1") == 0.0


class TestFundingRate:
    """Test funding rate signal analysis."""

    def test_extreme_positive_is_bearish(self):
        """High positive funding → overleveraged long → sell signal."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", 0.08)  # 0.08% = very high
        assert result["direction"] == "SELL"
        assert result["confidence"] > 0.5

    def test_extreme_negative_is_bullish(self):
        """High negative funding → overleveraged short → buy signal."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", -0.07)
        assert result["direction"] == "BUY"
        assert result["confidence"] > 0.5

    def test_neutral_funding_is_hold(self):
        """Normal funding → no signal."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", 0.01)  # normal
        assert result["direction"] == "HOLD"

    def test_moderate_positive_is_bearish(self):
        """Moderately high funding → weak sell signal."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", 0.04)
        assert result["direction"] == "SELL"
        assert result["confidence"] < 0.6  # not very confident

    def test_moderate_negative_is_bullish(self):
        """Moderately negative funding → weak buy signal."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", -0.04)
        assert result["direction"] == "BUY"

    def test_no_data_returns_hold(self):
        """No funding data → hold."""
        signal = FundingRateSignal()
        result = signal.analyze("UNKNOWN/USDT")
        assert result["direction"] == "HOLD"

    def test_confidence_capped(self):
        """Confidence should never exceed 0.85."""
        signal = FundingRateSignal()
        result = signal.analyze("BTC/USDT", 0.50)  # absurdly high
        assert result["confidence"] <= 0.85
