# tests/test_trade_filter.py
# Tests for the Trade Quality Filter — the final gate before execution.

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.intelligence.trade_filter import TradeFilter


class TestTimeFilter:
    """Test dead hour filtering."""

    def test_blocks_during_dead_hours(self):
        """Should block trades during low-volume hours."""
        f = TradeFilter(dead_hours_utc=[0, 1])
        now = datetime(2026, 8, 13, 0, 30, tzinfo=timezone.utc)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is False
        assert result["filter"] == "time"

    def test_allows_during_active_hours(self):
        """Should allow trades during active hours."""
        f = TradeFilter(dead_hours_utc=[0, 1])
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is True


class TestSpreadFilter:
    """Test bid-ask spread filtering."""

    def test_blocks_wide_spread(self):
        """Should block when spread is too wide."""
        f = TradeFilter(max_spread_pct=0.05)
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.7, spread_pct=0.08,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is False
        assert result["filter"] == "spread"

    def test_allows_tight_spread(self):
        """Should allow when spread is acceptable."""
        f = TradeFilter(max_spread_pct=0.05)
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is True

    def test_spread_calculation(self):
        """Spread percentage should be calculated correctly."""
        f = TradeFilter()
        # BTC at $100,000 with $50 spread
        spread = f.get_spread_pct(99975.0, 100025.0)
        assert abs(spread - 0.05) < 0.001  # 0.05%


class TestCooldownFilter:
    """Test post-loss cooldown."""

    def test_blocks_immediately_after_loss(self):
        """Should block trades right after a loss."""
        f = TradeFilter(cooldown_minutes=5)
        loss_time = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        f._last_loss_time = loss_time
        # 2 minutes later
        now = loss_time + timedelta(minutes=2)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is False
        assert result["filter"] == "cooldown"

    def test_allows_after_cooldown_expires(self):
        """Should allow trades after cooldown period."""
        f = TradeFilter(cooldown_minutes=5)
        loss_time = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        f._last_loss_time = loss_time
        # 6 minutes later
        now = loss_time + timedelta(minutes=6)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is True

    def test_record_loss_sets_cooldown(self):
        """record_loss should start the cooldown timer."""
        f = TradeFilter(cooldown_minutes=5)
        f.record_loss()
        assert f._last_loss_time is not None


class TestConfidenceFilter:
    """Test minimum confidence threshold."""

    def test_blocks_weak_signals(self):
        """Should block when confidence is too low."""
        f = TradeFilter(min_confidence=0.55)
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.51, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is False
        assert result["filter"] == "confidence"

    def test_allows_strong_signals(self):
        """Should allow when confidence is high enough."""
        f = TradeFilter(min_confidence=0.55)
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.70, spread_pct=0.03,
                        open_positions=[], pair="BTC/USDT", now=now)
        assert result["pass"] is True


class TestCorrelationFilter:
    """Test correlated position limiting."""

    def test_blocks_too_many_correlated_positions(self):
        """Should block when too many correlated assets are open."""
        f = TradeFilter(max_correlated_positions=2)
        # Mock 2 existing positions in correlated assets
        pos1 = MagicMock()
        pos1.pair = "BTC/USDT"
        pos2 = MagicMock()
        pos2.pair = "ETH/USDT"
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        # Try to open a 3rd correlated position
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[pos1, pos2], pair="SOL/USDT", now=now)
        assert result["pass"] is False
        assert result["filter"] == "correlation"

    def test_allows_when_under_correlation_limit(self):
        """Should allow when under the correlated position limit."""
        f = TradeFilter(max_correlated_positions=2)
        pos1 = MagicMock()
        pos1.pair = "BTC/USDT"
        now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)
        result = f.check(confidence=0.7, spread_pct=0.03,
                        open_positions=[pos1], pair="ETH/USDT", now=now)
        assert result["pass"] is True
