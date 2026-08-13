# tests/test_realtime_feed.py
# Tests for the real-time WebSocket feed.

import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from src.data.realtime_feed import RealtimeFeed


class TestRealtimeFeed:
    """Test RealtimeFeed data storage and helpers."""

    def test_get_price_returns_none_when_no_data(self):
        """Should return None for pairs with no data yet."""
        feed = RealtimeFeed(MagicMock())
        assert feed.get_price("BTC/USDT") is None

    def test_get_price_returns_latest(self):
        """Should return the latest stored price."""
        feed = RealtimeFeed(MagicMock())
        feed._prices["BTC/USDT"] = 67500.0
        assert feed.get_price("BTC/USDT") == 67500.0

    def test_get_spread_pct_default(self):
        """Should return 999 (block trade) when no spread data."""
        feed = RealtimeFeed(MagicMock())
        assert feed.get_spread_pct("BTC/USDT") == 999.0

    def test_get_spread_pct_with_data(self):
        """Should return stored spread percentage."""
        feed = RealtimeFeed(MagicMock())
        feed._spreads["BTC/USDT"] = 0.03
        assert feed.get_spread_pct("BTC/USDT") == 0.03

    def test_is_stale_when_no_data(self):
        """Should be stale when no data received yet."""
        feed = RealtimeFeed(MagicMock())
        assert feed.is_stale("BTC/USDT") is True

    def test_is_stale_after_recent_update(self):
        """Should not be stale right after an update."""
        feed = RealtimeFeed(MagicMock())
        feed._last_update["BTC/USDT"] = time.time()
        assert feed.is_stale("BTC/USDT", max_age=10.0) is False

    def test_is_stale_after_old_update(self):
        """Should be stale when data is too old."""
        feed = RealtimeFeed(MagicMock())
        feed._last_update["BTC/USDT"] = time.time() - 60
        assert feed.is_stale("BTC/USDT", max_age=10.0) is True

    def test_prices_returns_copy(self):
        """prices property should return a copy, not the internal dict."""
        feed = RealtimeFeed(MagicMock())
        feed._prices["BTC/USDT"] = 67500.0
        prices = feed.prices
        prices["BTC/USDT"] = 0  # modify the copy
        assert feed._prices["BTC/USDT"] == 67500.0  # original unchanged

    def test_get_best_bid_ask_defaults(self):
        """Should return 0,0 when no data."""
        feed = RealtimeFeed(MagicMock())
        bid, ask = feed.get_best_bid_ask("BTC/USDT")
        assert bid == 0.0
        assert ask == 0.0

    def test_get_best_bid_ask_with_data(self):
        """Should return stored bid/ask."""
        feed = RealtimeFeed(MagicMock())
        feed._best_bids["BTC/USDT"] = 67490.0
        feed._best_asks["BTC/USDT"] = 67510.0
        bid, ask = feed.get_best_bid_ask("BTC/USDT")
        assert bid == 67490.0
        assert ask == 67510.0

    def test_seconds_since_update_no_data(self):
        """Should return large number when no data."""
        feed = RealtimeFeed(MagicMock())
        assert feed.seconds_since_update("BTC/USDT") > 900

    def test_callback_registration(self):
        """Should store callbacks without error."""
        feed = RealtimeFeed(MagicMock())
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        feed.on_price_update(cb1)
        feed.on_candle_close(cb2)
        assert feed._on_price_update is cb1
        assert feed._on_candle_close is cb2


class TestFastEngineInit:
    """Test FastEngine initialization (doesn't require exchange connection)."""

    def test_fast_engine_creates(self):
        """FastEngine should instantiate without errors."""
        from src.core.fast_engine import FastEngine
        engine = FastEngine(mode="paper")
        assert engine._mode == "paper"
        assert engine._running is False

    def test_fast_engine_has_all_components(self):
        """FastEngine should have all trading components."""
        from src.core.fast_engine import FastEngine
        engine = FastEngine()
        assert engine._trade_gate is not None
        assert engine._smart_exit is not None
        assert engine._adaptive_sizer is not None
        assert engine._trade_filter is not None
        assert engine._macro_calendar is not None
        assert len(engine._strategies) == 4
