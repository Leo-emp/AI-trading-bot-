# tests/test_feed.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.feed import BinanceClient, PriceFeed
from src.data.order_book import OrderBookAnalyzer, OrderBookSignal


class TestBinanceClient:
    """Tests for the Binance API wrapper."""

    @pytest.mark.asyncio
    async def test_get_ohlcv_returns_dataframe(self):
        """get_ohlcv should return a pandas DataFrame with OHLCV columns."""
        client = BinanceClient(paper_mode=True)
        # In paper mode, uses ccxt's sandbox or returns mock data
        with patch.object(client, '_exchange') as mock_ex:
            mock_ex.fetch_ohlcv = AsyncMock(return_value=[
                [1700000000000, 50000, 50100, 49900, 50050, 100],
                [1700000060000, 50050, 50200, 50000, 50150, 120],
            ])
            df = await client.get_ohlcv("BTC/USDT", "5m", limit=2)
            assert len(df) == 2
            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
            assert df.iloc[0]["close"] == 50050

    @pytest.mark.asyncio
    async def test_place_limit_order_enforces_limit_type(self):
        """Should always place LIMIT orders, never market."""
        client = BinanceClient(paper_mode=True)
        with patch.object(client, '_exchange') as mock_ex:
            mock_ex.create_limit_buy_order = AsyncMock(return_value={"id": "123"})
            result = await client.place_limit_order("BTC/USDT", "buy", 50000, 0.0002)
            mock_ex.create_limit_buy_order.assert_called_once()
            assert result["id"] == "123"


class TestOrderBookAnalyzer:
    """Tests for order book depth analysis."""

    def test_detects_buy_side_imbalance(self):
        """When bids >> asks, imbalance should be positive (bullish)."""
        analyzer = OrderBookAnalyzer()
        # More volume on bid side → bullish imbalance
        book = {
            "bids": [[50000, 10], [49990, 8], [49980, 7]],  # total: 25
            "asks": [[50010, 3], [50020, 2], [50030, 2]],   # total: 7
        }
        signal = analyzer.analyze(book)
        assert signal.imbalance > 0.5  # strongly bullish
        assert signal.is_liquid  # has orders on both sides

    def test_detects_whale_order(self):
        """Orders > $50K on a single level should flag whale."""
        analyzer = OrderBookAnalyzer(whale_threshold_usd=50000)
        book = {
            # 2 BTC at $50000 = $100K → whale
            "bids": [[50000, 2.0], [49990, 0.1]],
            "asks": [[50010, 0.1], [50020, 0.1]],
        }
        signal = analyzer.analyze(book)
        assert signal.whale_detected is True
        assert signal.whale_side == "bid"

    def test_spread_calculation(self):
        """Spread should be (best_ask - best_bid) / mid_price * 100."""
        analyzer = OrderBookAnalyzer()
        book = {
            "bids": [[50000, 1]],
            "asks": [[50100, 1]],
        }
        signal = analyzer.analyze(book)
        # spread = (50100 - 50000) / 50050 * 100 = 0.1998%
        assert abs(signal.spread_pct - 0.1998) < 0.01

    def test_illiquid_when_spread_too_wide(self):
        """Spread > 0.15% should mark as illiquid."""
        analyzer = OrderBookAnalyzer(max_spread_pct=0.15)
        book = {
            "bids": [[50000, 1]],
            "asks": [[50200, 1]],  # 0.4% spread
        }
        signal = analyzer.analyze(book)
        assert signal.is_liquid is False
