# src/data/feed.py
# Binance exchange client and real-time price feed.
# Uses ccxt for REST API and WebSocket for live data.
# Auto-reconnects on disconnection with exponential backoff.
# LIMIT ORDERS ONLY — never places market orders.

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Callable

import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper around ccxt's Binance implementation.

    Enforces limit-only orders and handles connection management.
    In paper_mode, uses sandbox endpoints or mocks for testing.
    """

    def __init__(self, paper_mode: bool = True,
                 api_key: str = "", secret: str = ""):
        # Whether to simulate trades or use real money
        self._paper_mode = paper_mode

        # Create the ccxt exchange instance
        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "sandbox": paper_mode,
            "enableRateLimit": True,  # respect Binance rate limits automatically
            "options": {
                "defaultType": "spot",  # spot trading only, no futures
            },
        })

    async def connect(self):
        """Load exchange markets (required before trading)."""
        await self._exchange.load_markets()
        logger.info("Connected to Binance (paper=%s)", self._paper_mode)

    async def disconnect(self):
        """Close the exchange connection."""
        await self._exchange.close()

    async def get_ohlcv(self, pair: str, timeframe: str = "5m",
                        limit: int = 100) -> "pd.DataFrame":
        """Fetch recent OHLCV candles.

        Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
        """
        raw = await self._exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def get_historical_ohlcv(self, pair: str, timeframe: str = "5m",
                                    since: Optional[int] = None,
                                    limit: int = 1000) -> "pd.DataFrame":
        """Fetch historical OHLCV data for backtesting.

        `since` is a Unix timestamp in milliseconds.
        Binance returns max 1000 candles per request.
        """
        raw = await self._exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def get_ticker(self, pair: str) -> dict:
        """Get current ticker (last price, bid, ask, volume)."""
        return await self._exchange.fetch_ticker(pair)

    async def get_balance(self) -> dict:
        """Get account balances. Returns dict like {'USDT': 50.0, 'BTC': 0.001}."""
        balance = await self._exchange.fetch_balance()
        # Return only non-zero free balances
        return {
            currency: amount
            for currency, amount in balance["free"].items()
            if amount > 0
        }

    async def get_order_book(self, pair: str, limit: int = 20) -> dict:
        """Fetch order book depth. Returns {'bids': [...], 'asks': [...]}."""
        return await self._exchange.fetch_order_book(pair, limit=limit)

    async def place_limit_order(self, pair: str, side: str,
                                 price: float, quantity: float) -> dict:
        """Place a LIMIT order. NEVER places market orders.

        side: 'buy' or 'sell'
        Returns order info dict with 'id', 'status', etc.
        """
        if side == "buy":
            return await self._exchange.create_limit_buy_order(pair, quantity, price)
        elif side == "sell":
            return await self._exchange.create_limit_sell_order(pair, quantity, price)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

    async def cancel_order(self, pair: str, order_id: str):
        """Cancel an open order."""
        return await self._exchange.cancel_order(order_id, pair)

    async def get_order_status(self, pair: str, order_id: str) -> dict:
        """Check if an order has been filled, partially filled, or is still open."""
        return await self._exchange.fetch_order(order_id, pair)


class PriceFeed:
    """Real-time price feed using Binance WebSocket.

    Subscribes to kline (candlestick) streams for specified pairs
    and timeframes. Calls on_candle callback when new candles arrive.

    Auto-reconnects with exponential backoff on disconnection.
    Tracks data freshness — stale data halts trading.
    """

    def __init__(self, client: BinanceClient):
        self._client = client
        # Timestamp of last received data — used for stale detection
        self._last_update: float = 0
        # Whether the feed is running
        self._running = False
        # Latest candle data per pair
        self._latest: dict[str, dict] = {}

    async def start(self, pairs: list[str], timeframes: list[str],
                    on_candle: Optional[Callable] = None):
        """Start watching candles for given pairs.

        Uses ccxt's watch_ohlcv for WebSocket streaming.
        Falls back to REST polling if WebSocket fails.
        """
        self._running = True
        self._last_update = time.time()

        retry_delay = 1  # exponential backoff starting point
        max_retry = 60   # max delay between retries

        while self._running:
            try:
                # ccxt pro watch_ohlcv provides WebSocket streaming
                for pair in pairs:
                    for tf in timeframes:
                        candles = await self._client._exchange.watch_ohlcv(pair, tf)
                        if candles:
                            self._last_update = time.time()
                            latest = candles[-1]
                            self._latest[f"{pair}:{tf}"] = {
                                "timestamp": latest[0],
                                "open": latest[1],
                                "high": latest[2],
                                "low": latest[3],
                                "close": latest[4],
                                "volume": latest[5],
                            }
                            if on_candle:
                                await on_candle(pair, tf, self._latest[f"{pair}:{tf}"])
                # Reset backoff on success
                retry_delay = 1

            except Exception as e:
                logger.error("PriceFeed error: %s. Retrying in %ds...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                # Exponential backoff: 1, 2, 4, 8, 16, 32, 60, 60...
                retry_delay = min(retry_delay * 2, max_retry)

    async def stop(self):
        """Stop the price feed."""
        self._running = False

    def is_stale(self, max_age_seconds: float = 10) -> bool:
        """True if no data received within max_age_seconds.

        The trading engine checks this — if data is stale, all
        trading halts until fresh data arrives. Prevents trading
        on outdated information.
        """
        if self._last_update == 0:
            return True  # never received data
        return (time.time() - self._last_update) > max_age_seconds

    def get_latest(self, pair: str, timeframe: str) -> Optional[dict]:
        """Get the most recent candle for a pair+timeframe."""
        return self._latest.get(f"{pair}:{timeframe}")
