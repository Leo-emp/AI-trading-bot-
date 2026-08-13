# src/data/realtime_feed.py
# Real-Time Price Feed — WebSocket streaming from Binance.
#
# Problem with REST polling (old approach):
# - Check price every 30 seconds
# - Good trade appears at second 1, we see it at second 30
# - By then price moved, opportunity gone
#
# WebSocket streaming (this approach):
# - Binance pushes price updates every 100ms
# - We see every tick in real-time
# - React within 1-2 seconds of signal forming
#
# Architecture:
# - RealtimeFeed runs in background, receives all price updates
# - Maintains latest prices, candle buffers, and order book snapshots
# - Engine subscribes to events instead of polling
# - Signal evaluations trigger immediately when conditions align
#
# Uses ccxt's watch_* methods which wrap Binance WebSocket streams.

import asyncio
import logging
import time
from typing import Optional, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class RealtimeFeed:
    """Real-time price feed using Binance WebSocket streams.

    Streams:
    1. Ticker stream — real-time price + spread (updates every 100ms)
    2. Kline stream — candle updates as they form (not just on close)
    3. Order book stream — live bid/ask depth

    The engine registers callbacks that fire immediately when new data arrives.
    No more sleeping 30 seconds between checks.
    """

    def __init__(self, exchange):
        # ccxt exchange instance (from BinanceClient._exchange)
        self._exchange = exchange
        self._running = False

        # Latest data stores — updated by WebSocket, read by engine
        self._prices: dict[str, float] = {}
        self._spreads: dict[str, float] = {}  # spread as percentage
        self._best_bids: dict[str, float] = {}
        self._best_asks: dict[str, float] = {}
        self._last_update: dict[str, float] = {}

        # Callbacks — engine registers these to react to events
        self._on_price_update: Optional[Callable] = None
        self._on_candle_close: Optional[Callable] = None

        # Track candle state to detect closes
        self._candle_timestamps: dict[str, int] = {}

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    @property
    def prices(self) -> dict[str, float]:
        """Current prices for all tracked pairs."""
        return self._prices.copy()

    def get_price(self, pair: str) -> Optional[float]:
        """Get latest price for a pair."""
        return self._prices.get(pair)

    def get_spread_pct(self, pair: str) -> float:
        """Get current spread percentage for a pair."""
        return self._spreads.get(pair, 999.0)

    def get_best_bid_ask(self, pair: str) -> tuple[float, float]:
        """Get best bid and ask for a pair."""
        return self._best_bids.get(pair, 0.0), self._best_asks.get(pair, 0.0)

    def on_price_update(self, callback: Callable):
        """Register callback for price updates: callback(pair, price, spread_pct)"""
        self._on_price_update = callback

    def on_candle_close(self, callback: Callable):
        """Register callback for candle closes: callback(pair, timeframe, candle_data)"""
        self._on_candle_close = callback

    def seconds_since_update(self, pair: str) -> float:
        """How many seconds since last update for this pair."""
        last = self._last_update.get(pair, 0)
        if last == 0:
            return 999.0
        return time.time() - last

    def is_stale(self, pair: str, max_age: float = 10.0) -> bool:
        """True if data for this pair is older than max_age seconds."""
        return self.seconds_since_update(pair) > max_age

    async def start(self, pairs: list[str], timeframes: list[str]):
        """Start all WebSocket streams in background tasks."""
        self._running = True
        logger.info("Starting real-time feed for %s on %s", pairs, timeframes)

        # Start ticker stream for each pair (real-time prices + spread)
        for pair in pairs:
            task = asyncio.create_task(
                self._stream_ticker(pair),
                name=f"ticker:{pair}",
            )
            self._tasks.append(task)

        # Start kline streams for each pair + timeframe combo
        for pair in pairs:
            for tf in timeframes:
                task = asyncio.create_task(
                    self._stream_klines(pair, tf),
                    name=f"kline:{pair}:{tf}",
                )
                self._tasks.append(task)

        logger.info("Started %d WebSocket streams", len(self._tasks))

    async def stop(self):
        """Stop all streams gracefully."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Real-time feed stopped")

    async def _stream_ticker(self, pair: str):
        """Stream real-time ticker data (price + bid/ask spread).

        This is the fastest update — fires every ~100ms on active pairs.
        Gives us real-time spread monitoring to avoid bad fills.
        """
        retry_delay = 1
        max_retry = 30

        while self._running:
            try:
                ticker = await self._exchange.watch_ticker(pair)
                if ticker:
                    price = ticker.get("last", 0.0)
                    bid = ticker.get("bid", 0.0)
                    ask = ticker.get("ask", 0.0)

                    if price > 0:
                        self._prices[pair] = price
                        self._last_update[pair] = time.time()

                    if bid > 0 and ask > 0:
                        self._best_bids[pair] = bid
                        self._best_asks[pair] = ask
                        mid = (bid + ask) / 2
                        self._spreads[pair] = ((ask - bid) / mid) * 100

                    # Notify engine of price update
                    if self._on_price_update and price > 0:
                        try:
                            await self._on_price_update(
                                pair, price, self._spreads.get(pair, 0)
                            )
                        except Exception as e:
                            logger.debug("Price update callback error: %s", e)

                retry_delay = 1  # reset backoff on success

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning("Ticker stream %s error: %s. Retry in %ds",
                                  pair, e, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry)

    async def _stream_klines(self, pair: str, timeframe: str):
        """Stream kline (candlestick) data.

        Fires on every kline update (not just candle close).
        We detect candle close by comparing timestamps.
        Candle close = new signal evaluation opportunity.
        """
        retry_delay = 1
        max_retry = 30
        key = f"{pair}:{timeframe}"

        while self._running:
            try:
                candles = await self._exchange.watch_ohlcv(pair, timeframe)
                if candles:
                    latest = candles[-1]
                    ts = latest[0]  # candle open timestamp

                    # Detect candle close: new timestamp means previous candle closed
                    prev_ts = self._candle_timestamps.get(key)
                    self._candle_timestamps[key] = ts

                    if prev_ts is not None and ts != prev_ts:
                        # Previous candle just closed — this is when signals form
                        if self._on_candle_close:
                            candle_data = {
                                "timestamp": ts,
                                "open": latest[1],
                                "high": latest[2],
                                "low": latest[3],
                                "close": latest[4],
                                "volume": latest[5],
                            }
                            try:
                                await self._on_candle_close(pair, timeframe, candle_data)
                            except Exception as e:
                                logger.debug("Candle close callback error: %s", e)

                retry_delay = 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning("Kline stream %s:%s error: %s. Retry in %ds",
                                  pair, timeframe, e, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry)
