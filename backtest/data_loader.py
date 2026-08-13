# backtest/data_loader.py
# Downloads and caches historical OHLCV data from Binance.
# Saves to CSV for reuse (avoids re-downloading on every backtest run).

import asyncio
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from src.data.feed import BinanceClient

logger = logging.getLogger(__name__)


class DataLoader:
    """Downloads historical price data for backtesting.

    Fetches from Binance REST API and caches to CSV files.
    Handles pagination (Binance returns max 1000 candles per request).
    """

    def __init__(self, cache_dir: str = "backtest/data"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    async def download(self, pair: str, timeframe: str = "5m",
                       days: int = 180) -> pd.DataFrame:
        """Download historical OHLCV data from Binance.

        Fetches `days` worth of data, paginating through the API.
        Caches to CSV for future use.
        """
        cache_file = self._cache_path(pair, timeframe, days)
        if os.path.exists(cache_file):
            logger.info("Loading cached data from %s", cache_file)
            return self.load_csv(cache_file)

        client = BinanceClient(paper_mode=True)
        await client.connect()

        all_data = []
        # Calculate starting timestamp
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        while True:
            df = await client.get_historical_ohlcv(pair, timeframe, since=since, limit=1000)
            if df.empty:
                break
            all_data.append(df)
            # Move `since` to after the last candle
            last_ts = int(df.iloc[-1]["timestamp"].timestamp() * 1000)
            since = last_ts + 1
            # Rate limiting
            await asyncio.sleep(0.5)
            # Stop if we've reached current time
            if len(df) < 1000:
                break

        await client.disconnect()

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        self.save_csv(result, cache_file)
        logger.info("Downloaded %d candles for %s %s (%d days)", len(result), pair, timeframe, days)
        return result

    def save_csv(self, df: pd.DataFrame, path: str):
        """Save DataFrame to CSV."""
        df.to_csv(path, index=False)

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _cache_path(self, pair: str, timeframe: str, days: int) -> str:
        safe_pair = pair.replace("/", "_")
        return os.path.join(self._cache_dir, f"{safe_pair}_{timeframe}_{days}d.csv")
