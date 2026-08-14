# test_connection.py
# Quick test: connect to Binance, fetch BTC price, check account.
# Run: python test_connection.py

import asyncio
import os
from dotenv import load_dotenv
import ccxt.async_support as ccxt

load_dotenv()


async def test():
    api_key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("ERROR: API keys not set in .env")
        return

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "sandbox": False,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    try:
        # Test 1: Fetch BTC price
        ticker = await exchange.fetch_ticker("BTC/USDT")
        print(f"  BTC/USDT price: ${ticker['last']:,.2f}")
        print(f"  24h volume: ${ticker['quoteVolume']:,.0f}")
        print(f"  CONNECTION OK")

        # Test 2: Fetch account balance
        balance = await exchange.fetch_balance()
        usdt = balance.get("USDT", {}).get("free", 0)
        print(f"\n  USDT balance: ${usdt:.2f}")

        # Test 3: Fetch 5 candles to confirm data feed works
        candles = await exchange.fetch_ohlcv("BTC/USDT", "5m", limit=5)
        print(f"  Last 5 candles fetched: {len(candles)} OK")

        print(f"\n  ALL TESTS PASSED — ready for paper trading")

    except ccxt.AuthenticationError as e:
        print(f"  AUTH ERROR: {e}")
        print("  Check your API key and secret in .env")
    except ccxt.ExchangeNotAvailable as e:
        print(f"  EXCHANGE ERROR: {e}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    print("\n  Testing Binance connection...\n")
    asyncio.run(test())
