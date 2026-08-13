# main.py
# Entry point for the AI Trading Bot.
#
# Usage:
#   python main.py              → Paper trading (default, SAFE)
#   python main.py --live       → Live trading (REAL MONEY — be careful!)
#   python main.py --backtest   → Backtest strategies on historical data
#
# IMPORTANT: Always start with paper trading.
# The bot must pass the backtesting gate AND 14+ days of paper trading
# before live mode should even be considered.
#
# Environment variables needed:
#   BINANCE_API_KEY    — from binance.com (NEVER enable withdrawal!)
#   BINANCE_SECRET     — from binance.com
#   TELEGRAM_BOT_TOKEN — optional, for notifications
#   TELEGRAM_CHAT_ID   — optional, for notifications

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env file FIRST, before any other imports that might use env vars
load_dotenv()


def setup_logging(level: str = "INFO"):
    """Configure logging for the trading bot."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trading.log", encoding="utf-8"),
        ],
    )


async def run_paper():
    """Start paper trading mode."""
    from src.core.engine import TradingEngine

    engine = TradingEngine(mode="paper")
    try:
        await engine.start()
        await engine.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    finally:
        await engine.stop()


async def run_live():
    """Start live trading mode. REAL MONEY."""
    from src.core.engine import TradingEngine

    # Safety check — require explicit confirmation
    if os.getenv("TRADING_MODE") != "live":
        print("ERROR: Set TRADING_MODE=live in .env to enable live trading.")
        print("This is a safety check to prevent accidental real-money trading.")
        sys.exit(1)

    if not os.getenv("BINANCE_API_KEY"):
        print("ERROR: BINANCE_API_KEY not set in .env")
        sys.exit(1)

    engine = TradingEngine(mode="live")
    try:
        await engine.start()
        await engine.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    finally:
        await engine.stop()


async def run_backtest():
    """Run backtesting on historical data."""
    from backtest.engine import BacktestEngine
    from backtest.data_loader import DataLoader
    from backtest.report import BacktestReport
    from src.strategies.smart_scalp import SmartScalpStrategy
    from src.core.config import load_strategies

    print("Starting backtest...")

    # Load strategy config
    strategies_config = load_strategies()
    scalp_config = strategies_config.get("smart_scalp", {})

    # Download historical data
    loader = DataLoader()
    print("Downloading historical data (this may take a minute)...")
    df = await loader.download("BTC/USDT", timeframe="5m", days=180)

    if df.empty:
        print("ERROR: No historical data available. Check your Binance connection.")
        sys.exit(1)

    print(f"Loaded {len(df)} candles. Running backtest...")

    # Run backtest
    engine = BacktestEngine(
        initial_balance=50.0,
        maker_fee_rate=0.00075,
    )

    strategy = SmartScalpStrategy()
    result = engine.run(strategy, df, scalp_config, pair="BTC/USDT")

    # Print results
    BacktestReport.print_summary(result, strategy_name="Smart Scalp")

    # Save report
    os.makedirs("backtest/reports", exist_ok=True)
    BacktestReport.save_report(result, "backtest/reports/latest.txt")
    print("Report saved to backtest/reports/latest.txt")


def main():
    """Parse CLI arguments and launch the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="AI Trading Bot — Autonomous crypto trading with multi-brain consensus"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (REAL MONEY — requires TRADING_MODE=live in .env)"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtesting on historical data"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.backtest:
        asyncio.run(run_backtest())
    elif args.live:
        asyncio.run(run_live())
    else:
        asyncio.run(run_paper())


if __name__ == "__main__":
    main()
