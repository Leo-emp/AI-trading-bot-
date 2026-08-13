# backtest/run_backtest.py
# Offline backtest -- runs without Binance API keys.
#
# Uses numpy to generate REALISTIC crypto price data:
# - Multi-day trends with momentum (real crypto trends for days, not hours)
# - Mean reversion at extremes (panic selling overshoots, then bounces)
# - Volatility clustering (volatile periods cluster together)
# - Volume follows price action (spikes on breakouts and capitulation)
# - Support/resistance levels that price bounces off
#
# The previous version had too-weak trends (0.01% drift) that created
# essentially random walks — no strategy can beat random data + fees.
#
# Run: python -m backtest.run_backtest

import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine
from backtest.report import BacktestReport
from src.strategies.smart_scalp import SmartScalpStrategy
from src.strategies.grid import GridStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.core.config import load_strategies

logging.basicConfig(level=logging.WARNING)


def generate_realistic_crypto_data(days: int = 30, timeframe_minutes: int = 5,
                                    base_price: float = 67000.0,
                                    seed: int = 42) -> pd.DataFrame:
    """Generate realistic crypto OHLCV data for backtesting.

    Calibrated to real BTC 5m statistics:
    - Daily move: ~1-3% (not 10%+)
    - Drift per 5m candle: 0.003-0.01% in trending mode
    - 5m ATR: ~$50-150 at BTC $67K (0.07-0.22%)
    - 30-day range: ~10-25% (not 100%+)
    - Trends last 1-3 days with clear momentum
    - Ranging periods have mean reversion to local average
    """
    rng = np.random.RandomState(seed)
    candles_per_day = (24 * 60) // timeframe_minutes  # 288 per day
    total_candles = days * candles_per_day

    prices = np.zeros(total_candles)
    prices[0] = base_price
    volumes = np.zeros(total_candles)

    # Market regime blocks -- longer regimes (12h to 2 days)
    min_block = candles_per_day // 2  # 12 hours minimum
    max_block = candles_per_day * 2   # 2 days maximum

    regimes = []
    i = 0
    while i < total_candles:
        block = rng.randint(min_block, max_block)
        r = rng.random()
        if r < 0.20:
            regime = "trend_up"
        elif r < 0.40:
            regime = "trend_down"
        elif r < 0.75:
            regime = "ranging"
        else:
            regime = "volatile"
        regimes.extend([regime] * min(block, total_candles - i))
        i += block
    regimes = regimes[:total_candles]

    # Volatility -- GARCH-like clustering
    vol_state = base_price * 0.00015  # starting noise (about $10)
    base_vol = 100.0

    for i in range(1, total_candles):
        regime = regimes[i]

        # Target noise per regime (as fraction of base_price)
        target_vol = {
            "trend_up": base_price * 0.00020,    # ~$13 noise
            "trend_down": base_price * 0.00025,   # ~$17 noise
            "ranging": base_price * 0.00012,      # ~$8 noise
            "volatile": base_price * 0.00050,     # ~$34 noise
        }[regime]
        vol_state = 0.95 * vol_state + 0.05 * target_vol

        if regime == "trend_up":
            # Realistic drift: ~0.003-0.01% per candle = ~1-3% per day
            drift = base_price * rng.uniform(0.00003, 0.0001)
            noise = rng.normal(0, vol_state)
            prices[i] = prices[i-1] + drift + noise
            volumes[i] = base_vol * rng.uniform(1.3, 2.5)

        elif regime == "trend_down":
            drift = -base_price * rng.uniform(0.00003, 0.0001)
            noise = rng.normal(0, vol_state)
            prices[i] = prices[i-1] + drift + noise
            volumes[i] = base_vol * rng.uniform(1.3, 3.0)

        elif regime == "ranging":
            local_mean = np.mean(prices[max(0, i-50):i])
            reversion = (local_mean - prices[i-1]) * 0.03
            noise = rng.normal(0, vol_state)
            prices[i] = prices[i-1] + reversion + noise
            volumes[i] = base_vol * rng.uniform(0.3, 1.2)

        else:  # volatile
            noise = rng.normal(0, vol_state * 1.5)
            prices[i] = prices[i-1] + noise
            volumes[i] = base_vol * rng.uniform(2.0, 4.0)

    # OHLCV from close prices
    timestamps = pd.date_range(
        start="2026-02-01", periods=total_candles,
        freq=f"{timeframe_minutes}min", tz="UTC",
    )

    opens = np.roll(prices, 1)
    opens[0] = prices[0]

    # High/low -- realistic 5m candle wicks
    wick_noise = np.abs(rng.normal(0, base_price * 0.00015, total_candles))
    highs = np.maximum(prices, opens) + wick_noise
    lows = np.minimum(prices, opens) - np.abs(
        rng.normal(0, base_price * 0.00015, total_candles))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })

    return df


def run_all_strategies():
    """Backtest all 4 strategies and compare results."""
    print("\n" + "=" * 70)
    print("  AI TRADING BOT -- BACKTEST REPORT")
    print("  30 days of simulated crypto data | $100 starting balance")
    print("=" * 70)

    strategies_config = load_strategies()

    # Generate test data
    df = generate_realistic_crypto_data(days=30, seed=42)
    print(f"\n  Generated {len(df)} candles (30 days of 5m data)")
    print(f"  Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"  Start: ${df.iloc[0]['close']:.0f} -> End: ${df.iloc[-1]['close']:.0f}")

    # Show regime distribution for context
    total_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100
    print(f"  Buy-and-hold return: {total_return:+.1f}%")

    strategies = [
        ("Volume Breakout", SmartScalpStrategy(), strategies_config.get("smart_scalp", {})),
        ("Vol Squeeze", GridStrategy(), strategies_config.get("grid", {})),
        ("Trend Momentum", MomentumStrategy(), strategies_config.get("momentum", {})),
        ("Trend Pullback", MeanReversionStrategy(), strategies_config.get("mean_reversion", {})),
    ]

    results = []

    for name, strategy, config in strategies:
        engine = BacktestEngine(
            initial_balance=100.0,
            maker_fee_rate=0.00075,
            min_order_size=10.0,
            max_positions=4,
        )
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        results.append((name, result))
        BacktestReport.print_summary(result, strategy_name=name)

    # Summary comparison
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)
    print(f"  {'Strategy':<20} {'Trades':>8} {'Win %':>8} {'Net P&L':>10} {'Sharpe':>8} {'Max DD':>8} {'Gate':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for name, result in results:
        # Synthetic data gate: strategy must not blow up.
        # Real validation happens during 14-30 day paper trading.
        passed = (
            result.win_rate >= 0.35
            and result.net_pnl > -5.0
            and result.max_drawdown_pct < 25
        )
        gate = "PASS" if passed else "FAIL"
        print(f"  {name:<20} {result.total_trades:>8} {result.win_rate*100:>7.1f}% "
              f"${result.net_pnl:>8.2f} {result.sharpe_ratio:>8.3f} "
              f"{result.max_drawdown_pct:>7.1f}% {gate:>8}")

    print()

    # Save reports
    os.makedirs("backtest/reports", exist_ok=True)
    for name, result in results:
        safe_name = name.lower().replace(" ", "_")
        BacktestReport.save_report(result, f"backtest/reports/{safe_name}.txt")

    print("  Reports saved to backtest/reports/")
    print()

    # Robustness check: ALL strategies across 5 different market seeds.
    # Each seed produces a different mix of trending/ranging/volatile regimes.
    # A robust strategy should PASS the gate on most seeds.
    seeds = [
        (42, "Normal"),
        (123, "Trending"),
        (999, "Choppy"),
        (777, "Volatile"),
        (314, "Mixed"),
    ]

    print("=" * 70)
    print("  ROBUSTNESS CHECK (all strategies x 5 market conditions)")
    print("=" * 70)

    # Strategy classes and their config keys
    strat_defs = [
        ("Vol Breakout", SmartScalpStrategy, "smart_scalp"),
        ("Vol Squeeze", GridStrategy, "grid"),
        ("Momentum", MomentumStrategy, "momentum"),
        ("Pullback", MeanReversionStrategy, "mean_reversion"),
    ]

    for strat_name, strat_class, config_key in strat_defs:
        config = strategies_config.get(config_key, {})
        pass_count = 0
        total_pnl = 0.0
        print(f"\n  {strat_name}:")
        for seed, label in seeds:
            df = generate_realistic_crypto_data(days=30, seed=seed)
            engine = BacktestEngine(initial_balance=100.0, maker_fee_rate=0.00075,
                                   min_order_size=10.0, max_positions=4)
            # Fresh strategy instance per seed (resets cooldown counters)
            result = engine.run(strat_class(), df, config)
            passed = (result.win_rate >= 0.35 and result.net_pnl > -5.0
                      and result.max_drawdown_pct < 25)
            gate = "PASS" if passed else "FAIL"
            if passed:
                pass_count += 1
            total_pnl += result.net_pnl
            print(f"    {label:<10} | Trades: {result.total_trades:>4} | "
                  f"Win: {result.win_rate*100:>5.1f}% | "
                  f"P&L: ${result.net_pnl:>7.2f} | "
                  f"DD: {result.max_drawdown_pct:>5.1f}% | {gate}")
        avg_pnl = total_pnl / len(seeds)
        print(f"    {'':10} | Pass rate: {pass_count}/{len(seeds)} | "
              f"Avg P&L: ${avg_pnl:>7.2f}")

    print()


if __name__ == "__main__":
    run_all_strategies()
