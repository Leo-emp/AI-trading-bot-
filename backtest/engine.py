# backtest/engine.py
# Backtesting engine that simulates strategy performance on historical data.
# Uses the PaperTrader for realistic fee simulation.
#
# OPTIMIZATION: Computes indicators ONCE on the full dataset upfront,
# then walks through pre-computed rows. Previous version recomputed
# indicators 8500+ times (once per candle), causing 5+ minute timeouts.

from dataclasses import dataclass, field
import pandas as pd
import logging

from src.data.indicators import IndicatorEngine
from src.execution.paper_trader import PaperTrader
from src.strategies.base import BaseStrategy
from src.storage.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    final_balance: float = 0.0
    trades: list[Trade] = field(default_factory=list)


class BacktestEngine:
    """Simulates trading strategy on historical OHLCV data.

    1. Compute indicators ONCE on full dataset (fast)
    2. Walk candle by candle, slicing pre-computed data
    3. Check SL/TP using candle high/low (realistic)
    4. Execute via PaperTrader (with fees)
    """

    def __init__(self, initial_balance: float = 50.0,
                 maker_fee_rate: float = 0.00075,
                 min_order_size: float = 10.0,
                 max_positions: int = 3):
        self._initial_balance = initial_balance
        self._fee_rate = maker_fee_rate
        self._min_order = min_order_size
        self._max_positions = max_positions

    def run(self, strategy: BaseStrategy, df: pd.DataFrame,
            config: dict, pair: str = "BTC/USDT") -> BacktestResult:
        """Run a full backtest on historical data."""
        indicator_engine = IndicatorEngine()
        trader = PaperTrader(
            initial_balance=self._initial_balance,
            maker_fee_rate=self._fee_rate,
            min_order_size=self._min_order,
            max_positions=self._max_positions,
            time_exit_hours=999,  # disable time exit in backtest
        )

        # Compute indicators ONCE on full dataset — the big optimization.
        # Rolling indicators (RSI, EMA, BB, ATR) produce the same values
        # whether computed on the full set or on a growing window.
        df_with_indicators = indicator_engine.compute_all(df.copy())

        all_closed_trades: list[Trade] = []
        equity_curve = [self._initial_balance]

        # Need at least 50 candles for indicator warmup
        warmup = 50
        # Strategy looks back up to 30 candles for patterns
        lookback = 30

        for i in range(warmup, len(df_with_indicators)):
            # Slice pre-computed data — strategy gets a window of recent candles
            start = max(0, i - lookback)
            window = df_with_indicators.iloc[start:i + 1]

            # Use candle high/low for realistic SL/TP simulation
            current_row = df_with_indicators.iloc[i]
            current_price = current_row["close"]
            candle_high = current_row["high"]
            candle_low = current_row["low"]

            # Check SL/TP on open positions FIRST (before new signals)
            closed = trader.check_open_positions(
                {pair: current_price},
                highs={pair: candle_high},
                lows={pair: candle_low},
            )
            all_closed_trades.extend(closed)

            # Ask strategy for a signal
            signal = strategy.evaluate(window, config)

            # Execute if we get a BUY or SELL
            if signal.direction != "HOLD":
                position_size = min(self._min_order, trader.get_balance() * 0.25)
                position_size = max(position_size, self._min_order)

                if position_size <= trader.get_balance():
                    trader.execute_signal(signal, pair, position_size)

            # Track TOTAL equity (free cash + unrealized position value)
            # Without this, drawdown looks huge when positions are open
            unrealized = 0.0
            for pos in trader.get_open_positions():
                if pos.side == "buy":
                    unrealized += (current_price - pos.entry_price) * pos.quantity
                else:
                    unrealized += (pos.entry_price - current_price) * pos.quantity
                unrealized += pos.quantity * pos.entry_price  # capital in position
            equity_curve.append(trader.get_balance() + unrealized)

        # Force-close ALL remaining open positions at final price
        final_price = df.iloc[-1]["close"]
        remaining = trader.force_close_all({pair: final_price})
        all_closed_trades.extend(remaining)

        return self._compute_metrics(all_closed_trades, equity_curve, trader.get_balance())

    def _compute_metrics(self, trades: list[Trade],
                         equity_curve: list[float],
                         final_balance: float) -> BacktestResult:
        """Calculate performance metrics from trade list."""
        if not trades:
            return BacktestResult(
                final_balance=final_balance,
                net_pnl=final_balance - self._initial_balance,
            )

        wins = [t for t in trades if t.pnl > 0]
        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0

        net_pnl = sum(t.pnl for t in trades)
        gross_pnl = net_pnl + sum(t.fees for t in trades)
        total_fees = sum(t.fees for t in trades)

        # Sharpe ratio from equity curve
        if len(equity_curve) > 1:
            returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                       for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
            if returns:
                mean_r = sum(returns) / len(returns)
                std_r = (sum((r - mean_r)**2 for r in returns) / len(returns)) ** 0.5
                sharpe = mean_r / std_r if std_r > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            peak = max(peak, val)
            if peak > 0:
                dd = (peak - val) / peak * 100
                max_dd = max(max_dd, dd)

        return BacktestResult(
            total_trades=total,
            win_rate=win_rate,
            net_pnl=net_pnl,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            final_balance=final_balance,
            trades=trades,
        )
