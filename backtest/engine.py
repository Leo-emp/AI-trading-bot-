# backtest/engine.py
# Backtesting engine that simulates strategy performance on historical data.
# Uses the PaperTrader for realistic fee simulation.
# Walks through candles one by one, applying indicators and strategy logic.
# Returns comprehensive performance metrics.

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

    Walks through data candle by candle:
    1. Compute indicators on data up to current candle
    2. Ask strategy for a signal
    3. Execute via PaperTrader (with fees)
    4. Check stop-loss / take-profit on open positions
    5. Record results

    This is the MANDATORY gate — no strategy goes live without
    passing backtesting first.
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
        """Run a full backtest on historical data.

        df: historical OHLCV DataFrame (at least 100 candles for indicator warmup)
        config: strategy parameters from strategies.yaml
        pair: trading pair name for logging

        Returns BacktestResult with all performance metrics.
        """
        indicator_engine = IndicatorEngine()
        trader = PaperTrader(
            initial_balance=self._initial_balance,
            maker_fee_rate=self._fee_rate,
            min_order_size=self._min_order,
            max_positions=self._max_positions,
            time_exit_hours=999,  # disable time exit in backtest
        )

        all_closed_trades: list[Trade] = []
        equity_curve = [self._initial_balance]

        # Need at least 50 candles for indicator warmup
        warmup = 50

        # Walk through candles one by one
        for i in range(warmup, len(df)):
            # Use only data up to current candle (no lookahead bias)
            window = df.iloc[:i + 1].copy()

            # Compute indicators
            window = indicator_engine.compute_all(window)

            # Get current price for position checks
            current_price = window.iloc[-1]["close"]
            current_prices = {pair: current_price}

            # Check open positions for SL/TP hits
            closed = trader.check_open_positions(current_prices)
            all_closed_trades.extend(closed)

            # Ask strategy for a signal
            signal = strategy.evaluate(window, config)

            # Execute if we get a BUY or SELL
            if signal.direction != "HOLD":
                position_size = min(self._min_order, trader.get_balance() * 0.25)
                position_size = max(position_size, self._min_order)

                if position_size <= trader.get_balance():
                    trade = trader.execute_signal(signal, pair, position_size)

            equity_curve.append(trader.get_balance())

        # Close any remaining open positions at final price
        final_price = df.iloc[-1]["close"]
        remaining = trader.check_open_positions({pair: final_price})
        all_closed_trades.extend(remaining)

        # Calculate metrics
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
        losses = [t for t in trades if t.pnl <= 0]
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
