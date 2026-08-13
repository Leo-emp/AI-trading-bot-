# tests/test_executor.py
import pytest
from datetime import datetime, timezone

from src.execution.paper_trader import PaperTrader
from src.strategies.base import StrategySignal
from src.storage.models import Trade


class TestPaperTrader:

    def test_execute_buy_reduces_balance(self):
        """Executing a BUY should reduce USDT balance by position size + fees."""
        trader = PaperTrader(initial_balance=50.0, maker_fee_rate=0.00075)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        trade = trader.execute_signal(signal, "BTC/USDT", position_size=10.0)
        assert trade is not None
        assert trade.side == "buy"
        # Balance should decrease by ~$10 + fees
        assert trader.get_balance() < 50.0
        assert len(trader.get_open_positions()) == 1

    def test_rejects_order_below_minimum(self):
        """Should reject orders below $10 minimum."""
        trader = PaperTrader(initial_balance=50.0)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        trade = trader.execute_signal(signal, "BTC/USDT", position_size=5.0)
        assert trade is None  # rejected

    def test_check_positions_triggers_stop_loss(self):
        """When price hits stop-loss, position should close at a loss."""
        trader = PaperTrader(initial_balance=50.0, maker_fee_rate=0.00075)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        trader.execute_signal(signal, "BTC/USDT", position_size=10.0)
        # Price drops below stop loss
        closed = trader.check_open_positions({"BTC/USDT": 49700.0})
        assert len(closed) == 1
        assert closed[0].pnl < 0  # loss
        assert len(trader.get_open_positions()) == 0

    def test_check_positions_triggers_take_profit(self):
        """When price hits take-profit, position should close at a profit."""
        trader = PaperTrader(initial_balance=50.0, maker_fee_rate=0.00075)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        trader.execute_signal(signal, "BTC/USDT", position_size=10.0)
        # Price rises above take profit
        closed = trader.check_open_positions({"BTC/USDT": 50500.0})
        assert len(closed) == 1
        assert closed[0].pnl > 0  # profit
        assert closed[0].fees > 0  # fees were charged

    def test_fees_are_accurate(self):
        """Fees should match maker rate x 2 (entry + exit)."""
        fee_rate = 0.00075
        trader = PaperTrader(initial_balance=100.0, maker_fee_rate=fee_rate)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        trader.execute_signal(signal, "BTC/USDT", position_size=10.0)
        closed = trader.check_open_positions({"BTC/USDT": 50400.0})
        trade = closed[0]
        # Round-trip fees: entry_fee + exit_fee
        expected_fees = (10.0 * fee_rate) + (trade.quantity * 50400.0 * fee_rate)
        assert abs(trade.fees - expected_fees) < 0.01

    def test_respects_max_positions(self):
        """Should not open more than max_positions trades."""
        trader = PaperTrader(initial_balance=100.0, max_positions=3)
        signal = StrategySignal(
            direction="BUY", confidence=0.8,
            entry_price=50000.0, stop_loss=49800.0,
            take_profit=50400.0, strategy_name="smart_scalp",
        )
        for _ in range(3):
            trader.execute_signal(signal, "BTC/USDT", position_size=10.0)
        # 4th should be rejected
        trade = trader.execute_signal(signal, "ETH/USDT", position_size=10.0)
        assert trade is None
        assert len(trader.get_open_positions()) == 3
