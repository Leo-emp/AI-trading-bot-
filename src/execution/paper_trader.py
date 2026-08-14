# src/execution/paper_trader.py
# Simulated trading engine for paper trading mode.
# Mirrors exactly how live trading would work, but with virtual money.
# Includes accurate fee simulation (maker rate with BNB discount).
# Every trade is logged identically to live — same Trade dataclass.
#
# The paper trader tracks:
# - Virtual USDT balance
# - Open positions with entry price, SL, TP
# - Trade history with accurate P&L after fees

import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from src.strategies.base import StrategySignal
from src.storage.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """A currently open simulated position."""
    pair: str
    side: str            # "buy" or "sell"
    entry_price: float
    quantity: float      # asset quantity
    stop_loss: float
    take_profit: float
    strategy: str
    opened_at: float     # time.time() for age tracking
    entry_fee: float     # fee paid on entry


class PaperTrader:
    """Simulated trading engine with realistic fee simulation.

    Behaves identically to live trading. Used for:
    1. Testing strategies with zero financial risk
    2. Collecting data for ML model training (30+ days)
    3. Proving profitability before going live
    """

    def __init__(self, initial_balance: float = 50.0,
                 maker_fee_rate: float = 0.00075,
                 min_order_size: float = 10.0,
                 max_positions: int = 3,
                 max_per_pair: int = 2,
                 time_exit_hours: float = 4.0):
        self._balance = initial_balance
        self._fee_rate = maker_fee_rate
        self._min_order = min_order_size
        self._max_positions = max_positions
        # Cap per-pair concentration — allows scaling in (2 entries)
        # but prevents going all-in on one asset with a $100 account
        self._max_per_pair = max_per_pair
        self._time_exit = time_exit_hours * 3600  # convert to seconds
        self._positions: list[OpenPosition] = []
        self._trade_history: list[Trade] = []

    def get_balance(self) -> float:
        """Current USDT balance."""
        return self._balance

    def get_open_positions(self) -> list[OpenPosition]:
        """All currently open positions."""
        return list(self._positions)

    def execute_signal(self, signal: StrategySignal, pair: str,
                       position_size: float) -> Optional[Trade]:
        """Execute a simulated trade based on a strategy signal.

        Returns the Trade record if executed, or None if rejected.
        """
        if signal.direction == "HOLD":
            return None

        # --- Validation ---
        if position_size < self._min_order:
            logger.info("Rejected: position $%.2f below minimum $%.2f", position_size, self._min_order)
            return None

        if len(self._positions) >= self._max_positions:
            logger.info("Rejected: max positions (%d) reached", self._max_positions)
            return None

        # Cap per-pair positions — allows scaling in but limits concentration
        pair_count = sum(1 for pos in self._positions if pos.pair == pair)
        if pair_count >= self._max_per_pair:
            logger.info("Rejected: already have %d/%d positions on %s",
                         pair_count, self._max_per_pair, pair)
            return None

        if position_size > self._balance:
            logger.info("Rejected: insufficient balance ($%.2f < $%.2f)", self._balance, position_size)
            return None

        # --- Calculate entry ---
        entry_price = signal.entry_price
        quantity = position_size / entry_price  # how much asset we're buying
        entry_fee = position_size * self._fee_rate  # fee on entry side

        # Deduct position size + entry fee from balance
        self._balance -= (position_size + entry_fee)

        # Create open position
        pos = OpenPosition(
            pair=pair,
            side=signal.direction.lower(),
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy_name,
            opened_at=time.time(),
            entry_fee=entry_fee,
        )
        self._positions.append(pos)

        logger.info(
            "PAPER %s %s: %.8f @ $%.2f (SL: $%.2f, TP: $%.2f) [fee: $%.4f]",
            signal.direction, pair, quantity, entry_price,
            signal.stop_loss, signal.take_profit, entry_fee,
        )

        # Return an "open" trade record
        return Trade(
            timestamp=datetime.now(timezone.utc),
            pair=pair, side=signal.direction.lower(),
            strategy=signal.strategy_name,
            entry_price=entry_price, exit_price=0.0,
            quantity=quantity, pnl=0.0,
            fees=entry_fee, status="open",
        )

    def check_open_positions(self, current_prices: dict,
                             highs: dict = None,
                             lows: dict = None) -> list[Trade]:
        """Check all open positions against current prices.

        Closes positions that hit stop-loss, take-profit, or time limit.
        When highs/lows are provided (backtest mode), uses the candle's
        high and low to check SL/TP — a candle's low might hit SL even
        if the close price is above it.

        Returns list of closed Trade records.
        """
        closed_trades = []
        still_open = []
        highs = highs or {}
        lows = lows or {}

        for pos in self._positions:
            price = current_prices.get(pos.pair)
            if price is None:
                still_open.append(pos)
                continue

            # Use candle extremes when available (backtest), else just close
            candle_low = lows.get(pos.pair, price)
            candle_high = highs.get(pos.pair, price)

            close_reason = None
            exit_price = price

            # --- Check stop-loss using candle low/high ---
            # SL is checked FIRST (if both SL and TP hit in same candle,
            # assume worst case: SL hit first)
            if pos.side == "buy" and candle_low <= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss
            elif pos.side == "sell" and candle_high >= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss

            # --- Check take-profit using candle high/low ---
            # Only check TP if SL wasn't already hit
            if close_reason is None:
                if pos.side == "buy" and candle_high >= pos.take_profit:
                    close_reason = "take_profit"
                    exit_price = pos.take_profit
                elif pos.side == "sell" and candle_low <= pos.take_profit:
                    close_reason = "take_profit"
                    exit_price = pos.take_profit

            # --- Check time exit (4 hours) ---
            if close_reason is None and time.time() - pos.opened_at > self._time_exit:
                close_reason = "time_exit"

            if close_reason:
                trade = self._close_position(pos, exit_price, close_reason)
                closed_trades.append(trade)
            else:
                still_open.append(pos)

        self._positions = still_open
        return closed_trades

    def force_close_all(self, current_prices: dict) -> list[Trade]:
        """Force-close all open positions at current market price.
        Used at end of backtest to return all capital to balance.
        """
        closed = []
        for pos in self._positions:
            price = current_prices.get(pos.pair, pos.entry_price)
            trade = self._close_position(pos, price, "force_close")
            closed.append(trade)
        self._positions = []
        return closed

    def _close_position(self, pos: OpenPosition, exit_price: float,
                         reason: str) -> Trade:
        """Close a position and calculate P&L after fees."""
        exit_notional = pos.quantity * exit_price
        exit_fee = exit_notional * self._fee_rate
        total_fees = pos.entry_fee + exit_fee

        # Calculate P&L
        if pos.side == "buy":
            # Bought low, selling high = profit
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            # Sold high, buying back low = profit
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_fees

        # Return notional + P&L to balance
        entry_notional = pos.quantity * pos.entry_price
        self._balance += entry_notional + gross_pnl - exit_fee

        logger.info(
            "PAPER CLOSE %s %s @ $%.2f -> $%.2f | P&L: $%.4f (fees: $%.4f) [%s]",
            pos.side.upper(), pos.pair, pos.entry_price, exit_price,
            net_pnl, total_fees, reason,
        )

        return Trade(
            timestamp=datetime.now(timezone.utc),
            pair=pos.pair, side=pos.side,
            strategy=pos.strategy,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=net_pnl, fees=total_fees,
            status="closed",
        )
