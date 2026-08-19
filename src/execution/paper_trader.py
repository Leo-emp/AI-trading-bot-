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
#
# STATE PERSISTENCE: saves balance + positions to JSON after every
# trade open/close. Survives restarts, watchdog crashes, weekly reboots.

import json
import logging
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from src.strategies.base import StrategySignal
from src.storage.models import Trade

logger = logging.getLogger(__name__)

# State file location — next to the database in data/
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_state.json")


@dataclass
class OpenPosition:
    """A currently open simulated position."""
    pair: str
    side: str            # "buy" or "sell"
    entry_price: float
    quantity: float      # asset quantity (based on NOTIONAL, not margin)
    stop_loss: float
    take_profit: float
    strategy: str
    opened_at: float     # time.time() for age tracking
    entry_fee: float     # fee paid on entry
    partial_taken: bool = False  # has 50% profit been taken at 1R?
    leverage: float = 1.0       # futures leverage (1.0 = spot-like)
    margin_locked: float = 0.0  # cash actually reserved (notional / leverage)


class PaperTrader:
    """Simulated trading engine with realistic fee simulation.

    Behaves identically to live trading. Used for:
    1. Testing strategies with zero financial risk
    2. Collecting data for ML model training (30+ days)
    3. Proving profitability before going live
    """

    def __init__(self, initial_balance: float = 50.0,
                 maker_fee_rate: float = 0.0002,
                 min_order_size: float = 10.0,
                 max_positions: int = 3,
                 max_per_pair: int = 2,
                 time_exit_hours: float = 4.0):
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._fee_rate = maker_fee_rate
        self._min_order = min_order_size
        self._max_positions = max_positions
        self._max_per_pair = max_per_pair
        self._time_exit = time_exit_hours * 3600
        self._positions: list[OpenPosition] = []
        self._trade_history: list[Trade] = []
        self._total_trades = 0
        self._total_wins = 0

        # Try to restore saved state from disk
        self._load_state()

    def get_balance(self) -> float:
        """Current USDT balance."""
        return self._balance

    def get_initial_balance(self) -> float:
        """Balance at the start of paper trading (doesn't change on restart)."""
        return self._initial_balance

    def get_open_positions(self) -> list[OpenPosition]:
        """All currently open positions."""
        return list(self._positions)

    def get_equity(self, current_prices: dict = None) -> float:
        """P0.2: Mark-to-market account equity.

        With leverage: only margin was deducted from cash, so we add back
        margin + unrealized P&L (not full notional).

        equity = cash + sum(margin_locked + unrealized_pnl) for each position
        """
        current_prices = current_prices or {}
        equity = self._balance

        for pos in self._positions:
            price = current_prices.get(pos.pair, pos.entry_price)
            if pos.side == "buy":
                unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                unrealized_pnl = (pos.entry_price - price) * pos.quantity
            # margin_locked is what was deducted; add it back + P&L
            locked = pos.margin_locked if pos.margin_locked > 0 else pos.entry_price * pos.quantity
            equity += locked + unrealized_pnl

        return equity

    def get_unrealized_pnl(self, current_prices: dict = None) -> float:
        """P0.2: Total unrealized P&L across all open positions."""
        current_prices = current_prices or {}
        pnl = 0.0
        for pos in self._positions:
            price = current_prices.get(pos.pair, pos.entry_price)
            if pos.side == "buy":
                pnl += (price - pos.entry_price) * pos.quantity
            else:
                pnl += (pos.entry_price - price) * pos.quantity
        return pnl

    def execute_signal(self, signal: StrategySignal, pair: str,
                       position_size: float,
                       leverage: int = 1) -> Optional[Trade]:
        """Execute a simulated trade based on a strategy signal.

        With futures leverage, only margin (notional / leverage) is deducted
        from balance. The full notional determines quantity and P&L.

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

        pair_count = sum(1 for pos in self._positions if pos.pair == pair)
        if pair_count >= self._max_per_pair:
            logger.info("Rejected: already have %d/%d positions on %s",
                         pair_count, self._max_per_pair, pair)
            return None

        # With leverage: only margin is locked, not full notional
        margin = position_size / leverage
        entry_fee = position_size * self._fee_rate

        if margin + entry_fee > self._balance:
            logger.info("Rejected: insufficient margin ($%.2f needed, $%.2f available)",
                       margin + entry_fee, self._balance)
            return None

        # --- Calculate entry ---
        entry_price = signal.entry_price
        quantity = position_size / entry_price

        # Deduct MARGIN + fee from balance (not full notional)
        self._balance -= (margin + entry_fee)

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
            leverage=float(leverage),
            margin_locked=margin,
        )
        self._positions.append(pos)

        logger.info(
            "PAPER %s %s: %.8f @ $%.2f | notional=$%.0f margin=$%.0f %dx | "
            "SL=$%.2f TP=$%.2f [fee=$%.4f]",
            signal.direction, pair, quantity, entry_price,
            position_size, margin, leverage,
            signal.stop_loss, signal.take_profit, entry_fee,
        )

        # Persist state after opening
        self._save_state()

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
        Also handles partial profit-taking at 1R (50% of position closed
        when profit reaches the SL distance — locks in guaranteed win).
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

            candle_low = lows.get(pos.pair, price)
            candle_high = highs.get(pos.pair, price)

            close_reason = None
            exit_price = price

            # --- Liquidation check (leveraged positions) ---
            # With isolated margin, if loss exceeds margin the exchange
            # force-closes. SL should always trigger first, but this is
            # the safety net for flash crashes that gap through SL.
            if pos.leverage > 1.0:
                maint_rate = 0.004
                if pos.side == "buy":
                    liq_price = pos.entry_price * (1 - (1 / pos.leverage) * (1 - maint_rate))
                    if candle_low <= liq_price:
                        close_reason = "liquidation"
                        exit_price = liq_price
                        logger.critical(
                            "LIQUIDATED %s BUY | low=%.2f <= liq=%.2f | %dx leverage",
                            pos.pair, candle_low, liq_price, int(pos.leverage),
                        )
                else:
                    liq_price = pos.entry_price * (1 + (1 / pos.leverage) * (1 - maint_rate))
                    if candle_high >= liq_price:
                        close_reason = "liquidation"
                        exit_price = liq_price
                        logger.critical(
                            "LIQUIDATED %s SELL | high=%.2f >= liq=%.2f | %dx leverage",
                            pos.pair, candle_high, liq_price, int(pos.leverage),
                        )

            # --- Partial profit-taking at 2R (lock in profit, let rest run) ---
            # Take 25% at 2R instead of 50% at 1R.
            # Old: 50% at 1R capped upside → average win 0.5R vs 1R loss = negative EV
            # New: 25% at 2R → guaranteed 0.5R banked, 75% rides for 3R-5R moves
            if not pos.partial_taken:
                sl_distance = abs(pos.entry_price - pos.stop_loss)
                if pos.side == "buy":
                    target_2r = pos.entry_price + sl_distance * 2.0
                    if price >= target_2r:
                        partial_trade = self._take_partial_profit(pos, price)
                        if partial_trade:
                            closed_trades.append(partial_trade)
                else:
                    target_2r = pos.entry_price - sl_distance * 2.0
                    if price <= target_2r:
                        partial_trade = self._take_partial_profit(pos, price)
                        if partial_trade:
                            closed_trades.append(partial_trade)

            # P0.7: SL checked first (worst case if both hit same candle)
            if pos.side == "buy" and candle_low <= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss
                logger.warning(
                    "SL HIT BUY %s | low=%.2f <= SL=%.2f",
                    pos.pair, candle_low, pos.stop_loss,
                )
            elif pos.side == "sell" and candle_high >= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss
                logger.warning(
                    "SL HIT SELL %s | high=%.2f >= SL=%.2f",
                    pos.pair, candle_high, pos.stop_loss,
                )

            if close_reason is None:
                if pos.side == "buy" and candle_high >= pos.take_profit:
                    close_reason = "take_profit"
                    exit_price = pos.take_profit
                    logger.info(
                        "TP HIT BUY %s | high=%.2f >= TP=%.2f",
                        pos.pair, candle_high, pos.take_profit,
                    )
                elif pos.side == "sell" and candle_low <= pos.take_profit:
                    close_reason = "take_profit"
                    exit_price = pos.take_profit
                    logger.info(
                        "TP HIT SELL %s | low=%.2f <= TP=%.2f",
                        pos.pair, candle_low, pos.take_profit,
                    )

            # Time exit — gives trades time to develop
            if close_reason is None and time.time() - pos.opened_at > self._time_exit:
                close_reason = "time_exit"
                age_h = (time.time() - pos.opened_at) / 3600
                logger.info(
                    "TIME EXIT %s %s | age=%.1fh >= limit=%.1fh",
                    pos.side.upper(), pos.pair, age_h, self._time_exit / 3600,
                )

            if close_reason:
                trade = self._close_position(pos, exit_price, close_reason)
                closed_trades.append(trade)
            else:
                still_open.append(pos)

        self._positions = still_open

        if closed_trades:
            self._save_state()

        return closed_trades

    def _take_partial_profit(self, pos: OpenPosition, current_price: float) -> Optional[Trade]:
        """Close 25% of position at 2R profit. Banks 0.5R guaranteed."""
        partial_qty = pos.quantity * 0.25
        if partial_qty * current_price < 5.0:
            return None

        partial_notional = partial_qty * current_price
        exit_fee = partial_notional * self._fee_rate
        entry_fee_partial = pos.entry_fee * 0.25

        if pos.side == "buy":
            gross_pnl = (current_price - pos.entry_price) * partial_qty
        else:
            gross_pnl = (pos.entry_price - current_price) * partial_qty

        net_pnl = gross_pnl - exit_fee - entry_fee_partial

        # Return partial margin + profit (leverage-aware)
        partial_margin = pos.margin_locked * 0.25 if pos.margin_locked > 0 else partial_qty * pos.entry_price
        self._balance += partial_margin + gross_pnl - exit_fee

        # Reduce position (keep 75% running)
        pos.quantity -= partial_qty
        pos.entry_fee *= 0.75
        if pos.margin_locked > 0:
            pos.margin_locked *= 0.75
        pos.partial_taken = True

        self._total_trades += 1
        if net_pnl > 0:
            self._total_wins += 1

        logger.info(
            "PARTIAL TAKE 25%% %s %s @ $%.2f | P&L: $%.2f [2R hit] | Remaining: %.4f units (75%%)",
            pos.side.upper(), pos.pair, current_price, net_pnl, pos.quantity,
        )

        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            pair=pos.pair, side=pos.side,
            strategy=pos.strategy,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=partial_qty,
            pnl=net_pnl, fees=exit_fee + entry_fee_partial,
            status="partial_close",
            close_reason="partial_2r",
        )
        return trade

    def force_close_all(self, current_prices: dict) -> list[Trade]:
        """Force-close all open positions at current market price."""
        closed = []
        for pos in self._positions:
            price = current_prices.get(pos.pair, pos.entry_price)
            trade = self._close_position(pos, price, "force_close")
            closed.append(trade)
        self._positions = []
        self._save_state()
        return closed

    def _close_position(self, pos: OpenPosition, exit_price: float,
                         reason: str) -> Trade:
        """Close a position and calculate P&L after fees.

        With leverage: return margin + P&L (not full notional).
        Isolated margin: balance_return can't go below 0 (you lose
        at most your margin, never more).
        """
        exit_notional = pos.quantity * exit_price
        exit_fee = exit_notional * self._fee_rate
        total_fees = pos.entry_fee + exit_fee

        if pos.side == "buy":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_fees

        # Return margin + P&L to balance (isolated margin: floor at 0)
        locked = pos.margin_locked if pos.margin_locked > 0 else pos.entry_price * pos.quantity
        balance_return = locked + gross_pnl - exit_fee
        if balance_return < 0:
            balance_return = 0
        self._balance += balance_return

        # Track stats
        self._total_trades += 1
        if net_pnl > 0:
            self._total_wins += 1

        logger.info(
            "PAPER CLOSE %s %s @ $%.2f -> $%.2f | P&L: $%.4f (fees: $%.4f) [%s] | Balance: $%.2f",
            pos.side.upper(), pos.pair, pos.entry_price, exit_price,
            net_pnl, total_fees, reason, self._balance,
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
            close_reason=reason,
        )

    def _save_state(self):
        """Persist balance + positions to disk. Survives restarts."""
        try:
            state = {
                "balance": self._balance,
                "initial_balance": self._initial_balance,
                "total_trades": self._total_trades,
                "total_wins": self._total_wins,
                "saved_at": time.time(),
                "positions": [
                    {
                        "pair": p.pair, "side": p.side,
                        "entry_price": p.entry_price, "quantity": p.quantity,
                        "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                        "strategy": p.strategy, "opened_at": p.opened_at,
                        "entry_fee": p.entry_fee,
                        "partial_taken": p.partial_taken,
                        "leverage": p.leverage,
                        "margin_locked": p.margin_locked,
                    }
                    for p in self._positions
                ],
            }
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("Failed to save paper trader state: %s", e)

    def _load_state(self):
        """Restore balance + positions from disk if available."""
        try:
            if not os.path.exists(STATE_FILE):
                logger.info("No saved state, starting fresh at $%.2f", self._balance)
                return

            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            self._balance = state["balance"]
            self._initial_balance = state.get("initial_balance", self._initial_balance)
            self._total_trades = state.get("total_trades", 0)
            self._total_wins = state.get("total_wins", 0)

            for p in state.get("positions", []):
                # Backward compat: old positions without leverage default to 1× (spot)
                lev = p.get("leverage", 1.0)
                margin = p.get("margin_locked", 0.0)
                if margin <= 0:
                    margin = p["entry_price"] * p["quantity"]
                pos = OpenPosition(
                    pair=p["pair"], side=p["side"],
                    entry_price=p["entry_price"], quantity=p["quantity"],
                    stop_loss=p["stop_loss"], take_profit=p["take_profit"],
                    strategy=p["strategy"], opened_at=p["opened_at"],
                    entry_fee=p["entry_fee"],
                    partial_taken=p.get("partial_taken", False),
                    leverage=lev,
                    margin_locked=margin,
                )
                self._positions.append(pos)

            age_sec = time.time() - state.get("saved_at", time.time())
            logger.info(
                "Restored paper state: $%.2f balance, %d open positions, "
                "%d total trades (%d wins), saved %.0fs ago",
                self._balance, len(self._positions),
                self._total_trades, self._total_wins, age_sec,
            )

            # Resave if positions were loaded with default leverage
            # (migrates old state files to include leverage/margin_locked)
            if self._positions:
                self._save_state()
        except Exception as e:
            logger.error("Failed to load paper state: %s (starting fresh)", e)
