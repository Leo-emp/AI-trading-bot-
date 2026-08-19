# src/storage/models.py
# Data models for all database records.
# Every model is a frozen dataclass — immutable after creation.
# P&L always includes fees (net, never gross).

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class Trade:
    """A single completed trade record.

    pnl is NET (after fees). fees is the total fees paid for this
    round-trip (entry + exit). status is 'open' or 'closed'.
    """
    timestamp: datetime
    pair: str           # e.g. "BTC/USDT"
    side: str           # "buy" or "sell"
    strategy: str       # which strategy generated this trade
    entry_price: float
    exit_price: float
    quantity: float     # asset quantity (e.g. 0.0002 BTC)
    pnl: float          # net P&L in quote currency (USDT) after fees
    fees: float         # total fees paid (both sides)
    status: str         # "open" or "closed"
    close_reason: str = ""  # stop_loss, take_profit, partial_1r, time_exit, force_close
    id: Optional[int] = None


@dataclass(frozen=True)
class AIDecision:
    """A logged AI decision for post-analysis.

    Every time the trade gate evaluates, we log what each brain said
    and what action was taken. This lets us analyze decision quality
    over time.
    """
    timestamp: datetime
    regime: str              # BULLISH/BEARISH/SIDEWAYS/VOLATILE/CRASH
    gemini_confidence: float # 0-100
    ml_signal: str           # BUY/SELL/HOLD
    ml_probability: float    # 0.0-1.0
    action_taken: str        # what the bot actually did
    id: Optional[int] = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Hourly portfolio state for tracking equity curve."""
    timestamp: datetime
    total_balance: float       # total in USDT
    unrealized_pnl: float      # P&L on open positions
    open_positions_count: int


@dataclass
class StrategyStats:
    """Rolling performance statistics for one strategy.

    Computed from the most recent trades (rolling window).
    Used by the self-learner to score and rank strategies.
    """
    strategy_name: str
    total_trades: int
    win_rate: float          # 0.0 to 1.0
    avg_pnl: float           # average net P&L per trade
    avg_win: float           # average winning trade P&L
    avg_loss: float          # average losing trade P&L (negative)
    total_fees: float        # total fees paid
    sharpe_ratio: float      # risk-adjusted return
    max_drawdown: float      # worst peak-to-trough
    score: float = 0.0       # composite score for ranking
