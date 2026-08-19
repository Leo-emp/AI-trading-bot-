# src/intelligence/trade_journal.py
# Trade Journal — records full context of every trade for adaptive learning.
#
# Unlike the performance tracker (which just counts wins/losses),
# the journal records WHY trades won or lost:
# - Max favorable excursion (MFE): how far price went in our favor
# - Max adverse excursion (MAE): how far price went against us
# - What regime was active when we entered
# - What confidence level the gate gave us
# - How long the trade lasted
# - What caused the exit (SL, TP, trailing, time)
#
# This data feeds into AdaptiveParams to learn:
# - "We keep getting stopped out then price recovers" → widen SL
# - "Price goes 3x our TP regularly" → widen TP
# - "Momentum trades in trending markets win 80%" → boost confidence
#
# Persists to JSON so learning survives restarts.

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

JOURNAL_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "trade_journal.json")


@dataclass
class JournalEntry:
    """Full context record for one trade."""
    trade_id: str
    pair: str
    side: str
    strategy: str
    regime: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_pct: float
    # Excursion data (filled during trade lifecycle)
    max_favorable_pct: float = 0.0    # max profit point reached
    max_adverse_pct: float = 0.0      # max drawdown point reached
    # Context
    confidence: float = 0.0           # gate confidence at entry
    atr_at_entry: float = 0.0         # ATR when trade opened
    sl_distance_pct: float = 0.0      # SL distance as % of entry
    tp_distance_pct: float = 0.0      # TP distance as % of entry
    # Outcome
    close_reason: str = ""            # stop_loss, take_profit, trailing, time_exit
    duration_seconds: float = 0.0     # how long trade was open
    opened_at: float = 0.0            # timestamp
    closed_at: float = 0.0            # timestamp


class TradeJournal:
    """Records complete trade lifecycle for adaptive learning.

    Two responsibilities:
    1. Track live positions (MFE/MAE updates each cycle)
    2. Store completed entries for AdaptiveParams to learn from
    """

    def __init__(self, max_entries: int = 500):
        self._max_entries = max_entries
        # Active position tracking (pos_id -> partial entry)
        self._active: dict[str, JournalEntry] = {}
        # Completed trade history
        self._history: list[dict] = []
        self._load()

    def open_trade(self, pos_id: str, pair: str, side: str,
                   strategy: str, regime: str, entry_price: float,
                   stop_loss: float, take_profit: float,
                   position_size: float, confidence: float,
                   atr: float):
        """Register a new trade for tracking."""
        sl_dist = abs(entry_price - stop_loss) / entry_price * 100
        tp_dist = abs(take_profit - entry_price) / entry_price * 100

        entry = JournalEntry(
            trade_id=pos_id,
            pair=pair,
            side=side,
            strategy=strategy,
            regime=regime,
            entry_price=entry_price,
            exit_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            pnl=0.0,
            pnl_pct=0.0,
            confidence=confidence,
            atr_at_entry=atr,
            sl_distance_pct=sl_dist,
            tp_distance_pct=tp_dist,
            opened_at=time.time(),
        )
        self._active[pos_id] = entry

    def update_price(self, pos_id: str, current_price: float):
        """Update MFE/MAE tracking for an active position."""
        entry = self._active.get(pos_id)
        if not entry:
            return

        if entry.side == "buy":
            favorable_pct = (current_price - entry.entry_price) / entry.entry_price * 100
            adverse_pct = (entry.entry_price - current_price) / entry.entry_price * 100
        else:
            favorable_pct = (entry.entry_price - current_price) / entry.entry_price * 100
            adverse_pct = (current_price - entry.entry_price) / entry.entry_price * 100

        # Track maximums
        if favorable_pct > entry.max_favorable_pct:
            entry.max_favorable_pct = favorable_pct
        if adverse_pct > entry.max_adverse_pct:
            entry.max_adverse_pct = adverse_pct

    def close_trade(self, pos_id: str, exit_price: float,
                    pnl: float, close_reason: str) -> Optional[dict]:
        """Close a tracked trade and record the outcome.

        Returns the completed journal entry as a dict for AdaptiveParams.
        """
        entry = self._active.pop(pos_id, None)
        if not entry:
            return None

        entry.exit_price = exit_price
        entry.pnl = pnl
        entry.pnl_pct = pnl / entry.position_size * 100 if entry.position_size > 0 else 0
        entry.close_reason = close_reason
        entry.closed_at = time.time()
        entry.duration_seconds = entry.closed_at - entry.opened_at

        # Convert to dict for storage
        record = asdict(entry)
        self._history.append(record)

        # Cap history size
        if len(self._history) > self._max_entries:
            self._history = self._history[-self._max_entries:]

        self._save()

        logger.info(
            "JOURNAL: %s %s %s | P&L: $%.2f (%.2f%%) | MFE: %.2f%% MAE: %.2f%% | "
            "Duration: %.0fm | Exit: %s",
            entry.pair, entry.side, entry.strategy,
            pnl, entry.pnl_pct,
            entry.max_favorable_pct, entry.max_adverse_pct,
            entry.duration_seconds / 60, close_reason,
        )

        return record

    def get_history(self, n: int = 100) -> list[dict]:
        """Get recent trade history for adaptive learning."""
        return self._history[-n:]

    def get_regime_stats(self, regime: str) -> dict:
        """Get performance stats for a specific regime."""
        regime_trades = [t for t in self._history if t.get("regime") == regime]
        if not regime_trades:
            return {"count": 0, "win_rate": 0.5}

        wins = sum(1 for t in regime_trades if t["pnl"] > 0)
        return {
            "count": len(regime_trades),
            "win_rate": wins / len(regime_trades),
            "avg_pnl_pct": sum(t["pnl_pct"] for t in regime_trades) / len(regime_trades),
            "avg_mfe": sum(t["max_favorable_pct"] for t in regime_trades) / len(regime_trades),
            "avg_mae": sum(t["max_adverse_pct"] for t in regime_trades) / len(regime_trades),
        }

    def get_strategy_stats(self, strategy: str) -> dict:
        """Get performance stats for a specific strategy."""
        strat_trades = [t for t in self._history if t.get("strategy") == strategy]
        if not strat_trades:
            return {"count": 0, "win_rate": 0.5}

        wins = sum(1 for t in strat_trades if t["pnl"] > 0)
        return {
            "count": len(strat_trades),
            "win_rate": wins / len(strat_trades),
            "avg_pnl_pct": sum(t["pnl_pct"] for t in strat_trades) / len(strat_trades),
            "avg_duration_min": sum(t["duration_seconds"] for t in strat_trades) / len(strat_trades) / 60,
        }

    @property
    def total_trades(self) -> int:
        return len(self._history)

    def _save(self):
        """Persist journal to disk."""
        try:
            os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
            with open(JOURNAL_FILE, "w") as f:
                json.dump(self._history, f)
        except Exception as e:
            logger.debug("Failed to save trade journal: %s", e)

    def _load(self):
        """Load journal from disk."""
        try:
            if os.path.exists(JOURNAL_FILE):
                with open(JOURNAL_FILE, "r") as f:
                    self._history = json.load(f)
                logger.info("Trade journal loaded: %d historical trades", len(self._history))
        except Exception as e:
            logger.debug("No trade journal found: %s", e)
            self._history = []
