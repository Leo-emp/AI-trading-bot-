# src/storage/database.py
# Async SQLite database for trade logging, P&L tracking, and AI decision history.
# All timestamps stored as ISO-8601 UTC strings.
# Uses aiosqlite for non-blocking I/O in the async trading loop.

import aiosqlite
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.storage.models import Trade, AIDecision, PortfolioSnapshot, StrategyStats


class Database:
    """Async SQLite database for the trading bot.

    Stores trades, AI decisions, and portfolio snapshots.
    Provides computed metrics: daily P&L, consecutive losses,
    strategy stats, drawdown calculations.
    """

    def __init__(self, db_path: str = "trading_bot.db"):
        # Path to the SQLite database file
        self._db_path = db_path
        # Connection handle — set by init()
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        """Create tables if they don't exist and open connection."""
        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance
        await self._conn.execute("PRAGMA journal_mode=WAL")

        # trades: every executed trade with full P&L breakdown
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                strategy TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                fees REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)

        # ai_decisions: what each brain said and what we did
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                gemini_confidence REAL NOT NULL,
                ml_signal TEXT NOT NULL,
                ml_probability REAL NOT NULL,
                action_taken TEXT NOT NULL
            )
        """)

        # portfolio_snapshots: hourly equity curve
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_balance REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                open_positions_count INTEGER NOT NULL
            )
        """)

        await self._conn.commit()

    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()

    async def log_trade(self, trade: Trade):
        """Insert a trade record."""
        await self._conn.execute(
            """INSERT INTO trades
               (timestamp, pair, side, strategy, entry_price, exit_price,
                quantity, pnl, fees, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.timestamp.isoformat(),
                trade.pair, trade.side, trade.strategy,
                trade.entry_price, trade.exit_price, trade.quantity,
                trade.pnl, trade.fees, trade.status,
            ),
        )
        await self._conn.commit()

    async def get_recent_trades(self, limit: int = 50) -> list[Trade]:
        """Fetch the most recent trades, newest first."""
        cursor = await self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    async def get_daily_pnl(self) -> float:
        """Sum of net P&L for all closed trades today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cursor = await self._conn.execute(
            """SELECT COALESCE(SUM(pnl), 0) FROM trades
               WHERE status = 'closed' AND timestamp >= ?""",
            (today,),
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_consecutive_losses(self) -> int:
        """Count consecutive losing trades from the most recent."""
        cursor = await self._conn.execute(
            "SELECT pnl FROM trades WHERE status = 'closed' ORDER BY id DESC LIMIT 50"
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            # If P&L is negative, it's a loss
            if row[0] < 0:
                count += 1
            else:
                break  # first win breaks the streak
        return count

    async def get_daily_trade_count(self) -> int:
        """Count trades placed today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (today,)
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_strategy_stats(self, strategy: str, window: int = 50) -> StrategyStats:
        """Compute rolling performance stats for a strategy.

        Uses the most recent `window` trades for that strategy.
        """
        cursor = await self._conn.execute(
            """SELECT pnl, fees FROM trades
               WHERE strategy = ? AND status = 'closed'
               ORDER BY id DESC LIMIT ?""",
            (strategy, window),
        )
        rows = await cursor.fetchall()

        if not rows:
            return StrategyStats(
                strategy_name=strategy, total_trades=0,
                win_rate=0, avg_pnl=0, avg_win=0, avg_loss=0,
                total_fees=0, sharpe_ratio=0, max_drawdown=0,
            )

        pnls = [r[0] for r in rows]
        fees = [r[1] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(pnls)
        win_rate = len(wins) / total if total > 0 else 0
        avg_pnl = sum(pnls) / total
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        total_fees_val = sum(fees)

        # Sharpe ratio: mean / std of returns (annualized not needed for comparison)
        mean_pnl = avg_pnl
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / total) ** 0.5 if total > 1 else 1
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

        # Max drawdown from the P&L series
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in reversed(pnls):  # oldest first
            cumulative += p
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return StrategyStats(
            strategy_name=strategy,
            total_trades=total,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_fees=total_fees_val,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )

    async def log_decision(self, decision: AIDecision):
        """Log an AI decision for post-analysis."""
        await self._conn.execute(
            """INSERT INTO ai_decisions
               (timestamp, regime, gemini_confidence, ml_signal,
                ml_probability, action_taken)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                decision.timestamp.isoformat(),
                decision.regime, decision.gemini_confidence,
                decision.ml_signal, decision.ml_probability,
                decision.action_taken,
            ),
        )
        await self._conn.commit()

    async def snapshot_portfolio(self, snapshot: PortfolioSnapshot):
        """Save an hourly portfolio snapshot."""
        await self._conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, total_balance, unrealized_pnl, open_positions_count)
               VALUES (?, ?, ?, ?)""",
            (
                snapshot.timestamp.isoformat(),
                snapshot.total_balance, snapshot.unrealized_pnl,
                snapshot.open_positions_count,
            ),
        )
        await self._conn.commit()

    async def get_weekly_drawdown(self) -> float:
        """Max drawdown over the past 7 days from portfolio snapshots."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        cursor = await self._conn.execute(
            """SELECT total_balance FROM portfolio_snapshots
               WHERE timestamp >= ? ORDER BY timestamp ASC""",
            (week_ago,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return 0.0

        peak = rows[0][0]
        max_dd_pct = 0.0
        for row in rows:
            bal = row[0]
            peak = max(peak, bal)
            if peak > 0:
                dd_pct = ((peak - bal) / peak) * 100
                max_dd_pct = max(max_dd_pct, dd_pct)
        return max_dd_pct

    async def get_monthly_drawdown(self) -> float:
        """Max drawdown over the past 30 days from portfolio snapshots."""
        month_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        cursor = await self._conn.execute(
            """SELECT total_balance FROM portfolio_snapshots
               WHERE timestamp >= ? ORDER BY timestamp ASC""",
            (month_ago,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return 0.0

        peak = rows[0][0]
        max_dd_pct = 0.0
        for row in rows:
            bal = row[0]
            peak = max(peak, bal)
            if peak > 0:
                dd_pct = ((peak - bal) / peak) * 100
                max_dd_pct = max(max_dd_pct, dd_pct)
        return max_dd_pct

    def _row_to_trade(self, row) -> Trade:
        """Convert a database row to a Trade dataclass."""
        return Trade(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            pair=row[2], side=row[3], strategy=row[4],
            entry_price=row[5], exit_price=row[6], quantity=row[7],
            pnl=row[8], fees=row[9], status=row[10],
        )
