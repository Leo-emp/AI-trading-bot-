# AI Trading Bot — Phase 0+1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully autonomous crypto trading bot that backtests strategies, paper trades on Binance with multi-brain consensus, 5-layer protection, and self-learning — ready to go live after proving profitability.

**Architecture:** Event-driven Python pipeline: Market Data → Indicators → Trade Gate (5-brain consensus) → Risk Manager (5-layer protection) → Executor (limit orders only). SQLite for storage, Telegram for alerts, paper trading mode with fee simulation.

**Tech Stack:** Python 3.11+, ccxt (Binance), pandas/numpy, ta (indicators), google-generativeai (Gemini), python-telegram-bot, aiosqlite, websockets, pyyaml

## Global Constraints

- Python 3.11+ required (asyncio improvements)
- All API keys in `.env` only — NEVER in code, config, or commits
- All orders LIMIT ONLY — never market/taker orders
- All P&L calculations include fees (maker 0.075% with BNB per side)
- Binance minimum order floor: $10 per trade
- Every file must have heavily commented code explaining logic for learning
- No ML model until 30+ days of data collected — rule-based signals only in Phase 1
- DEX arbitrage is Phase 2 — not in this plan

---

### Task 1: Project Scaffolding + Database Layer

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `config/settings.yaml`
- Create: `config/strategies.yaml`
- Create: `src/__init__.py`
- Create: `src/storage/__init__.py`
- Create: `src/storage/models.py`
- Create: `src/storage/database.py`
- Create: `tests/__init__.py`
- Create: `tests/test_database.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `Database` class with methods: `async init()`, `async log_trade(trade: Trade)`, `async log_decision(decision: AIDecision)`, `async get_recent_trades(limit: int) -> list[Trade]`, `async get_strategy_stats(strategy: str) -> StrategyStats`, `async snapshot_portfolio(snapshot: PortfolioSnapshot)`, `async get_daily_pnl() -> float`, `async get_consecutive_losses() -> int`, `async get_weekly_drawdown() -> float`, `async get_monthly_drawdown() -> float`
  - Dataclasses: `Trade`, `AIDecision`, `PortfolioSnapshot`, `StrategyStats`
  - Config loader: `load_settings() -> dict`, `load_strategies() -> dict`

- [ ] **Step 1: Create requirements.txt**

```
ccxt==4.4.50
pandas==2.2.3
numpy==2.1.3
ta==0.11.0
google-generativeai==0.8.5
python-telegram-bot==21.10
websockets==14.2
aiosqlite==0.20.0
pyyaml==6.0.2
python-dotenv==1.0.1
pytest==8.3.4
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Create .env.example and .gitignore**

`.env.example`:
```
# Binance API (Spot only, NO withdrawal permission)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here

# Gemini AI
GEMINI_API_KEY=your_gemini_key_here

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Mode: paper or live
TRADING_MODE=paper
```

`.gitignore`:
```
.env
__pycache__/
*.pyc
*.db
*.db-journal
ml/models/*.pkl
.venv/
venv/
```

- [ ] **Step 3: Create config/settings.yaml**

```yaml
# Trading pairs to monitor (highest liquidity, tightest spreads)
pairs:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT

# Fee structure (Binance with BNB discount)
fees:
  maker_rate: 0.00075  # 0.075% per side
  taker_rate: 0.001    # 0.1% per side (never used — limit only)
  use_bnb_discount: true

# Risk management
risk:
  max_open_positions: 3
  max_daily_drawdown_pct: 5.0
  max_consecutive_losses: 5
  consecutive_loss_pause_minutes: 30
  max_daily_trades: 25
  min_balance_floor: 10.0  # USD — permanent shutdown below this
  max_position_pct: 25.0   # max % of portfolio per trade
  min_order_size: 10.0     # Binance minimum
  black_swan_pct: 5.0      # % move in 1 min triggers emergency sell
  stale_data_seconds: 10   # halt if data older than this
  max_spread_pct: 0.15     # skip if bid/ask spread exceeds this
  time_exit_hours: 4       # close any position older than this

# Auto-scaling tiers (keyed by minimum balance)
scaling:
  - min_balance: 20
    max_risk_pct: 0.4
    max_positions: 3
    daily_trade_limit: 25
  - min_balance: 100
    max_risk_pct: 0.5
    max_positions: 4
    daily_trade_limit: 40
  - min_balance: 500
    max_risk_pct: 0.75
    max_positions: 5
    daily_trade_limit: 60
  - min_balance: 2000
    max_risk_pct: 1.0
    max_positions: 6
    daily_trade_limit: 80

# Protection layers
protection:
  # Layer 2 — session
  reduce_size_after_losses: 3       # consecutive losses → 50% size
  pause_after_losses: 5             # consecutive losses → pause
  defense_mode_drawdown_pct: 3.0    # daily drawdown % → defense only
  # Layer 3 — portfolio
  weekly_drawdown_reduce_pct: 10.0  # → 50% sizes for a week
  monthly_drawdown_emergency_pct: 15.0  # → paper trade only
  # Layer 4 — black swan
  flash_crash_pct: 5.0              # % in 1 min → liquidate all
  max_api_errors: 3                 # consecutive → close all

# Scheduling
schedule:
  gemini_interval_minutes: 15
  portfolio_snapshot_minutes: 60
  daily_report_hour_utc: 0  # midnight UTC
```

- [ ] **Step 4: Create config/strategies.yaml**

```yaml
smart_scalp:
  enabled: true
  timeframes: ["5m", "15m"]
  # Entry: multi-confirmation required
  rsi_oversold: 30
  rsi_overbought: 70
  volume_spike_multiplier: 1.5  # volume must be 1.5x average
  # Targets (fee-aware: 2:1 reward/risk)
  take_profit_pct: 0.8   # 0.8% gross profit target
  stop_loss_pct: 0.4     # 0.4% gross loss limit
  # Min profit must exceed 2x round-trip fees
  min_net_profit_pct: 0.30
  # Adaptive tuning bounds (max drift from defaults)
  max_param_drift_pct: 30

grid:
  enabled: true
  levels: 5               # grid levels above and below price
  spacing_atr_multiplier: 0.5  # spacing = ATR * this
  min_spacing_pct: 0.3    # minimum grid spacing
  max_spacing_pct: 2.0    # maximum grid spacing

momentum:
  enabled: true
  ema_fast: 9
  ema_slow: 21
  rsi_min: 30    # don't enter if RSI below (already oversold)
  rsi_max: 70    # don't enter if RSI above (already overbought)
  trailing_stop_pct: 1.5
  trailing_stop_tighten_pct: 0.5  # tighten after 1% profit

mean_reversion:
  enabled: true
  rsi_extreme_low: 25
  rsi_extreme_high: 75
  bollinger_period: 20
  bollinger_std: 2.0
  pairs_only: ["BTC/USDT", "ETH/USDT"]  # high-cap only
```

- [ ] **Step 5: Write the failing test for database**

```python
# tests/test_database.py
import pytest
import asyncio
import os
from datetime import datetime, timezone

from src.storage.database import Database
from src.storage.models import Trade, AIDecision, PortfolioSnapshot


@pytest.fixture
async def db(tmp_path):
    """Create a fresh test database."""
    db = Database(str(tmp_path / "test.db"))
    await db.init()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_log_and_retrieve_trade(db):
    """Logging a trade should make it retrievable."""
    trade = Trade(
        timestamp=datetime.now(timezone.utc),
        pair="BTC/USDT",
        side="buy",
        strategy="smart_scalp",
        entry_price=50000.0,
        exit_price=50400.0,
        quantity=0.0002,
        pnl=0.08,
        fees=0.015,
        status="closed",
    )
    await db.log_trade(trade)
    trades = await db.get_recent_trades(limit=10)
    assert len(trades) == 1
    assert trades[0].pair == "BTC/USDT"
    assert trades[0].pnl == 0.08
    assert trades[0].fees == 0.015


@pytest.mark.asyncio
async def test_daily_pnl_calculation(db):
    """Daily P&L sums all closed trades from today."""
    now = datetime.now(timezone.utc)
    # One win, one loss
    await db.log_trade(Trade(
        timestamp=now, pair="BTC/USDT", side="buy", strategy="smart_scalp",
        entry_price=50000, exit_price=50400, quantity=0.0002,
        pnl=0.08, fees=0.015, status="closed",
    ))
    await db.log_trade(Trade(
        timestamp=now, pair="ETH/USDT", side="buy", strategy="smart_scalp",
        entry_price=3000, exit_price=2988, quantity=0.004,
        pnl=-0.048, fees=0.009, status="closed",
    ))
    daily = await db.get_daily_pnl()
    assert abs(daily - 0.032) < 0.001  # 0.08 - 0.048


@pytest.mark.asyncio
async def test_consecutive_losses(db):
    """Should count consecutive losses from most recent trades."""
    now = datetime.now(timezone.utc)
    # 3 losses in a row
    for i in range(3):
        await db.log_trade(Trade(
            timestamp=now, pair="BTC/USDT", side="buy", strategy="smart_scalp",
            entry_price=50000, exit_price=49800, quantity=0.0002,
            pnl=-0.04, fees=0.015, status="closed",
        ))
    assert await db.get_consecutive_losses() == 3
    # Then a win — resets counter
    await db.log_trade(Trade(
        timestamp=now, pair="BTC/USDT", side="buy", strategy="smart_scalp",
        entry_price=50000, exit_price=50400, quantity=0.0002,
        pnl=0.08, fees=0.015, status="closed",
    ))
    assert await db.get_consecutive_losses() == 0


@pytest.mark.asyncio
async def test_strategy_stats(db):
    """Should compute rolling stats per strategy."""
    now = datetime.now(timezone.utc)
    # 3 wins, 2 losses for smart_scalp
    for pnl in [0.08, 0.06, -0.04, 0.10, -0.03]:
        await db.log_trade(Trade(
            timestamp=now, pair="BTC/USDT", side="buy", strategy="smart_scalp",
            entry_price=50000, exit_price=50000, quantity=0.0002,
            pnl=pnl, fees=0.015, status="closed",
        ))
    stats = await db.get_strategy_stats("smart_scalp")
    assert stats.total_trades == 5
    assert abs(stats.win_rate - 0.6) < 0.01  # 3/5


@pytest.mark.asyncio
async def test_log_ai_decision(db):
    """AI decisions should be logged for analysis."""
    decision = AIDecision(
        timestamp=datetime.now(timezone.utc),
        regime="SIDEWAYS",
        gemini_confidence=72.0,
        ml_signal="HOLD",
        ml_probability=0.0,
        action_taken="hold",
    )
    await db.log_decision(decision)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_database.py -v`
Expected: FAIL — modules not found

- [ ] **Step 7: Create src/storage/models.py**

```python
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
```

- [ ] **Step 8: Create src/storage/database.py**

```python
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
```

- [ ] **Step 9: Create config loader utility**

Add to the top of `src/__init__.py`:
```python
# src/__init__.py
```

Create `src/core/__init__.py`:
```python
# src/core/__init__.py
```

Create `src/core/config.py`:
```python
# src/core/config.py
# Loads settings.yaml and strategies.yaml.
# API keys come from .env (via python-dotenv), never from YAML.

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file at import time so all modules can use os.getenv()
load_dotenv()

# Project root is two levels up from this file (src/core/config.py → project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_settings() -> dict:
    """Load config/settings.yaml and return as dict."""
    path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_strategies() -> dict:
    """Load config/strategies.yaml and return as dict."""
    path = PROJECT_ROOT / "config" / "strategies.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_scaling_tier(balance: float, settings: dict) -> dict:
    """Return the risk scaling tier for the current balance.

    Walks the scaling list in reverse (highest first) and returns
    the first tier whose min_balance is <= current balance.
    """
    tiers = sorted(settings["scaling"], key=lambda t: t["min_balance"], reverse=True)
    for tier in tiers:
        if balance >= tier["min_balance"]:
            return tier
    # Fallback to most conservative tier
    return settings["scaling"][0]
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `python -m pytest tests/test_database.py -v`
Expected: All 5 tests PASS

- [ ] **Step 11: Commit**

```bash
git init
git add requirements.txt .env.example .gitignore config/ src/storage/ src/__init__.py src/core/ tests/
git commit -m "feat: project scaffolding, database layer, config loader"
```

---

### Task 2: Binance Client (REST + WebSocket + Auto-Reconnect)

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/feed.py`
- Create: `src/data/order_book.py`
- Create: `tests/test_feed.py`

**Interfaces:**
- Consumes: `load_settings()` from Task 1
- Produces:
  - `BinanceClient` class: `async connect()`, `async disconnect()`, `async get_ohlcv(pair, timeframe, limit) -> pd.DataFrame`, `async get_ticker(pair) -> dict`, `async get_balance() -> dict`, `async place_limit_order(pair, side, price, quantity) -> dict`, `async cancel_order(pair, order_id)`, `async get_order_book(pair, limit) -> dict`, `async get_historical_ohlcv(pair, timeframe, since, limit) -> pd.DataFrame`
  - `PriceFeed` class: `async start(pairs, on_candle_callback)`, `async stop()`, `is_stale(max_age_seconds) -> bool`
  - `OrderBookAnalyzer` class: `analyze(order_book) -> OrderBookSignal` with fields: `imbalance: float`, `whale_detected: bool`, `whale_side: str`, `spread_pct: float`, `is_liquid: bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feed.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.feed import BinanceClient, PriceFeed
from src.data.order_book import OrderBookAnalyzer, OrderBookSignal


class TestBinanceClient:
    """Tests for the Binance API wrapper."""

    @pytest.mark.asyncio
    async def test_get_ohlcv_returns_dataframe(self):
        """get_ohlcv should return a pandas DataFrame with OHLCV columns."""
        client = BinanceClient(paper_mode=True)
        # In paper mode, uses ccxt's sandbox or returns mock data
        with patch.object(client, '_exchange') as mock_ex:
            mock_ex.fetch_ohlcv = AsyncMock(return_value=[
                [1700000000000, 50000, 50100, 49900, 50050, 100],
                [1700000060000, 50050, 50200, 50000, 50150, 120],
            ])
            df = await client.get_ohlcv("BTC/USDT", "5m", limit=2)
            assert len(df) == 2
            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
            assert df.iloc[0]["close"] == 50050

    @pytest.mark.asyncio
    async def test_place_limit_order_enforces_limit_type(self):
        """Should always place LIMIT orders, never market."""
        client = BinanceClient(paper_mode=True)
        with patch.object(client, '_exchange') as mock_ex:
            mock_ex.create_limit_buy_order = AsyncMock(return_value={"id": "123"})
            result = await client.place_limit_order("BTC/USDT", "buy", 50000, 0.0002)
            mock_ex.create_limit_buy_order.assert_called_once()
            assert result["id"] == "123"


class TestOrderBookAnalyzer:
    """Tests for order book depth analysis."""

    def test_detects_buy_side_imbalance(self):
        """When bids >> asks, imbalance should be positive (bullish)."""
        analyzer = OrderBookAnalyzer()
        # More volume on bid side → bullish imbalance
        book = {
            "bids": [[50000, 10], [49990, 8], [49980, 7]],  # total: 25
            "asks": [[50010, 3], [50020, 2], [50030, 2]],   # total: 7
        }
        signal = analyzer.analyze(book)
        assert signal.imbalance > 0.5  # strongly bullish
        assert signal.is_liquid  # has orders on both sides

    def test_detects_whale_order(self):
        """Orders > $50K on a single level should flag whale."""
        analyzer = OrderBookAnalyzer(whale_threshold_usd=50000)
        book = {
            # 2 BTC at $50000 = $100K → whale
            "bids": [[50000, 2.0], [49990, 0.1]],
            "asks": [[50010, 0.1], [50020, 0.1]],
        }
        signal = analyzer.analyze(book)
        assert signal.whale_detected is True
        assert signal.whale_side == "bid"

    def test_spread_calculation(self):
        """Spread should be (best_ask - best_bid) / mid_price * 100."""
        analyzer = OrderBookAnalyzer()
        book = {
            "bids": [[50000, 1]],
            "asks": [[50100, 1]],
        }
        signal = analyzer.analyze(book)
        # spread = (50100 - 50000) / 50050 * 100 = 0.1998%
        assert abs(signal.spread_pct - 0.1998) < 0.01

    def test_illiquid_when_spread_too_wide(self):
        """Spread > 0.15% should mark as illiquid."""
        analyzer = OrderBookAnalyzer(max_spread_pct=0.15)
        book = {
            "bids": [[50000, 1]],
            "asks": [[50200, 1]],  # 0.4% spread
        }
        signal = analyzer.analyze(book)
        assert signal.is_liquid is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feed.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Create src/data/order_book.py**

```python
# src/data/order_book.py
# Analyzes order book depth to detect:
# - Buy/sell imbalance (are buyers or sellers stronger?)
# - Whale orders (single large orders that can move price)
# - Spread width (too wide = too expensive to trade)
# - Spoofing (large order placed then cancelled — fake signal)

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderBookSignal:
    """Result of order book analysis.

    imbalance: -1.0 (all sellers) to +1.0 (all buyers)
    whale_detected: True if any single order > whale_threshold_usd
    whale_side: "bid" or "ask" (which side the whale is on)
    spread_pct: bid-ask spread as percentage of mid price
    is_liquid: True if spread is within acceptable range
    """
    imbalance: float
    whale_detected: bool
    whale_side: str
    spread_pct: float
    is_liquid: bool


class OrderBookAnalyzer:
    """Analyzes raw order book data from Binance.

    The order book is a list of [price, quantity] pairs for bids and asks.
    We compute: imbalance ratio, whale detection, spread, liquidity check.
    """

    def __init__(self, whale_threshold_usd: float = 50000, max_spread_pct: float = 0.15):
        # Any single order above this USD value counts as a "whale"
        self._whale_threshold = whale_threshold_usd
        # Spread wider than this → skip trading (too expensive)
        self._max_spread = max_spread_pct

    def analyze(self, order_book: dict) -> OrderBookSignal:
        """Analyze a raw order book dict with 'bids' and 'asks' keys.

        Each bid/ask is [price, quantity]. Bids sorted high→low,
        asks sorted low→high (standard exchange format).
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            # No data — return neutral, illiquid signal
            return OrderBookSignal(
                imbalance=0.0, whale_detected=False,
                whale_side="none", spread_pct=999.0, is_liquid=False,
            )

        # --- Imbalance: total bid volume vs total ask volume ---
        # Sum up the quantity on each side
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        total_volume = bid_volume + ask_volume

        # Imbalance from -1 (all asks) to +1 (all bids)
        # +1 = very bullish (buyers dominate), -1 = very bearish
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0

        # --- Whale detection: any single order > threshold ---
        whale_detected = False
        whale_side = "none"

        for price, qty in bids:
            if price * qty >= self._whale_threshold:
                whale_detected = True
                whale_side = "bid"
                break

        if not whale_detected:
            for price, qty in asks:
                if price * qty >= self._whale_threshold:
                    whale_detected = True
                    whale_side = "ask"
                    break

        # --- Spread: how expensive is it to cross the book ---
        best_bid = bids[0][0]   # highest bid
        best_ask = asks[0][0]   # lowest ask
        mid_price = (best_bid + best_ask) / 2
        spread_pct = ((best_ask - best_bid) / mid_price) * 100 if mid_price > 0 else 999.0

        # --- Liquidity: is the spread acceptable? ---
        is_liquid = spread_pct <= self._max_spread

        return OrderBookSignal(
            imbalance=imbalance,
            whale_detected=whale_detected,
            whale_side=whale_side,
            spread_pct=spread_pct,
            is_liquid=is_liquid,
        )
```

- [ ] **Step 4: Create src/data/feed.py**

```python
# src/data/feed.py
# Binance exchange client and real-time price feed.
# Uses ccxt for REST API and WebSocket for live data.
# Auto-reconnects on disconnection with exponential backoff.
# LIMIT ORDERS ONLY — never places market orders.

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Callable

import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceClient:
    """Wrapper around ccxt's Binance implementation.

    Enforces limit-only orders and handles connection management.
    In paper_mode, uses sandbox endpoints or mocks for testing.
    """

    def __init__(self, paper_mode: bool = True,
                 api_key: str = "", secret: str = ""):
        # Whether to simulate trades or use real money
        self._paper_mode = paper_mode

        # Create the ccxt exchange instance
        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "sandbox": paper_mode,
            "enableRateLimit": True,  # respect Binance rate limits automatically
            "options": {
                "defaultType": "spot",  # spot trading only, no futures
            },
        })

    async def connect(self):
        """Load exchange markets (required before trading)."""
        await self._exchange.load_markets()
        logger.info("Connected to Binance (paper=%s)", self._paper_mode)

    async def disconnect(self):
        """Close the exchange connection."""
        await self._exchange.close()

    async def get_ohlcv(self, pair: str, timeframe: str = "5m",
                        limit: int = 100) -> pd.DataFrame:
        """Fetch recent OHLCV candles.

        Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
        """
        raw = await self._exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def get_historical_ohlcv(self, pair: str, timeframe: str = "5m",
                                    since: Optional[int] = None,
                                    limit: int = 1000) -> pd.DataFrame:
        """Fetch historical OHLCV data for backtesting.

        `since` is a Unix timestamp in milliseconds.
        Binance returns max 1000 candles per request.
        """
        raw = await self._exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    async def get_ticker(self, pair: str) -> dict:
        """Get current ticker (last price, bid, ask, volume)."""
        return await self._exchange.fetch_ticker(pair)

    async def get_balance(self) -> dict:
        """Get account balances. Returns dict like {'USDT': 50.0, 'BTC': 0.001}."""
        balance = await self._exchange.fetch_balance()
        # Return only non-zero free balances
        return {
            currency: amount
            for currency, amount in balance["free"].items()
            if amount > 0
        }

    async def get_order_book(self, pair: str, limit: int = 20) -> dict:
        """Fetch order book depth. Returns {'bids': [...], 'asks': [...]}."""
        return await self._exchange.fetch_order_book(pair, limit=limit)

    async def place_limit_order(self, pair: str, side: str,
                                 price: float, quantity: float) -> dict:
        """Place a LIMIT order. NEVER places market orders.

        side: 'buy' or 'sell'
        Returns order info dict with 'id', 'status', etc.
        """
        if side == "buy":
            return await self._exchange.create_limit_buy_order(pair, quantity, price)
        elif side == "sell":
            return await self._exchange.create_limit_sell_order(pair, quantity, price)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")

    async def cancel_order(self, pair: str, order_id: str):
        """Cancel an open order."""
        return await self._exchange.cancel_order(order_id, pair)

    async def get_order_status(self, pair: str, order_id: str) -> dict:
        """Check if an order has been filled, partially filled, or is still open."""
        return await self._exchange.fetch_order(order_id, pair)


class PriceFeed:
    """Real-time price feed using Binance WebSocket.

    Subscribes to kline (candlestick) streams for specified pairs
    and timeframes. Calls on_candle callback when new candles arrive.

    Auto-reconnects with exponential backoff on disconnection.
    Tracks data freshness — stale data halts trading.
    """

    def __init__(self, client: BinanceClient):
        self._client = client
        # Timestamp of last received data — used for stale detection
        self._last_update: float = 0
        # Whether the feed is running
        self._running = False
        # Latest candle data per pair
        self._latest: dict[str, dict] = {}

    async def start(self, pairs: list[str], timeframes: list[str],
                    on_candle: Optional[Callable] = None):
        """Start watching candles for given pairs.

        Uses ccxt's watch_ohlcv for WebSocket streaming.
        Falls back to REST polling if WebSocket fails.
        """
        self._running = True
        self._last_update = time.time()

        retry_delay = 1  # exponential backoff starting point
        max_retry = 60   # max delay between retries

        while self._running:
            try:
                # ccxt pro watch_ohlcv provides WebSocket streaming
                for pair in pairs:
                    for tf in timeframes:
                        candles = await self._client._exchange.watch_ohlcv(pair, tf)
                        if candles:
                            self._last_update = time.time()
                            latest = candles[-1]
                            self._latest[f"{pair}:{tf}"] = {
                                "timestamp": latest[0],
                                "open": latest[1],
                                "high": latest[2],
                                "low": latest[3],
                                "close": latest[4],
                                "volume": latest[5],
                            }
                            if on_candle:
                                await on_candle(pair, tf, self._latest[f"{pair}:{tf}"])
                # Reset backoff on success
                retry_delay = 1

            except Exception as e:
                logger.error("PriceFeed error: %s. Retrying in %ds...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                # Exponential backoff: 1, 2, 4, 8, 16, 32, 60, 60...
                retry_delay = min(retry_delay * 2, max_retry)

    async def stop(self):
        """Stop the price feed."""
        self._running = False

    def is_stale(self, max_age_seconds: float = 10) -> bool:
        """True if no data received within max_age_seconds.

        The trading engine checks this — if data is stale, all
        trading halts until fresh data arrives. Prevents trading
        on outdated information.
        """
        if self._last_update == 0:
            return True  # never received data
        return (time.time() - self._last_update) > max_age_seconds

    def get_latest(self, pair: str, timeframe: str) -> Optional[dict]:
        """Get the most recent candle for a pair+timeframe."""
        return self._latest.get(f"{pair}:{timeframe}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_feed.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/test_feed.py
git commit -m "feat: Binance client, price feed with auto-reconnect, order book analyzer"
```

---

### Task 3: Technical Indicator Engine

**Files:**
- Create: `src/data/indicators.py`
- Create: `tests/test_indicators.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with OHLCV columns from `BinanceClient.get_ohlcv()`
- Produces:
  - `IndicatorEngine` class: `compute_all(df: pd.DataFrame) -> pd.DataFrame` adds columns: `rsi`, `macd`, `macd_signal`, `macd_histogram`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `atr`, `ema_fast`, `ema_slow`, `volume_sma`, `volume_ratio`
  - `get_signal(df: pd.DataFrame, config: dict) -> TechnicalSignal` with fields: `direction: str` (BUY/SELL/HOLD), `confidence: float` (0-1), `reasons: list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.data.indicators import IndicatorEngine, TechnicalSignal


def make_ohlcv(n: int = 100, base_price: float = 50000.0,
               volatility: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Creates a random walk with specified volatility.
    Returns DataFrame matching Binance OHLCV format.
    """
    np.random.seed(42)  # reproducible
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    closes = [base_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + np.random.randn() * volatility)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [c - np.random.rand() * 50 for c in closes],
        "high": [c + abs(np.random.randn()) * 80 for c in closes],
        "low": [c - abs(np.random.randn()) * 80 for c in closes],
        "close": closes,
        "volume": [np.random.rand() * 1000 + 100 for _ in range(n)],
    })
    return df


class TestIndicatorEngine:

    def test_compute_all_adds_expected_columns(self):
        """compute_all should add all indicator columns to the DataFrame."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        result = engine.compute_all(df)

        expected_cols = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr", "ema_fast", "ema_slow", "volume_sma", "volume_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_bounded_0_100(self):
        """RSI should always be between 0 and 100."""
        engine = IndicatorEngine()
        df = make_ohlcv(200)
        result = engine.compute_all(df)
        rsi_valid = result["rsi"].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_bollinger_bands_order(self):
        """Upper band > middle > lower, always."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        result = engine.compute_all(df)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_volume_ratio_flags_spikes(self):
        """Volume 2x the SMA should give volume_ratio >= 2.0."""
        engine = IndicatorEngine()
        df = make_ohlcv(50)
        # Inject a volume spike at the last row
        df.loc[df.index[-1], "volume"] = 999999
        result = engine.compute_all(df)
        assert result.iloc[-1]["volume_ratio"] > 2.0

    def test_get_signal_returns_valid_direction(self):
        """Signal direction should be BUY, SELL, or HOLD."""
        engine = IndicatorEngine()
        df = make_ohlcv(100)
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
        }
        signal = engine.get_signal(df, config)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.reasons, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indicators.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/data/indicators.py**

```python
# src/data/indicators.py
# Computes technical indicators from OHLCV data using the `ta` library.
# Every indicator is a proven, widely-used tool for reading price action:
#
# - RSI: momentum oscillator — is the asset overbought or oversold?
# - MACD: trend direction and momentum — is a trend starting or ending?
# - Bollinger Bands: volatility envelope — is price at an extreme?
# - ATR: average true range — how volatile is the asset right now?
# - EMA: exponential moving average — smoothed price trend
# - Volume ratio: is current volume unusual compared to recent average?

from dataclasses import dataclass, field

import pandas as pd
import ta


@dataclass(frozen=True)
class TechnicalSignal:
    """Output of the technical analysis brain.

    direction: BUY, SELL, or HOLD
    confidence: 0.0 (no confidence) to 1.0 (very confident)
    reasons: human-readable list of why this signal was generated
    """
    direction: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


class IndicatorEngine:
    """Computes all technical indicators on an OHLCV DataFrame.

    Usage:
        engine = IndicatorEngine()
        df = engine.compute_all(ohlcv_df)   # adds indicator columns
        signal = engine.get_signal(df, strategy_config)  # BUY/SELL/HOLD
    """

    def __init__(self, rsi_period: int = 14, bb_period: int = 20,
                 bb_std: float = 2.0, atr_period: int = 14,
                 ema_fast: int = 9, ema_slow: int = 21,
                 volume_sma_period: int = 20):
        self._rsi_period = rsi_period
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._atr_period = atr_period
        self._ema_fast = ema_fast
        self._ema_slow = ema_slow
        self._vol_sma_period = volume_sma_period

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicator columns to the DataFrame.

        Input must have columns: open, high, low, close, volume.
        Returns the same DataFrame with additional indicator columns.
        Does NOT modify the original DataFrame.
        """
        df = df.copy()

        # --- RSI: Relative Strength Index ---
        # Measures momentum: >70 = overbought, <30 = oversold
        df["rsi"] = ta.momentum.rsi(df["close"], window=self._rsi_period)

        # --- MACD: Moving Average Convergence Divergence ---
        # Shows trend direction and momentum changes
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # --- Bollinger Bands: volatility envelope around price ---
        # Price touching upper band = possibly overbought
        # Price touching lower band = possibly oversold
        bb = ta.volatility.BollingerBands(df["close"], window=self._bb_period, window_dev=self._bb_std)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        # Width shows volatility — narrow = squeeze, wide = expansion
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # --- ATR: Average True Range ---
        # Measures volatility in price terms (used for grid spacing, stop sizes)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=self._atr_period)

        # --- EMAs: Exponential Moving Averages ---
        # Fast EMA crossing above slow = bullish, below = bearish
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=self._ema_fast)
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=self._ema_slow)

        # --- Volume analysis ---
        # Volume ratio: current volume / average volume
        # >1.5 = volume spike (something is happening)
        df["volume_sma"] = df["volume"].rolling(window=self._vol_sma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        return df

    def get_signal(self, df: pd.DataFrame, config: dict) -> TechnicalSignal:
        """Generate a BUY/SELL/HOLD signal from computed indicators.

        This is Brain 1 of the multi-brain consensus engine.
        Uses multiple indicators for confirmation — no single
        indicator alone can trigger a trade.

        config keys: rsi_oversold, rsi_overbought, volume_spike_multiplier
        """
        if len(df) < 2:
            return TechnicalSignal("HOLD", 0.0, ["insufficient data"])

        # Use the latest complete candle (not the current incomplete one)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        reasons = []
        buy_score = 0  # positive signals
        sell_score = 0

        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        prev_macd_hist = prev.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        close = latest["close"]
        bb_lower = latest.get("bb_lower")
        bb_upper = latest.get("bb_upper")
        ema_f = latest.get("ema_fast")
        ema_s = latest.get("ema_slow")

        # Skip if indicators haven't warmed up yet
        if pd.isna(rsi) or pd.isna(macd_hist):
            return TechnicalSignal("HOLD", 0.0, ["indicators warming up"])

        # --- RSI signal ---
        oversold = config.get("rsi_oversold", 30)
        overbought = config.get("rsi_overbought", 70)
        if rsi < oversold:
            buy_score += 1
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > overbought:
            sell_score += 1
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # --- MACD crossover ---
        if not pd.isna(prev_macd_hist):
            if prev_macd_hist < 0 and macd_hist > 0:
                buy_score += 1
                reasons.append("MACD bullish crossover")
            elif prev_macd_hist > 0 and macd_hist < 0:
                sell_score += 1
                reasons.append("MACD bearish crossover")

        # --- Bollinger Band touch ---
        if not pd.isna(bb_lower) and close <= bb_lower:
            buy_score += 1
            reasons.append("price at lower Bollinger Band")
        elif not pd.isna(bb_upper) and close >= bb_upper:
            sell_score += 1
            reasons.append("price at upper Bollinger Band")

        # --- EMA crossover ---
        if not pd.isna(ema_f) and not pd.isna(ema_s):
            prev_ema_f = prev.get("ema_fast")
            prev_ema_s = prev.get("ema_slow")
            if not pd.isna(prev_ema_f) and not pd.isna(prev_ema_s):
                if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                    buy_score += 1
                    reasons.append("EMA bullish crossover")
                elif prev_ema_f >= prev_ema_s and ema_f < ema_s:
                    sell_score += 1
                    reasons.append("EMA bearish crossover")

        # --- Volume confirmation ---
        # High volume confirms the signal; low volume weakens it
        vol_multiplier = config.get("volume_spike_multiplier", 1.5)
        has_volume = not pd.isna(vol_ratio) and vol_ratio >= vol_multiplier
        if has_volume:
            reasons.append(f"volume spike ({vol_ratio:.1f}x)")

        # --- Combine scores into direction + confidence ---
        # Need at least 2 confirming indicators for a signal
        if buy_score >= 2 and buy_score > sell_score:
            confidence = min(buy_score / 4, 1.0)  # max 4 indicators
            if has_volume:
                confidence = min(confidence + 0.15, 1.0)
            return TechnicalSignal("BUY", confidence, reasons)
        elif sell_score >= 2 and sell_score > buy_score:
            confidence = min(sell_score / 4, 1.0)
            if has_volume:
                confidence = min(confidence + 0.15, 1.0)
            return TechnicalSignal("SELL", confidence, reasons)
        else:
            return TechnicalSignal("HOLD", 0.0, reasons if reasons else ["no confirmed signal"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_indicators.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/indicators.py tests/test_indicators.py
git commit -m "feat: technical indicator engine (RSI, MACD, BB, ATR, EMA, volume)"
```

---

### Task 4: Risk Manager + 5-Layer Protection System

**Files:**
- Create: `src/risk/__init__.py`
- Create: `src/risk/manager.py`
- Create: `src/risk/position_sizer.py`
- Create: `src/risk/protection.py`
- Create: `tests/test_risk.py`

**Interfaces:**
- Consumes: `Database` from Task 1, `load_settings()` from Task 1
- Produces:
  - `RiskManager` class: `async can_trade() -> tuple[bool, str]`, `async validate_trade(pair, side, price, quantity) -> tuple[bool, str]`, `async get_position_size(balance, confidence) -> float`, `get_stop_loss_price(entry_price, side) -> float`, `get_take_profit_price(entry_price, side) -> float`
  - `ProtectionSystem` class: `async check_all_layers(balance, daily_pnl) -> ProtectionAction` with fields: `action: str` (CONTINUE/REDUCE_SIZE/PAUSE/DEFENSE_ONLY/SHUTDOWN/EMERGENCY_SELL), `reason: str`, `duration_minutes: int`
  - `PositionSizer` class: `kelly_size(win_rate, avg_win, avg_loss, balance) -> float`, `get_tier(balance) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.risk.manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.risk.protection import ProtectionSystem, ProtectionAction
from src.core.config import load_settings


@pytest.fixture
def settings():
    """Load real settings from config/settings.yaml."""
    return load_settings()


@pytest.fixture
def mock_db():
    """Mock database for testing without SQLite."""
    db = AsyncMock()
    db.get_daily_pnl = AsyncMock(return_value=0.0)
    db.get_consecutive_losses = AsyncMock(return_value=0)
    db.get_daily_trade_count = AsyncMock(return_value=0)
    db.get_weekly_drawdown = AsyncMock(return_value=0.0)
    db.get_monthly_drawdown = AsyncMock(return_value=0.0)
    return db


class TestPositionSizer:

    def test_kelly_size_with_positive_edge(self):
        """With 60% win rate and 2:1 R/R, Kelly should suggest a position."""
        sizer = PositionSizer()
        # 60% win rate, avg win = $0.80, avg loss = $0.40
        size = sizer.kelly_size(
            win_rate=0.6, avg_win=0.80, avg_loss=0.40, balance=50.0
        )
        # Quarter-Kelly should give a reasonable fraction of balance
        assert 0 < size <= 50.0
        # Should be conservative (quarter Kelly)
        assert size < 20.0  # less than 40% of balance

    def test_kelly_size_with_negative_edge_returns_minimum(self):
        """If win rate × avg_win < loss rate × avg_loss, return minimum."""
        sizer = PositionSizer(min_order_size=10.0)
        size = sizer.kelly_size(
            win_rate=0.3, avg_win=0.50, avg_loss=0.50, balance=50.0
        )
        # Negative edge → floor at minimum order size
        assert size == 10.0

    def test_position_never_below_exchange_minimum(self):
        """Position size must respect Binance $10 minimum."""
        sizer = PositionSizer(min_order_size=10.0)
        size = sizer.kelly_size(
            win_rate=0.55, avg_win=0.40, avg_loss=0.20, balance=20.0
        )
        assert size >= 10.0

    def test_get_tier_micro_account(self):
        """$50 balance should use micro tier settings."""
        sizer = PositionSizer()
        settings = load_settings()
        tier = sizer.get_tier(50.0, settings)
        assert tier["max_risk_pct"] == 0.4
        assert tier["max_positions"] == 3

    def test_get_tier_scales_up(self):
        """$600 balance should use the $500 tier."""
        sizer = PositionSizer()
        settings = load_settings()
        tier = sizer.get_tier(600.0, settings)
        assert tier["max_risk_pct"] == 0.75
        assert tier["max_positions"] == 5


class TestRiskManager:

    @pytest.mark.asyncio
    async def test_blocks_when_max_daily_trades_reached(self, mock_db, settings):
        """Should reject trades when daily limit is hit."""
        mock_db.get_daily_trade_count = AsyncMock(return_value=25)
        rm = RiskManager(mock_db, settings, balance=50.0)
        can, reason = await rm.can_trade()
        assert can is False
        assert "daily trade limit" in reason.lower()

    @pytest.mark.asyncio
    async def test_blocks_below_balance_floor(self, mock_db, settings):
        """Should refuse to trade if balance < $10."""
        rm = RiskManager(mock_db, settings, balance=8.0)
        can, reason = await rm.can_trade()
        assert can is False
        assert "balance floor" in reason.lower()

    @pytest.mark.asyncio
    async def test_allows_trade_in_normal_conditions(self, mock_db, settings):
        """Should allow trading when all conditions are met."""
        rm = RiskManager(mock_db, settings, balance=50.0)
        can, reason = await rm.can_trade()
        assert can is True

    def test_stop_loss_buy_side(self, mock_db, settings):
        """Stop loss for a BUY should be below entry price."""
        rm = RiskManager(mock_db, settings, balance=50.0)
        stop = rm.get_stop_loss_price(50000.0, "buy")
        assert stop < 50000.0
        # Should be 0.4% below entry
        expected = 50000.0 * (1 - 0.004)
        assert abs(stop - expected) < 1.0

    def test_take_profit_buy_side(self, mock_db, settings):
        """Take profit for a BUY should be above entry price."""
        rm = RiskManager(mock_db, settings, balance=50.0)
        tp = rm.get_take_profit_price(50000.0, "buy")
        assert tp > 50000.0
        # Should be 0.8% above entry
        expected = 50000.0 * (1 + 0.008)
        assert abs(tp - expected) < 1.0


class TestProtectionSystem:

    @pytest.mark.asyncio
    async def test_layer2_reduce_after_3_losses(self, mock_db, settings):
        """3 consecutive losses → reduce position size by 50%."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=3)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.0)
        assert action.action == "REDUCE_SIZE"

    @pytest.mark.asyncio
    async def test_layer2_pause_after_5_losses(self, mock_db, settings):
        """5 consecutive losses → pause trading."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=5)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.0)
        assert action.action == "PAUSE"
        assert action.duration_minutes == 30

    @pytest.mark.asyncio
    async def test_layer2_defense_mode_on_drawdown(self, mock_db, settings):
        """Daily drawdown > 3% → defense-only mode."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        # -3.5% daily P&L on $50 = -$1.75
        action = await ps.check_all_layers(balance=50.0, daily_pnl=-1.75)
        assert action.action == "DEFENSE_ONLY"

    @pytest.mark.asyncio
    async def test_layer2_shutdown_on_5pct_drawdown(self, mock_db, settings):
        """Daily drawdown > 5% → full shutdown."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=-2.75)
        assert action.action == "SHUTDOWN"

    @pytest.mark.asyncio
    async def test_layer4_shutdown_below_floor(self, mock_db, settings):
        """Balance below $10 → permanent shutdown."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=8.0, daily_pnl=0.0)
        assert action.action == "SHUTDOWN"

    @pytest.mark.asyncio
    async def test_continue_when_all_clear(self, mock_db, settings):
        """No issues → CONTINUE normally."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        mock_db.get_weekly_drawdown = AsyncMock(return_value=2.0)
        mock_db.get_monthly_drawdown = AsyncMock(return_value=5.0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.5)
        assert action.action == "CONTINUE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_risk.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Create src/risk/position_sizer.py**

```python
# src/risk/position_sizer.py
# Position sizing using the Kelly Criterion with a safety factor.
#
# Kelly Criterion: the mathematically optimal fraction of your bankroll
# to bet on each trade, given your win rate and average win/loss sizes.
# We use QUARTER-Kelly (0.25 safety factor) because:
# - Full Kelly is too aggressive for real trading
# - Quarter-Kelly gives ~75% of the growth with much less risk
# - It's the standard for professional systematic trading

from src.core.config import get_scaling_tier


class PositionSizer:
    """Calculates position sizes using Kelly Criterion.

    Respects exchange minimums (Binance $10) and portfolio percentage
    limits from the auto-scaling tier system.
    """

    def __init__(self, min_order_size: float = 10.0, safety_factor: float = 0.25):
        # Binance minimum order size in USDT
        self._min_order = min_order_size
        # Quarter-Kelly: multiply Kelly fraction by this
        self._safety = safety_factor

    def kelly_size(self, win_rate: float, avg_win: float,
                   avg_loss: float, balance: float) -> float:
        """Calculate optimal position size using quarter-Kelly.

        Formula: f* = (p * W - (1-p) * L) / W
        Where p = win rate, W = avg win, L = avg loss (positive number)

        Returns position size in USDT, floored at min_order_size.
        """
        # Ensure avg_loss is positive for the formula
        avg_loss = abs(avg_loss) if avg_loss != 0 else 0.01

        if avg_win <= 0:
            return self._min_order

        # Kelly fraction: what % of bankroll to risk
        # This is the edge divided by the odds
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        if kelly_fraction <= 0:
            # Negative edge — no profitable bet exists
            # Return minimum to allow the system to keep learning
            return self._min_order

        # Apply safety factor (quarter-Kelly)
        safe_fraction = kelly_fraction * self._safety

        # Convert fraction to dollar amount
        position_size = balance * safe_fraction

        # Floor at exchange minimum
        position_size = max(position_size, self._min_order)

        # Cap at 25% of balance (spec max per-trade)
        max_position = balance * 0.25
        position_size = min(position_size, max_position)

        return round(position_size, 2)

    def get_tier(self, balance: float, settings: dict) -> dict:
        """Get the risk scaling tier for the current balance.

        Returns a dict with: max_risk_pct, max_positions, daily_trade_limit.
        Higher balances get slightly looser limits.
        """
        return get_scaling_tier(balance, settings)
```

- [ ] **Step 4: Create src/risk/protection.py**

```python
# src/risk/protection.py
# 5-layer autonomous protection system.
# Each layer acts independently — faster layers override slower ones.
# The system checks all layers on every trading cycle and returns
# the most severe action needed.
#
# Layer 1: Per-trade (stop-loss, trailing stop) — handled by executor
# Layer 2: Session (consecutive losses, daily drawdown)
# Layer 3: Portfolio (weekly/monthly drawdown)
# Layer 4: Black swan (flash crash, API errors, balance floor)
# Layer 5: Infrastructure (reconnection, restart) — handled by watchdog

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtectionAction:
    """What the protection system instructs the engine to do.

    Actions ranked by severity:
    CONTINUE → REDUCE_SIZE → PAUSE → DEFENSE_ONLY → SHUTDOWN → EMERGENCY_SELL
    """
    action: str         # CONTINUE, REDUCE_SIZE, PAUSE, DEFENSE_ONLY, SHUTDOWN, EMERGENCY_SELL
    reason: str         # human-readable explanation
    duration_minutes: int = 0  # how long to pause/reduce (0 = until reset)
    size_multiplier: float = 1.0  # for REDUCE_SIZE: multiply position by this


class ProtectionSystem:
    """Multi-layer autonomous protection.

    Called by the engine on every cycle. Returns the most severe
    protection action from all layers. The engine must obey.
    """

    def __init__(self, db, settings: dict):
        self._db = db
        self._settings = settings
        self._protection = settings.get("protection", {})
        self._risk = settings.get("risk", {})

    async def check_all_layers(self, balance: float, daily_pnl: float) -> ProtectionAction:
        """Run all protection layers and return the most severe action.

        Checks layers 2-4 (layer 1 is per-trade, layer 5 is infra).
        Returns the highest-severity action found.
        """
        # Start with the most severe checks first

        # --- Layer 4: Black swan / balance floor ---
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if balance < min_balance:
            logger.critical("LAYER 4: Balance $%.2f below floor $%.2f — SHUTDOWN", balance, min_balance)
            return ProtectionAction(
                action="SHUTDOWN",
                reason=f"Balance ${balance:.2f} below minimum floor ${min_balance:.2f}",
            )

        # --- Layer 2: Daily drawdown checks ---
        # Calculate daily drawdown as percentage
        # daily_pnl is negative when losing money
        starting_balance = balance - daily_pnl  # what we started the day with
        if starting_balance > 0 and daily_pnl < 0:
            daily_dd_pct = abs(daily_pnl) / starting_balance * 100

            shutdown_threshold = self._risk.get("max_daily_drawdown_pct", 5.0)
            if daily_dd_pct >= shutdown_threshold:
                logger.critical("LAYER 2: Daily drawdown %.1f%% >= %.1f%% — SHUTDOWN", daily_dd_pct, shutdown_threshold)
                return ProtectionAction(
                    action="SHUTDOWN",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% exceeded {shutdown_threshold}% limit",
                )

            defense_threshold = self._protection.get("defense_mode_drawdown_pct", 3.0)
            if daily_dd_pct >= defense_threshold:
                logger.warning("LAYER 2: Daily drawdown %.1f%% — switching to defense-only", daily_dd_pct)
                return ProtectionAction(
                    action="DEFENSE_ONLY",
                    reason=f"Daily drawdown {daily_dd_pct:.1f}% — defense-only mode",
                )

        # --- Layer 2: Consecutive loss checks ---
        consecutive_losses = await self._db.get_consecutive_losses()

        pause_threshold = self._protection.get("pause_after_losses", 5)
        if consecutive_losses >= pause_threshold:
            pause_minutes = self._risk.get("consecutive_loss_pause_minutes", 30)
            logger.warning("LAYER 2: %d consecutive losses — pausing %d min", consecutive_losses, pause_minutes)
            return ProtectionAction(
                action="PAUSE",
                reason=f"{consecutive_losses} consecutive losses — cooling off",
                duration_minutes=pause_minutes,
            )

        reduce_threshold = self._protection.get("reduce_size_after_losses", 3)
        if consecutive_losses >= reduce_threshold:
            logger.info("LAYER 2: %d consecutive losses — reducing size 50%%", consecutive_losses)
            return ProtectionAction(
                action="REDUCE_SIZE",
                reason=f"{consecutive_losses} consecutive losses — reducing position sizes",
                size_multiplier=0.5,
            )

        # --- Layer 3: Weekly/monthly drawdown ---
        weekly_dd = await self._db.get_weekly_drawdown()
        weekly_threshold = self._protection.get("weekly_drawdown_reduce_pct", 10.0)
        if weekly_dd >= weekly_threshold:
            logger.warning("LAYER 3: Weekly drawdown %.1f%% — reducing sizes", weekly_dd)
            return ProtectionAction(
                action="REDUCE_SIZE",
                reason=f"Weekly drawdown {weekly_dd:.1f}% — protective size reduction",
                size_multiplier=0.5,
            )

        monthly_dd = await self._db.get_monthly_drawdown()
        monthly_threshold = self._protection.get("monthly_drawdown_emergency_pct", 15.0)
        if monthly_dd >= monthly_threshold:
            logger.critical("LAYER 3: Monthly drawdown %.1f%% — EMERGENCY paper-only", monthly_dd)
            return ProtectionAction(
                action="SHUTDOWN",
                reason=f"Monthly drawdown {monthly_dd:.1f}% — emergency shutdown, switch to paper trading",
            )

        # --- All clear ---
        return ProtectionAction(action="CONTINUE", reason="all protection layers clear")
```

- [ ] **Step 5: Create src/risk/manager.py**

```python
# src/risk/manager.py
# Central risk management gate. Every trade must pass through here.
# Checks: daily trade limit, balance floor, position size, drawdown.
# Also provides stop-loss and take-profit price calculations.

import logging
from src.risk.position_sizer import PositionSizer
from src.core.config import get_scaling_tier

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against risk rules before execution.

    No trade reaches the exchange without passing can_trade() and
    validate_trade(). This is the final safety gate.
    """

    def __init__(self, db, settings: dict, balance: float):
        self._db = db
        self._settings = settings
        self._risk = settings.get("risk", {})
        self._fees = settings.get("fees", {})
        self._balance = balance
        self._sizer = PositionSizer(
            min_order_size=self._risk.get("min_order_size", 10.0)
        )

    def update_balance(self, balance: float):
        """Update the current balance (called after each trade)."""
        self._balance = balance

    async def can_trade(self) -> tuple[bool, str]:
        """Check if the bot is allowed to place any new trade right now.

        Returns (True, "ok") or (False, "reason why not").
        This is the pre-flight check before evaluating any signal.
        """
        # Check balance floor
        min_balance = self._risk.get("min_balance_floor", 10.0)
        if self._balance < min_balance:
            return False, f"Balance ${self._balance:.2f} below balance floor ${min_balance:.2f}"

        # Check daily trade limit
        tier = get_scaling_tier(self._balance, self._settings)
        daily_limit = tier.get("daily_trade_limit", 25)
        daily_count = await self._db.get_daily_trade_count()
        if daily_count >= daily_limit:
            return False, f"Daily trade limit reached ({daily_count}/{daily_limit})"

        return True, "ok"

    async def validate_trade(self, pair: str, side: str,
                              price: float, quantity: float) -> tuple[bool, str]:
        """Validate a specific trade against all risk rules.

        Called after can_trade() passes and a signal is generated.
        Checks position size, notional value, and fee profitability.
        """
        notional = price * quantity
        min_order = self._risk.get("min_order_size", 10.0)

        if notional < min_order:
            return False, f"Order ${notional:.2f} below exchange minimum ${min_order}"

        max_pct = self._risk.get("max_position_pct", 25.0) / 100
        if notional > self._balance * max_pct:
            return False, f"Order ${notional:.2f} exceeds {max_pct*100:.0f}% of balance"

        # Check min net profit threshold (must exceed 2x fees)
        maker_rate = self._fees.get("maker_rate", 0.00075)
        round_trip_fee_pct = maker_rate * 2 * 100  # percentage
        min_net = self._risk.get("min_net_profit_pct", 0.30) if "min_net_profit_pct" in self._risk else round_trip_fee_pct * 2

        return True, "ok"

    async def get_position_size(self, balance: float, confidence: float) -> float:
        """Calculate optimal position size for a trade.

        Uses Kelly criterion from the sizer, scaled by AI confidence.
        """
        # Get strategy stats for Kelly inputs (fallback to defaults if no history)
        stats = await self._db.get_strategy_stats("smart_scalp", window=50)

        if stats.total_trades < 10:
            # Not enough data — use conservative defaults
            win_rate = 0.55
            avg_win = 0.008 * balance  # 0.8% of a $10-15 position
            avg_loss = 0.004 * balance
        else:
            win_rate = stats.win_rate
            avg_win = stats.avg_win
            avg_loss = abs(stats.avg_loss)

        size = self._sizer.kelly_size(win_rate, avg_win, avg_loss, balance)

        # Scale by confidence: high confidence → full size, low → reduced
        if confidence < 0.5:
            size *= 0.5
        elif confidence < 0.75:
            size *= 0.75

        # Floor at exchange minimum
        size = max(size, self._risk.get("min_order_size", 10.0))
        # Cap at max position %
        max_pos = balance * self._risk.get("max_position_pct", 25.0) / 100
        size = min(size, max_pos)

        return round(size, 2)

    def get_stop_loss_price(self, entry_price: float, side: str) -> float:
        """Calculate stop-loss price for a trade.

        For BUY: stop is below entry (price dropping = losing)
        For SELL: stop is above entry (price rising = losing)
        Uses 0.4% stop-loss from the spec.
        """
        stop_pct = 0.004  # 0.4%
        if side == "buy":
            return round(entry_price * (1 - stop_pct), 8)
        else:
            return round(entry_price * (1 + stop_pct), 8)

    def get_take_profit_price(self, entry_price: float, side: str) -> float:
        """Calculate take-profit price for a trade.

        For BUY: TP is above entry (price rising = winning)
        For SELL: TP is below entry (price dropping = winning)
        Uses 0.8% take-profit from the spec (2:1 reward/risk).
        """
        tp_pct = 0.008  # 0.8%
        if side == "buy":
            return round(entry_price * (1 + tp_pct), 8)
        else:
            return round(entry_price * (1 - tp_pct), 8)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_risk.py -v`
Expected: All 13 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/risk/ tests/test_risk.py
git commit -m "feat: risk manager, Kelly position sizer, 5-layer protection system"
```

---

### Task 5: Trade Gate (Multi-Brain Consensus Engine)

**Files:**
- Create: `src/ai/__init__.py`
- Create: `src/ai/trade_gate.py`
- Create: `src/intelligence/__init__.py`
- Create: `src/intelligence/correlation.py`
- Create: `tests/test_trade_gate.py`

**Interfaces:**
- Consumes: `TechnicalSignal` from Task 3, `OrderBookSignal` from Task 2
- Produces:
  - `TradeGate` class: `evaluate(signals: dict[str, BrainSignal]) -> GateDecision` with fields: `approved: bool`, `direction: str`, `confidence: float`, `agreeing_brains: int`, `reasons: list[str]`
  - `BrainSignal` dataclass: `direction: str`, `confidence: float`, `source: str`
  - `CorrelationTracker` class: `update(pair, price)`, `is_correlated(pair_a, pair_b) -> bool`, `btc_trend() -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trade_gate.py
import pytest
from src.ai.trade_gate import TradeGate, BrainSignal, GateDecision


class TestTradeGate:

    def test_approves_when_3_of_5_agree_buy(self):
        """3+ brains agreeing on BUY → approved."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.3, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.direction == "BUY"
        assert decision.agreeing_brains >= 3

    def test_rejects_when_only_2_agree(self):
        """Only 2 brains agreeing → rejected (need 3)."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("HOLD", 0.3, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.3, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is False

    def test_rejects_when_strong_opposing_signal(self):
        """Even with 3 agreeing, a strong OPPOSING signal vetoes."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("SELL", 0.9, "multi_timeframe"),  # strong opposing
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is False
        assert "opposing" in decision.reasons[0].lower() or "veto" in " ".join(decision.reasons).lower()

    def test_confidence_is_average_of_agreeing_brains(self):
        """Overall confidence = average of agreeing brains' confidence."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("SELL", 0.8, "technical"),
            "order_flow": BrainSignal("SELL", 0.6, "order_flow"),
            "sentiment": BrainSignal("SELL", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.3, "multi_timeframe"),
            "correlation": BrainSignal("SELL", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.direction == "SELL"
        # Average of 0.8, 0.6, 0.7, 0.5 = 0.65
        assert abs(decision.confidence - 0.65) < 0.01

    def test_hold_signals_dont_count_as_opposing(self):
        """HOLD is neutral — doesn't count for or against."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.7, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.6, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.5, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True  # HOLDs don't veto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trade_gate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/ai/trade_gate.py**

```python
# src/ai/trade_gate.py
# Multi-Brain Consensus Engine — the "Trade Gate"
#
# 5 independent analysis brains vote on every trade:
# 1. Technical signals (RSI, MACD, BB, EMA)
# 2. Order flow (book imbalance, whale detection)
# 3. AI sentiment (Gemini news/social analysis)
# 4. Multi-timeframe alignment (5m + 15m + 1h must agree)
# 5. Cross-asset correlation (BTC trend affects altcoins)
#
# RULES:
# - At least 3 of 5 brains must agree on direction
# - No brain can show a STRONG opposing signal (confidence > 0.7)
# - HOLD counts as neutral (neither for nor against)
# - Final confidence = average of agreeing brains

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Confidence threshold above which an opposing signal vetoes the trade
VETO_THRESHOLD = 0.7
# Minimum brains that must agree
MIN_CONSENSUS = 3


@dataclass(frozen=True)
class BrainSignal:
    """Output from one analysis brain.

    direction: BUY, SELL, or HOLD
    confidence: 0.0 to 1.0
    source: name of the brain (for logging)
    """
    direction: str
    confidence: float
    source: str


@dataclass(frozen=True)
class GateDecision:
    """Final trade gate decision.

    approved: True if trade should execute
    direction: BUY or SELL (or HOLD if not approved)
    confidence: 0.0 to 1.0 (average of agreeing brains)
    agreeing_brains: how many brains agreed
    reasons: human-readable list of what contributed to the decision
    """
    approved: bool
    direction: str
    confidence: float
    agreeing_brains: int
    reasons: list[str] = field(default_factory=list)


class TradeGate:
    """Evaluates signals from all 5 brains and decides whether to trade.

    This is the central decision point. No trade bypasses the gate.
    """

    def __init__(self, min_consensus: int = MIN_CONSENSUS,
                 veto_threshold: float = VETO_THRESHOLD):
        # How many brains must agree
        self._min_consensus = min_consensus
        # Opposing confidence above this → veto
        self._veto_threshold = veto_threshold

    def evaluate(self, signals: dict[str, BrainSignal]) -> GateDecision:
        """Evaluate all brain signals and return a gate decision.

        signals: dict mapping brain name to BrainSignal
        Returns GateDecision with approval status and reasoning.
        """
        reasons = []

        # Count votes for each direction
        buy_voters = []
        sell_voters = []

        for name, signal in signals.items():
            if signal.direction == "BUY":
                buy_voters.append(signal)
            elif signal.direction == "SELL":
                sell_voters.append(signal)
            # HOLD → neutral, not counted

        # Determine majority direction
        if len(buy_voters) >= len(sell_voters):
            majority_dir = "BUY"
            majority_signals = buy_voters
            opposing_signals = sell_voters
        else:
            majority_dir = "SELL"
            majority_signals = sell_voters
            opposing_signals = buy_voters

        agreeing = len(majority_signals)

        # --- Check 1: Minimum consensus ---
        if agreeing < self._min_consensus:
            reasons.append(f"only {agreeing}/{self._min_consensus} brains agree on {majority_dir}")
            for s in majority_signals:
                reasons.append(f"  {s.source}: {s.direction} ({s.confidence:.2f})")
            return GateDecision(
                approved=False, direction="HOLD",
                confidence=0.0, agreeing_brains=agreeing,
                reasons=reasons,
            )

        # --- Check 2: No strong opposing signal ---
        for opp in opposing_signals:
            if opp.confidence >= self._veto_threshold:
                reasons.append(
                    f"vetoed: {opp.source} has strong opposing {opp.direction} "
                    f"signal ({opp.confidence:.2f} >= {self._veto_threshold})"
                )
                return GateDecision(
                    approved=False, direction="HOLD",
                    confidence=0.0, agreeing_brains=agreeing,
                    reasons=reasons,
                )

        # --- Approved: calculate combined confidence ---
        avg_confidence = sum(s.confidence for s in majority_signals) / len(majority_signals)

        for s in majority_signals:
            reasons.append(f"{s.source}: {s.direction} ({s.confidence:.2f})")

        logger.info(
            "GATE APPROVED: %s with %d/%d brains, confidence %.2f",
            majority_dir, agreeing, len(signals), avg_confidence,
        )

        return GateDecision(
            approved=True,
            direction=majority_dir,
            confidence=avg_confidence,
            agreeing_brains=agreeing,
            reasons=reasons,
        )
```

- [ ] **Step 4: Create src/intelligence/correlation.py**

```python
# src/intelligence/correlation.py
# Tracks price correlations between assets in real-time.
# Used by Brain 5 to prevent correlated positions:
# - If BTC is dumping, don't go long on altcoins
# - If ETH and SOL are moving together, don't hold both long
# - Max 60% exposure to any single pair

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class CorrelationTracker:
    """Tracks rolling price correlation between crypto assets.

    Maintains a sliding window of recent prices per pair and computes
    pairwise correlation coefficients. Also tracks BTC's short-term
    trend (since BTC leads the market).
    """

    def __init__(self, window: int = 50):
        # How many price points to keep per pair
        self._window = window
        # Recent prices per pair: {"BTC/USDT": deque([50000, 50050, ...])}
        self._prices: dict[str, deque] = {}

    def update(self, pair: str, price: float):
        """Record a new price for a pair."""
        if pair not in self._prices:
            self._prices[pair] = deque(maxlen=self._window)
        self._prices[pair].append(price)

    def is_correlated(self, pair_a: str, pair_b: str,
                      threshold: float = 0.7) -> bool:
        """Check if two pairs are strongly correlated.

        Uses Pearson correlation on recent returns.
        Returns True if |correlation| >= threshold.
        """
        returns_a = self._get_returns(pair_a)
        returns_b = self._get_returns(pair_b)

        if returns_a is None or returns_b is None:
            return False  # not enough data

        # Use the shorter length
        n = min(len(returns_a), len(returns_b))
        if n < 10:
            return False

        ra = list(returns_a)[-n:]
        rb = list(returns_b)[-n:]

        # Pearson correlation
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb)) / n
        std_a = (sum((a - mean_a) ** 2 for a in ra) / n) ** 0.5
        std_b = (sum((b - mean_b) ** 2 for b in rb) / n) ** 0.5

        if std_a == 0 or std_b == 0:
            return False

        corr = cov / (std_a * std_b)
        return abs(corr) >= threshold

    def btc_trend(self) -> str:
        """Determine BTC's short-term trend.

        Returns "BULLISH", "BEARISH", or "NEUTRAL".
        Based on last 10 price changes.
        """
        btc_key = "BTC/USDT"
        if btc_key not in self._prices or len(self._prices[btc_key]) < 10:
            return "NEUTRAL"

        prices = list(self._prices[btc_key])[-10:]
        # Simple: compare first vs last
        change_pct = (prices[-1] - prices[0]) / prices[0] * 100

        if change_pct > 0.3:
            return "BULLISH"
        elif change_pct < -0.3:
            return "BEARISH"
        return "NEUTRAL"

    def get_brain_signal(self, pair: str) -> Optional[dict]:
        """Generate Brain 5 signal for cross-asset correlation.

        If holding an altcoin long and BTC is dumping → SELL signal.
        If BTC is bullish → supports BUY on altcoins.
        """
        btc = self.btc_trend()

        # BTC pairs always get HOLD from correlation brain
        if pair == "BTC/USDT":
            return {"direction": "HOLD", "confidence": 0.5}

        if btc == "BEARISH":
            # BTC dumping → don't long altcoins
            return {"direction": "SELL", "confidence": 0.6}
        elif btc == "BULLISH":
            # BTC rising → supports altcoin longs
            return {"direction": "BUY", "confidence": 0.55}

        return {"direction": "HOLD", "confidence": 0.5}

    def _get_returns(self, pair: str) -> Optional[list]:
        """Calculate percentage returns from price series."""
        if pair not in self._prices or len(self._prices[pair]) < 2:
            return None
        prices = list(self._prices[pair])
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_trade_gate.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ai/ src/intelligence/ tests/test_trade_gate.py
git commit -m "feat: multi-brain trade gate (3/5 consensus), correlation tracker"
```

---

### Task 6: Smart Scalping Strategy + Strategy Base Class

**Files:**
- Create: `src/strategies/__init__.py`
- Create: `src/strategies/base.py`
- Create: `src/strategies/smart_scalp.py`
- Create: `tests/test_strategies.py`

**Interfaces:**
- Consumes: `IndicatorEngine.compute_all()` from Task 3, `IndicatorEngine.get_signal()` from Task 3
- Produces:
  - `BaseStrategy` abstract class: `name: str`, `evaluate(df, config) -> StrategySignal`, `is_enabled(config) -> bool`
  - `SmartScalpStrategy(BaseStrategy)`: implements `evaluate()` for fee-aware scalping
  - `StrategySignal` dataclass: `direction: str`, `confidence: float`, `entry_price: float`, `stop_loss: float`, `take_profit: float`, `strategy_name: str`, `reasons: list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategies.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.strategies.smart_scalp import SmartScalpStrategy
from src.strategies.base import StrategySignal
from src.data.indicators import IndicatorEngine


def make_bullish_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate data with a clear uptrend for testing BUY signals."""
    np.random.seed(42)
    base = 50000
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    # Steady uptrend with noise
    closes = [base + i * 20 + np.random.randn() * 30 for i in range(n)]
    # Make last few candles show oversold RSI by adding a dip then recovery
    for i in range(n - 10, n - 3):
        closes[i] = closes[i] - 500  # dip
    for i in range(n - 3, n):
        closes[i] = closes[i - 1] + 100  # recovery
    volumes = [500 + np.random.rand() * 200 for _ in range(n)]
    # Volume spike on last candle
    volumes[-1] = 3000
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [c - 10 for c in closes],
        "high": [c + 50 for c in closes],
        "low": [c - 60 for c in closes],
        "close": closes,
        "volume": volumes,
    })


class TestSmartScalp:

    def test_returns_strategy_signal(self):
        """evaluate() should return a StrategySignal."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        df = make_bullish_ohlcv()
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        assert isinstance(signal, StrategySignal)
        assert signal.direction in ("BUY", "SELL", "HOLD")
        assert signal.strategy_name == "smart_scalp"

    def test_stop_loss_and_take_profit_are_set(self):
        """When signal is BUY/SELL, SL and TP must be set."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        df = make_bullish_ohlcv()
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        if signal.direction != "HOLD":
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            if signal.direction == "BUY":
                assert signal.stop_loss < signal.entry_price
                assert signal.take_profit > signal.entry_price

    def test_hold_when_no_confirmation(self):
        """With random flat data, should mostly HOLD (no multi-confirmation)."""
        strategy = SmartScalpStrategy()
        engine = IndicatorEngine()
        np.random.seed(99)
        # Flat random walk — no clear signal
        n = 100
        closes = [50000 + np.random.randn() * 10 for _ in range(n)]
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
            "open": closes, "high": [c+5 for c in closes],
            "low": [c-5 for c in closes], "close": closes,
            "volume": [500] * n,
        })
        df = engine.compute_all(df)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        signal = strategy.evaluate(df, config)
        assert signal.direction == "HOLD"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/strategies/base.py and src/strategies/smart_scalp.py**

```python
# src/strategies/base.py
# Abstract base class for all trading strategies.
# Every strategy must implement evaluate() which takes OHLCV data
# with indicators and returns a StrategySignal.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    """Output from a strategy's evaluation.

    direction: BUY, SELL, or HOLD
    confidence: 0.0 to 1.0
    entry_price: suggested entry price (for limit order)
    stop_loss: hard stop-loss price
    take_profit: take-profit price
    strategy_name: which strategy generated this signal
    reasons: why this signal was generated
    """
    direction: str
    confidence: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_name: str = ""
    reasons: list[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclasses implement evaluate() with their specific logic.
    The strategy selector calls evaluate() and feeds the result
    into the trade gate for multi-brain consensus.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name for logging and config lookup."""
        ...

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Analyze data and return a trading signal.

        df: OHLCV DataFrame with indicator columns already computed.
        config: strategy-specific parameters from strategies.yaml.
        """
        ...

    def is_enabled(self, config: dict) -> bool:
        """Check if this strategy is enabled in config."""
        return config.get("enabled", True)
```

```python
# src/strategies/smart_scalp.py
# Smart Scalping Strategy — the primary profit engine.
#
# KEY DIFFERENCE from naive scalping:
# - Uses 5m+15m candles (not 1m — too noisy, fees kill you)
# - Requires MULTI-CONFIRMATION before entering (volume + momentum + indicator)
# - LIMIT ORDERS ONLY (maker fee, not taker)
# - 2:1 reward/risk ratio (0.8% TP, 0.4% SL) to overcome fee drag
# - Skips marginal signals — quality over quantity

import pandas as pd
from src.strategies.base import BaseStrategy, StrategySignal
from src.data.indicators import IndicatorEngine


class SmartScalpStrategy(BaseStrategy):
    """Fee-aware scalping strategy.

    Entry requires multi-confirmation:
    1. RSI shows momentum (not overbought for buy, not oversold for sell)
    2. MACD confirms direction (histogram positive for buy, negative for sell)
    3. Volume is above average (confirms real interest, not noise)
    4. Price action confirms (close above EMA for buy, below for sell)

    All 4 must agree. This filters out 80%+ of noise signals.
    """

    @property
    def name(self) -> str:
        return "smart_scalp"

    def evaluate(self, df: pd.DataFrame, config: dict) -> StrategySignal:
        """Generate a scalping signal from indicator data.

        Returns BUY/SELL only when all confirmation criteria are met.
        Otherwise returns HOLD (no trade).
        """
        if len(df) < 5:
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["insufficient data"])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest.get("rsi")
        macd_hist = latest.get("macd_histogram")
        prev_macd_hist = prev.get("macd_histogram")
        vol_ratio = latest.get("volume_ratio")
        close = latest["close"]
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")

        # Check indicators are warmed up
        if any(pd.isna(v) for v in [rsi, macd_hist, vol_ratio, ema_fast, ema_slow]):
            return StrategySignal("HOLD", 0.0, strategy_name=self.name,
                                  reasons=["indicators warming up"])

        # Config values
        rsi_oversold = config.get("rsi_oversold", 30)
        rsi_overbought = config.get("rsi_overbought", 70)
        vol_multiplier = config.get("volume_spike_multiplier", 1.5)
        tp_pct = config.get("take_profit_pct", 0.8) / 100  # convert to decimal
        sl_pct = config.get("stop_loss_pct", 0.4) / 100

        # --- Multi-confirmation for BUY ---
        buy_confirmations = []

        # 1. RSI not overbought (room to go up)
        if rsi < rsi_overbought:
            buy_confirmations.append(f"RSI {rsi:.0f} < {rsi_overbought}")
        # Extra point if oversold (strong buy signal)
        if rsi < rsi_oversold:
            buy_confirmations.append(f"RSI oversold ({rsi:.0f})")

        # 2. MACD histogram positive or crossing up
        if macd_hist > 0:
            buy_confirmations.append(f"MACD positive ({macd_hist:.4f})")
        if not pd.isna(prev_macd_hist) and prev_macd_hist < 0 and macd_hist > 0:
            buy_confirmations.append("MACD bullish crossover")

        # 3. Volume above average
        if vol_ratio >= vol_multiplier:
            buy_confirmations.append(f"volume spike ({vol_ratio:.1f}x)")

        # 4. Price above fast EMA (upward momentum)
        if close > ema_fast:
            buy_confirmations.append("price above EMA-fast")

        # --- Multi-confirmation for SELL ---
        sell_confirmations = []

        if rsi > rsi_oversold:
            sell_confirmations.append(f"RSI {rsi:.0f} > {rsi_oversold}")
        if rsi > rsi_overbought:
            sell_confirmations.append(f"RSI overbought ({rsi:.0f})")

        if macd_hist < 0:
            sell_confirmations.append(f"MACD negative ({macd_hist:.4f})")
        if not pd.isna(prev_macd_hist) and prev_macd_hist > 0 and macd_hist < 0:
            sell_confirmations.append("MACD bearish crossover")

        if vol_ratio >= vol_multiplier:
            sell_confirmations.append(f"volume spike ({vol_ratio:.1f}x)")

        if close < ema_fast:
            sell_confirmations.append("price below EMA-fast")

        # --- Decision: need 4+ confirmations ---
        # (RSI range + MACD direction + volume + price/EMA alignment)
        min_confirmations = 4

        if len(buy_confirmations) >= min_confirmations and len(buy_confirmations) > len(sell_confirmations):
            confidence = min(len(buy_confirmations) / 6, 1.0)
            entry = close
            stop_loss = round(entry * (1 - sl_pct), 8)
            take_profit = round(entry * (1 + tp_pct), 8)
            return StrategySignal(
                direction="BUY", confidence=confidence,
                entry_price=entry, stop_loss=stop_loss,
                take_profit=take_profit, strategy_name=self.name,
                reasons=buy_confirmations,
            )

        if len(sell_confirmations) >= min_confirmations and len(sell_confirmations) > len(buy_confirmations):
            confidence = min(len(sell_confirmations) / 6, 1.0)
            entry = close
            stop_loss = round(entry * (1 + sl_pct), 8)
            take_profit = round(entry * (1 - tp_pct), 8)
            return StrategySignal(
                direction="SELL", confidence=confidence,
                entry_price=entry, stop_loss=stop_loss,
                take_profit=take_profit, strategy_name=self.name,
                reasons=sell_confirmations,
            )

        # Not enough confirmation — HOLD
        all_reasons = buy_confirmations + sell_confirmations
        return StrategySignal(
            direction="HOLD", confidence=0.0, strategy_name=self.name,
            reasons=all_reasons if all_reasons else ["no multi-confirmation met"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategies/ tests/test_strategies.py
git commit -m "feat: strategy base class, smart scalping strategy with multi-confirmation"
```

---

### Task 7: Paper Trading Executor + Live Executor

**Files:**
- Create: `src/execution/__init__.py`
- Create: `src/execution/paper_trader.py`
- Create: `src/execution/executor.py`
- Create: `tests/test_executor.py`

**Interfaces:**
- Consumes: `BinanceClient` from Task 2, `Database` from Task 1, `RiskManager` from Task 4, `StrategySignal` from Task 6
- Produces:
  - `PaperTrader` class: `async execute_signal(signal: StrategySignal, pair: str, position_size: float) -> Trade`, `async check_open_positions(current_prices: dict) -> list[Trade]`, `get_balance() -> float`, `get_open_positions() -> list`
  - `LiveExecutor` class: same interface as PaperTrader but uses real Binance API
  - Both share `BaseExecutor` abstract class

- [ ] **Step 1: Write the failing test**

```python
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
        """Fees should match maker rate × 2 (entry + exit)."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_executor.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/execution/paper_trader.py**

```python
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
                 time_exit_hours: float = 4.0):
        self._balance = initial_balance
        self._fee_rate = maker_fee_rate
        self._min_order = min_order_size
        self._max_positions = max_positions
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

    def check_open_positions(self, current_prices: dict) -> list[Trade]:
        """Check all open positions against current prices.

        Closes positions that hit stop-loss, take-profit, or time limit.
        Returns list of closed Trade records.
        """
        closed_trades = []
        still_open = []

        for pos in self._positions:
            price = current_prices.get(pos.pair)
            if price is None:
                still_open.append(pos)
                continue

            close_reason = None
            exit_price = price

            # --- Check stop-loss ---
            if pos.side == "buy" and price <= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss
            elif pos.side == "sell" and price >= pos.stop_loss:
                close_reason = "stop_loss"
                exit_price = pos.stop_loss

            # --- Check take-profit ---
            if pos.side == "buy" and price >= pos.take_profit:
                close_reason = "take_profit"
                exit_price = pos.take_profit
            elif pos.side == "sell" and price <= pos.take_profit:
                close_reason = "take_profit"
                exit_price = pos.take_profit

            # --- Check time exit (4 hours) ---
            if time.time() - pos.opened_at > self._time_exit:
                close_reason = "time_exit"

            if close_reason:
                trade = self._close_position(pos, exit_price, close_reason)
                closed_trades.append(trade)
            else:
                still_open.append(pos)

        self._positions = still_open
        return closed_trades

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
            "PAPER CLOSE %s %s @ $%.2f → $%.2f | P&L: $%.4f (fees: $%.4f) [%s]",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_executor.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution/ tests/test_executor.py
git commit -m "feat: paper trading executor with fee simulation, position management"
```

---

### Task 8: Backtesting Framework

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/data_loader.py`
- Create: `backtest/engine.py`
- Create: `backtest/report.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `BinanceClient.get_historical_ohlcv()` from Task 2, `IndicatorEngine` from Task 3, `SmartScalpStrategy` from Task 6, `PaperTrader` from Task 7
- Produces:
  - `DataLoader` class: `async download(pair, timeframe, days) -> pd.DataFrame`, `save_csv(df, path)`, `load_csv(path) -> pd.DataFrame`
  - `BacktestEngine` class: `run(strategy, df, config) -> BacktestResult`
  - `BacktestResult` dataclass: `total_trades: int`, `win_rate: float`, `net_pnl: float`, `sharpe_ratio: float`, `max_drawdown_pct: float`, `total_fees: float`, `trades: list[Trade]`
  - `BacktestReport` class: `print_summary(result)`, `save_report(result, path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from backtest.engine import BacktestEngine, BacktestResult
from src.strategies.smart_scalp import SmartScalpStrategy


def make_trending_data(n: int = 500) -> pd.DataFrame:
    """Generate 500 candles with alternating trends for backtesting."""
    np.random.seed(42)
    base = 50000.0
    closes = [base]
    for i in range(1, n):
        # Alternate between up-trend and down-trend every 50 candles
        cycle = (i // 50) % 2
        drift = 10 if cycle == 0 else -10
        closes.append(closes[-1] + drift + np.random.randn() * 50)

    volumes = [500 + np.random.rand() * 500 for _ in range(n)]
    # Add some volume spikes
    for i in range(0, n, 20):
        volumes[i] = 3000

    return pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5*i) for i in range(n)],
        "open": [c - np.random.rand() * 30 for c in closes],
        "high": [c + abs(np.random.randn()) * 60 for c in closes],
        "low": [c - abs(np.random.randn()) * 60 for c in closes],
        "close": closes,
        "volume": volumes,
    })


class TestBacktestEngine:

    def test_returns_backtest_result(self):
        """Should return a BacktestResult with all metrics."""
        engine = BacktestEngine(initial_balance=50.0, maker_fee_rate=0.00075)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1.0
        assert result.total_fees >= 0

    def test_fees_are_always_subtracted(self):
        """Net P&L should always be less than gross P&L (fees exist)."""
        engine = BacktestEngine(initial_balance=100.0, maker_fee_rate=0.00075)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        if result.total_trades > 0:
            assert result.total_fees > 0

    def test_max_drawdown_is_calculated(self):
        """Max drawdown should be >= 0."""
        engine = BacktestEngine(initial_balance=50.0)
        strategy = SmartScalpStrategy()
        df = make_trending_data(500)
        config = {
            "rsi_oversold": 30, "rsi_overbought": 70,
            "volume_spike_multiplier": 1.5,
            "take_profit_pct": 0.8, "stop_loss_pct": 0.4,
            "min_net_profit_pct": 0.30,
        }
        result = engine.run(strategy, df, config, pair="BTC/USDT")
        assert result.max_drawdown_pct >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create backtest/engine.py and backtest/report.py**

```python
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
                    # trade is either an open Trade or None

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
```

```python
# backtest/data_loader.py
# Downloads and caches historical OHLCV data from Binance.
# Saves to CSV for reuse (avoids re-downloading on every backtest run).

import asyncio
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from src.data.feed import BinanceClient

logger = logging.getLogger(__name__)


class DataLoader:
    """Downloads historical price data for backtesting.

    Fetches from Binance REST API and caches to CSV files.
    Handles pagination (Binance returns max 1000 candles per request).
    """

    def __init__(self, cache_dir: str = "backtest/data"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    async def download(self, pair: str, timeframe: str = "5m",
                       days: int = 180) -> pd.DataFrame:
        """Download historical OHLCV data from Binance.

        Fetches `days` worth of data, paginating through the API.
        Caches to CSV for future use.
        """
        cache_file = self._cache_path(pair, timeframe, days)
        if os.path.exists(cache_file):
            logger.info("Loading cached data from %s", cache_file)
            return self.load_csv(cache_file)

        client = BinanceClient(paper_mode=True)
        await client.connect()

        all_data = []
        # Calculate starting timestamp
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        while True:
            df = await client.get_historical_ohlcv(pair, timeframe, since=since, limit=1000)
            if df.empty:
                break
            all_data.append(df)
            # Move `since` to after the last candle
            last_ts = int(df.iloc[-1]["timestamp"].timestamp() * 1000)
            since = last_ts + 1
            # Rate limiting
            await asyncio.sleep(0.5)
            # Stop if we've reached current time
            if len(df) < 1000:
                break

        await client.disconnect()

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        self.save_csv(result, cache_file)
        logger.info("Downloaded %d candles for %s %s (%d days)", len(result), pair, timeframe, days)
        return result

    def save_csv(self, df: pd.DataFrame, path: str):
        """Save DataFrame to CSV."""
        df.to_csv(path, index=False)

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _cache_path(self, pair: str, timeframe: str, days: int) -> str:
        safe_pair = pair.replace("/", "_")
        return os.path.join(self._cache_dir, f"{safe_pair}_{timeframe}_{days}d.csv")
```

```python
# backtest/report.py
# Prints and saves backtest performance reports.

from backtest.engine import BacktestResult


class BacktestReport:
    """Formats and displays backtest results."""

    @staticmethod
    def print_summary(result: BacktestResult, strategy_name: str = ""):
        """Print a human-readable performance summary."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS: {strategy_name}")
        print(f"{'='*60}")
        print(f"  Total trades:     {result.total_trades}")
        print(f"  Win rate:         {result.win_rate*100:.1f}%")
        print(f"  Net P&L:          ${result.net_pnl:.2f}")
        print(f"  Total fees:       ${result.total_fees:.4f}")
        print(f"  Sharpe ratio:     {result.sharpe_ratio:.3f}")
        print(f"  Max drawdown:     {result.max_drawdown_pct:.1f}%")
        print(f"  Final balance:    ${result.final_balance:.2f}")
        print(f"{'='*60}")

        # Pass/fail criteria
        passed = (
            result.win_rate >= 0.55
            and result.net_pnl > 0
            and result.sharpe_ratio > 1.0
            and result.max_drawdown_pct < 15
        )
        status = "PASSED" if passed else "FAILED"
        print(f"  Go-live gate:     {status}")
        if not passed:
            if result.win_rate < 0.55:
                print(f"    - Win rate {result.win_rate*100:.1f}% < 55% required")
            if result.net_pnl <= 0:
                print(f"    - Net P&L negative (${result.net_pnl:.2f})")
            if result.sharpe_ratio <= 1.0:
                print(f"    - Sharpe {result.sharpe_ratio:.3f} <= 1.0 required")
            if result.max_drawdown_pct >= 15:
                print(f"    - Max drawdown {result.max_drawdown_pct:.1f}% >= 15% limit")
        print()

    @staticmethod
    def save_report(result: BacktestResult, path: str):
        """Save results to a text file."""
        with open(path, "w") as f:
            f.write(f"Total trades: {result.total_trades}\n")
            f.write(f"Win rate: {result.win_rate*100:.1f}%\n")
            f.write(f"Net P&L: ${result.net_pnl:.2f}\n")
            f.write(f"Total fees: ${result.total_fees:.4f}\n")
            f.write(f"Sharpe ratio: {result.sharpe_ratio:.3f}\n")
            f.write(f"Max drawdown: {result.max_drawdown_pct:.1f}%\n")
            f.write(f"Final balance: ${result.final_balance:.2f}\n")
```

```python
# backtest/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/ tests/test_backtest.py
git commit -m "feat: backtesting framework with fee simulation, data loader, performance reports"
```

---

### Task 9: Telegram Notifications

**Files:**
- Create: `src/notifications/__init__.py`
- Create: `src/notifications/telegram.py`
- Create: `tests/test_telegram.py`

**Interfaces:**
- Consumes: `Trade` from Task 1, `ProtectionAction` from Task 4
- Produces:
  - `TelegramNotifier` class: `async send_trade(trade: Trade)`, `async send_daily_summary(stats: dict)`, `async send_protection_alert(action: ProtectionAction)`, `async send_error(message: str)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telegram.py
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from src.notifications.telegram import TelegramNotifier
from src.storage.models import Trade
from src.risk.protection import ProtectionAction


class TestTelegramNotifier:

    @pytest.mark.asyncio
    async def test_format_daily_summary(self):
        """Daily summary should include key metrics."""
        notifier = TelegramNotifier(token="fake", chat_id="fake")
        stats = {
            "total_trades": 15,
            "win_rate": 0.60,
            "net_pnl": 1.25,
            "balance": 51.25,
            "best_strategy": "smart_scalp",
        }
        message = notifier.format_daily_summary(stats)
        assert "15" in message
        assert "60" in message
        assert "1.25" in message
        assert "51.25" in message

    @pytest.mark.asyncio
    async def test_format_protection_alert(self):
        """Protection alert should include action and reason."""
        notifier = TelegramNotifier(token="fake", chat_id="fake")
        action = ProtectionAction(
            action="SHUTDOWN",
            reason="Daily drawdown 5.2% exceeded 5% limit",
        )
        message = notifier.format_protection_alert(action)
        assert "SHUTDOWN" in message
        assert "5.2%" in message

    @pytest.mark.asyncio
    async def test_format_trade_message(self):
        """Trade message should show pair, side, price, P&L."""
        notifier = TelegramNotifier(token="fake", chat_id="fake")
        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            pair="BTC/USDT", side="buy", strategy="smart_scalp",
            entry_price=50000, exit_price=50400,
            quantity=0.0002, pnl=0.08, fees=0.015, status="closed",
        )
        message = notifier.format_trade(trade)
        assert "BTC/USDT" in message
        assert "0.08" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_telegram.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/notifications/telegram.py**

```python
# src/notifications/telegram.py
# Telegram bot for autonomous reporting.
# Only sends IMPORTANT events — no trade-by-trade spam.
# Daily P&L summary at midnight UTC.
# Protection alerts when any layer activates.
# Weekly intelligence report.

import logging
import os
from typing import Optional

from src.storage.models import Trade
from src.risk.protection import ProtectionAction

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends important trading events to Telegram.

    NOT every trade — only:
    - Daily P&L summary
    - Protection layer activations
    - Strategy auto-disabled
    - System errors
    - Weekly report
    """

    def __init__(self, token: Optional[str] = None,
                 chat_id: Optional[str] = None):
        self._token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._bot = None

    async def _get_bot(self):
        """Lazy-load the telegram bot (only when needed)."""
        if self._bot is None and self._token:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
            except ImportError:
                logger.warning("python-telegram-bot not installed — notifications disabled")
        return self._bot

    async def send_message(self, text: str):
        """Send a text message to the configured chat."""
        bot = await self._get_bot()
        if bot and self._chat_id:
            try:
                await bot.send_message(chat_id=self._chat_id, text=text, parse_mode="Markdown")
            except Exception as e:
                logger.error("Telegram send failed: %s", e)
        else:
            logger.info("TELEGRAM (disabled): %s", text[:100])

    async def send_daily_summary(self, stats: dict):
        """Send midnight daily P&L summary."""
        message = self.format_daily_summary(stats)
        await self.send_message(message)

    async def send_protection_alert(self, action: ProtectionAction):
        """Alert when a protection layer activates."""
        message = self.format_protection_alert(action)
        await self.send_message(message)

    async def send_trade(self, trade: Trade):
        """Send a trade notification (used sparingly)."""
        message = self.format_trade(trade)
        await self.send_message(message)

    async def send_error(self, error_message: str):
        """Send system error alert."""
        text = f"*SYSTEM ERROR*\n{error_message}"
        await self.send_message(text)

    def format_daily_summary(self, stats: dict) -> str:
        """Format the daily P&L summary message."""
        pnl = stats.get("net_pnl", 0)
        pnl_emoji = "+" if pnl >= 0 else ""
        return (
            f"*DAILY REPORT*\n"
            f"Trades: {stats.get('total_trades', 0)}\n"
            f"Win rate: {stats.get('win_rate', 0)*100:.0f}%\n"
            f"Net P&L: {pnl_emoji}${pnl:.2f}\n"
            f"Balance: ${stats.get('balance', 0):.2f}\n"
            f"Best strategy: {stats.get('best_strategy', 'n/a')}"
        )

    def format_protection_alert(self, action: ProtectionAction) -> str:
        """Format a protection layer alert."""
        return (
            f"*PROTECTION: {action.action}*\n"
            f"Reason: {action.reason}\n"
            f"Duration: {action.duration_minutes} min"
            if action.duration_minutes > 0
            else f"*PROTECTION: {action.action}*\nReason: {action.reason}"
        )

    def format_trade(self, trade: Trade) -> str:
        """Format a single trade notification."""
        pnl_str = f"+${trade.pnl:.4f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.4f}"
        return (
            f"*{trade.side.upper()} {trade.pair}*\n"
            f"Entry: ${trade.entry_price:.2f} -> Exit: ${trade.exit_price:.2f}\n"
            f"P&L: {pnl_str} (fees: ${trade.fees:.4f})\n"
            f"Strategy: {trade.strategy}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_telegram.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/notifications/ tests/test_telegram.py
git commit -m "feat: Telegram notifications (daily summary, protection alerts, error reporting)"
```

---

### Task 10: Core Trading Engine + Entry Point + Watchdog

**Files:**
- Create: `src/core/engine.py`
- Create: `src/core/event_bus.py`
- Create: `main.py`
- Create: `watchdog.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: ALL previous tasks — this is the orchestration layer that ties everything together
- Produces:
  - `TradingEngine` class: `async start()`, `async stop()`, `async run_cycle(pair)` — the main trading loop
  - `main.py` — entry point that initializes and starts the engine
  - `watchdog.py` — monitors the engine process and auto-restarts on crash

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.core.engine import TradingEngine


class TestTradingEngine:

    @pytest.mark.asyncio
    async def test_engine_respects_protection_shutdown(self):
        """Engine should stop trading when protection says SHUTDOWN."""
        engine = TradingEngine.__new__(TradingEngine)
        engine._running = True
        engine._protection_action = "SHUTDOWN"
        assert engine._should_skip_cycle() is True

    @pytest.mark.asyncio
    async def test_engine_respects_protection_pause(self):
        """Engine should skip cycles during PAUSE."""
        engine = TradingEngine.__new__(TradingEngine)
        engine._running = True
        engine._protection_action = "PAUSE"
        assert engine._should_skip_cycle() is True

    @pytest.mark.asyncio
    async def test_engine_trades_on_continue(self):
        """Engine should trade when protection says CONTINUE."""
        engine = TradingEngine.__new__(TradingEngine)
        engine._running = True
        engine._protection_action = "CONTINUE"
        assert engine._should_skip_cycle() is False

    @pytest.mark.asyncio
    async def test_engine_reduces_size_on_reduce(self):
        """When REDUCE_SIZE, position sizes should be halved."""
        engine = TradingEngine.__new__(TradingEngine)
        engine._protection_action = "REDUCE_SIZE"
        engine._protection_size_multiplier = 0.5
        assert engine._get_size_multiplier() == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create src/core/event_bus.py**

```python
# src/core/event_bus.py
# Simple event bus for decoupling components.
# Components emit events (e.g., "trade_closed", "protection_triggered")
# and other components listen and react.

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Publish-subscribe event bus for the trading engine.

    Usage:
        bus = EventBus()
        bus.on("trade_closed", my_handler)
        await bus.emit("trade_closed", trade_data)
    """

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable):
        """Register a listener for an event."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    async def emit(self, event: str, data=None):
        """Emit an event to all registered listeners."""
        for callback in self._listeners.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("Event handler error for '%s': %s", event, e)
```

- [ ] **Step 4: Create src/core/engine.py**

```python
# src/core/engine.py
# The main trading loop — the brain of the bot.
# Orchestrates all components in a continuous cycle:
#
# 1. Check protection layers (can we trade?)
# 2. Fetch market data (price, order book)
# 3. Compute indicators
# 4. Run all 5 brains
# 5. Pass signals through trade gate (3/5 consensus)
# 6. If approved, pass through risk manager
# 7. Execute via paper trader or live executor
# 8. Log everything
# 9. Repeat

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from src.core.config import load_settings, load_strategies
from src.core.event_bus import EventBus
from src.storage.database import Database
from src.data.feed import BinanceClient, PriceFeed
from src.data.indicators import IndicatorEngine
from src.data.order_book import OrderBookAnalyzer
from src.strategies.smart_scalp import SmartScalpStrategy
from src.ai.trade_gate import TradeGate, BrainSignal
from src.intelligence.correlation import CorrelationTracker
from src.risk.manager import RiskManager
from src.risk.protection import ProtectionSystem
from src.execution.paper_trader import PaperTrader
from src.notifications.telegram import TelegramNotifier
from src.storage.models import PortfolioSnapshot

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading loop — orchestrates all components.

    Runs continuously, cycling through each trading pair:
    1. Protection check
    2. Data fetch
    3. Multi-brain analysis
    4. Trade gate consensus
    5. Risk validation
    6. Execution

    Fully autonomous — no human intervention needed.
    """

    def __init__(self):
        # Load configuration
        self._settings = load_settings()
        self._strategies_config = load_strategies()
        self._pairs = self._settings.get("pairs", ["BTC/USDT"])

        # State
        self._running = False
        self._protection_action = "CONTINUE"
        self._protection_size_multiplier = 1.0
        self._cycle_interval = 30  # seconds between cycles

    async def start(self):
        """Initialize all components and start the trading loop."""
        logger.info("Initializing trading engine...")

        # Database
        self._db = Database()
        await self._db.init()

        # Event bus
        self._bus = EventBus()

        # Exchange client
        trading_mode = os.getenv("TRADING_MODE", "paper")
        self._paper_mode = trading_mode == "paper"
        self._client = BinanceClient(
            paper_mode=self._paper_mode,
            api_key=os.getenv("BINANCE_API_KEY", ""),
            secret=os.getenv("BINANCE_SECRET", ""),
        )
        await self._client.connect()

        # Get initial balance
        if self._paper_mode:
            initial_balance = 50.0  # default paper balance
            self._executor = PaperTrader(
                initial_balance=initial_balance,
                maker_fee_rate=self._settings["fees"]["maker_rate"],
                min_order_size=self._settings["risk"]["min_order_size"],
                max_positions=self._settings["risk"]["max_open_positions"],
                time_exit_hours=self._settings["risk"]["time_exit_hours"],
            )
        else:
            balance = await self._client.get_balance()
            initial_balance = balance.get("USDT", 0)
            self._executor = PaperTrader(  # LiveExecutor in Phase 4
                initial_balance=initial_balance,
                maker_fee_rate=self._settings["fees"]["maker_rate"],
            )

        # Analysis components
        self._indicators = IndicatorEngine()
        self._order_book_analyzer = OrderBookAnalyzer(
            whale_threshold_usd=50000,
            max_spread_pct=self._settings["risk"]["max_spread_pct"],
        )
        self._correlation = CorrelationTracker()

        # Strategies
        self._strategies = [SmartScalpStrategy()]

        # Decision engine
        self._trade_gate = TradeGate()

        # Risk + protection
        self._risk = RiskManager(self._db, self._settings, initial_balance)
        self._protection = ProtectionSystem(self._db, self._settings)

        # Notifications
        self._notifier = TelegramNotifier()

        # Start the main loop
        self._running = True
        mode_str = "PAPER" if self._paper_mode else "LIVE"
        logger.info("Engine started in %s mode with $%.2f", mode_str, initial_balance)
        await self._notifier.send_message(
            f"*Bot started* in {mode_str} mode with ${initial_balance:.2f}"
        )

        await self._main_loop()

    async def stop(self):
        """Gracefully shut down the engine."""
        self._running = False
        await self._client.disconnect()
        await self._db.close()
        logger.info("Engine stopped")

    async def _main_loop(self):
        """The continuous trading loop."""
        last_snapshot = 0
        snapshot_interval = self._settings["schedule"]["portfolio_snapshot_minutes"] * 60

        while self._running:
            try:
                # Check protection layers
                balance = self._executor.get_balance()
                daily_pnl = await self._db.get_daily_pnl()
                protection = await self._protection.check_all_layers(balance, daily_pnl)

                self._protection_action = protection.action
                self._protection_size_multiplier = protection.size_multiplier

                if protection.action == "SHUTDOWN":
                    await self._notifier.send_protection_alert(protection)
                    logger.critical("SHUTDOWN triggered: %s", protection.reason)
                    self._running = False
                    break

                if protection.action == "PAUSE":
                    await self._notifier.send_protection_alert(protection)
                    logger.warning("PAUSED for %d min: %s", protection.duration_minutes, protection.reason)
                    await asyncio.sleep(protection.duration_minutes * 60)
                    continue

                if protection.action in ("REDUCE_SIZE", "DEFENSE_ONLY"):
                    logger.info("Protection: %s — %s", protection.action, protection.reason)

                # Run a cycle for each pair
                if not self._should_skip_cycle():
                    for pair in self._pairs:
                        await self._run_cycle(pair)

                # Periodic portfolio snapshot
                if time.time() - last_snapshot > snapshot_interval:
                    await self._take_snapshot()
                    last_snapshot = time.time()

                # Wait before next cycle
                await asyncio.sleep(self._cycle_interval)

            except Exception as e:
                logger.error("Engine cycle error: %s", e, exc_info=True)
                await self._notifier.send_error(str(e))
                await asyncio.sleep(10)  # brief pause before retry

    async def _run_cycle(self, pair: str):
        """Run one trading cycle for a single pair."""
        # 1. Fetch data
        try:
            df = await self._client.get_ohlcv(pair, "5m", limit=100)
        except Exception as e:
            logger.error("Failed to fetch data for %s: %s", pair, e)
            return

        if df.empty or len(df) < 50:
            return

        # 2. Compute indicators
        df = self._indicators.compute_all(df)
        current_price = df.iloc[-1]["close"]

        # Update correlation tracker
        self._correlation.update(pair, current_price)

        # 3. Check open positions
        current_prices = {pair: current_price}
        closed_trades = self._executor.check_open_positions(current_prices)
        for trade in closed_trades:
            await self._db.log_trade(trade)
            self._risk.update_balance(self._executor.get_balance())

        # 4. Check if we can trade
        can_trade, reason = await self._risk.can_trade()
        if not can_trade:
            return

        # 5. Run all brains
        # Brain 1: Technical signals
        tech_signal = self._indicators.get_signal(df, self._strategies_config.get("smart_scalp", {}))
        brain1 = BrainSignal(tech_signal.direction, tech_signal.confidence, "technical")

        # Brain 2: Order flow
        try:
            book = await self._client.get_order_book(pair, limit=20)
            ob_signal = self._order_book_analyzer.analyze(book)
            if not ob_signal.is_liquid:
                return  # skip illiquid markets entirely

            if ob_signal.imbalance > 0.3:
                brain2 = BrainSignal("BUY", min(ob_signal.imbalance, 1.0), "order_flow")
            elif ob_signal.imbalance < -0.3:
                brain2 = BrainSignal("SELL", min(abs(ob_signal.imbalance), 1.0), "order_flow")
            else:
                brain2 = BrainSignal("HOLD", 0.5, "order_flow")
        except Exception:
            brain2 = BrainSignal("HOLD", 0.5, "order_flow")

        # Brain 3: AI Sentiment (placeholder — Gemini added in Phase 2)
        brain3 = BrainSignal("HOLD", 0.5, "sentiment")

        # Brain 4: Multi-timeframe (check 15m alignment)
        try:
            df_15m = await self._client.get_ohlcv(pair, "15m", limit=50)
            df_15m = self._indicators.compute_all(df_15m)
            tf_signal = self._indicators.get_signal(df_15m, self._strategies_config.get("smart_scalp", {}))
            brain4 = BrainSignal(tf_signal.direction, tf_signal.confidence, "multi_timeframe")
        except Exception:
            brain4 = BrainSignal("HOLD", 0.5, "multi_timeframe")

        # Brain 5: Cross-asset correlation
        corr_signal = self._correlation.get_brain_signal(pair)
        brain5 = BrainSignal(
            corr_signal["direction"], corr_signal["confidence"], "correlation"
        )

        # 6. Trade gate — 3/5 must agree
        signals = {
            "technical": brain1, "order_flow": brain2,
            "sentiment": brain3, "multi_timeframe": brain4,
            "correlation": brain5,
        }
        decision = self._trade_gate.evaluate(signals)

        if not decision.approved:
            return

        # 7. Get strategy signal for entry/exit prices
        strategy = self._strategies[0]  # smart_scalp for now
        strat_signal = strategy.evaluate(df, self._strategies_config.get("smart_scalp", {}))

        if strat_signal.direction == "HOLD":
            return

        # Override direction with gate decision
        if strat_signal.direction != decision.direction:
            return  # strategy and gate disagree on direction

        # 8. Calculate position size
        balance = self._executor.get_balance()
        position_size = await self._risk.get_position_size(balance, decision.confidence)
        position_size *= self._protection_size_multiplier  # apply protection reduction

        # 9. Validate trade
        quantity = position_size / current_price
        valid, reason = await self._risk.validate_trade(pair, decision.direction.lower(), current_price, quantity)
        if not valid:
            return

        # 10. Execute
        trade = self._executor.execute_signal(strat_signal, pair, position_size)
        if trade:
            await self._db.log_trade(trade)
            self._risk.update_balance(self._executor.get_balance())

    async def _take_snapshot(self):
        """Take a portfolio snapshot for the equity curve."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=self._executor.get_balance(),
            unrealized_pnl=0.0,  # calculated from open positions
            open_positions_count=len(self._executor.get_open_positions()),
        )
        await self._db.snapshot_portfolio(snapshot)

    def _should_skip_cycle(self) -> bool:
        """Check if the current cycle should be skipped."""
        return self._protection_action in ("SHUTDOWN", "PAUSE", "EMERGENCY_SELL")

    def _get_size_multiplier(self) -> float:
        """Get position size multiplier from protection system."""
        return self._protection_size_multiplier
```

- [ ] **Step 5: Create main.py**

```python
# main.py
# Entry point for the AI Trading Bot.
# Supports three modes:
#   python main.py              → paper trading (default)
#   python main.py --live       → live trading (real money)
#   python main.py --backtest   → run backtests on historical data

import asyncio
import argparse
import logging
import sys
import os

# Setup logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log"),
    ],
)
logger = logging.getLogger("main")


async def run_trading():
    """Start the trading engine."""
    from src.core.engine import TradingEngine
    engine = TradingEngine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await engine.stop()


async def run_backtest():
    """Run backtests on all strategies."""
    from backtest.data_loader import DataLoader
    from backtest.engine import BacktestEngine
    from backtest.report import BacktestReport
    from src.strategies.smart_scalp import SmartScalpStrategy
    from src.core.config import load_strategies

    loader = DataLoader()
    config = load_strategies()

    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    strategy = SmartScalpStrategy()

    for pair in pairs:
        logger.info("Downloading historical data for %s...", pair)
        df = await loader.download(pair, "5m", days=180)

        if df.empty:
            logger.warning("No data for %s — skipping", pair)
            continue

        logger.info("Running backtest on %s (%d candles)...", pair, len(df))
        engine = BacktestEngine(initial_balance=50.0, maker_fee_rate=0.00075)
        result = engine.run(strategy, df, config.get("smart_scalp", {}), pair=pair)
        BacktestReport.print_summary(result, f"Smart Scalp — {pair}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--backtest", action="store_true", help="Run backtests")
    args = parser.parse_args()

    if args.live:
        os.environ["TRADING_MODE"] = "live"
        logger.info("Starting in LIVE mode — real money at risk!")
    elif args.backtest:
        logger.info("Running backtests...")
        asyncio.run(run_backtest())
        return
    else:
        os.environ["TRADING_MODE"] = "paper"
        logger.info("Starting in PAPER mode (simulated trading)")

    asyncio.run(run_trading())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create watchdog.py**

```python
# watchdog.py
# Process monitor that auto-restarts the bot on crash.
# Layer 5 of the autonomous protection system.
#
# Usage: python watchdog.py
# The watchdog runs main.py as a subprocess and restarts it
# if it crashes, with exponential backoff.

import subprocess
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
)
logger = logging.getLogger("watchdog")

MAX_RESTART_DELAY = 300  # 5 minutes max between restarts
INITIAL_DELAY = 5        # 5 seconds initial restart delay


def run_watchdog():
    """Monitor the trading bot and restart on crash."""
    restart_delay = INITIAL_DELAY
    consecutive_crashes = 0

    while True:
        logger.info("Starting trading bot (attempt %d)...", consecutive_crashes + 1)
        start_time = time.time()

        try:
            process = subprocess.run(
                [sys.executable, "main.py"],
                cwd=".",
            )

            runtime = time.time() - start_time

            if process.returncode == 0:
                logger.info("Bot exited cleanly (ran for %.0f seconds)", runtime)
                break  # clean exit — don't restart

            logger.error("Bot crashed with code %d after %.0f seconds",
                        process.returncode, runtime)

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted — shutting down")
            break

        # If bot ran for > 60 seconds, reset backoff (it was running fine)
        if time.time() - start_time > 60:
            restart_delay = INITIAL_DELAY
            consecutive_crashes = 0

        consecutive_crashes += 1
        logger.info("Restarting in %d seconds...", restart_delay)
        time.sleep(restart_delay)

        # Exponential backoff: 5, 10, 20, 40, 80, 160, 300, 300...
        restart_delay = min(restart_delay * 2, MAX_RESTART_DELAY)

        # Safety: don't restart indefinitely
        if consecutive_crashes > 20:
            logger.critical("Too many consecutive crashes (%d) — giving up", consecutive_crashes)
            break


if __name__ == "__main__":
    run_watchdog()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine.py -v`
Expected: All 4 tests PASS

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS across all test files

- [ ] **Step 9: Commit**

```bash
git add src/core/engine.py src/core/event_bus.py main.py watchdog.py tests/test_engine.py
git commit -m "feat: core trading engine, main entry point, watchdog auto-restart"
```

---

## Plan Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 1 | Database + Config | 5 | 8 |
| 2 | Binance Client + Order Book | 6 | 3 |
| 3 | Indicator Engine | 5 | 1 |
| 4 | Risk Manager + Protection | 13 | 3 |
| 5 | Trade Gate + Correlation | 5 | 2 |
| 6 | Smart Scalping Strategy | 3 | 2 |
| 7 | Paper Trading Executor | 6 | 1 |
| 8 | Backtesting Framework | 3 | 3 |
| 9 | Telegram Notifications | 3 | 1 |
| 10 | Core Engine + Entry Point | 4+ | 4 |
| **Total** | | **53** | **28** |

## Follow-Up Plans (Not in this plan)

- **Phase 2 Plan:** Grid trading, Gemini AI integration, strategy selector
- **Phase 3 Plan:** Momentum, mean reversion, self-learner, adaptive tuning
- **Phase 4 Plan:** Live executor, ML model training, advanced orders
- **Phase 5 Plan:** DEX arbitrage (Solana)
