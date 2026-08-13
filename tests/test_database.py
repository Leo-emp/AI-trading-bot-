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
