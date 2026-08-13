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
        """If win rate x avg_win < loss rate x avg_loss, return minimum."""
        sizer = PositionSizer(min_order_size=10.0)
        size = sizer.kelly_size(
            win_rate=0.3, avg_win=0.50, avg_loss=0.50, balance=50.0
        )
        # Negative edge -> floor at minimum order size
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
        mock_db.get_daily_trade_count = AsyncMock(return_value=50)
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
        """3 consecutive losses -> reduce position size by 50%."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=3)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.0)
        assert action.action == "REDUCE_SIZE"

    @pytest.mark.asyncio
    async def test_layer2_pause_after_5_losses(self, mock_db, settings):
        """5 consecutive losses -> pause trading."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=5)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.0)
        assert action.action == "PAUSE"
        assert action.duration_minutes == 30

    @pytest.mark.asyncio
    async def test_layer2_defense_mode_on_drawdown(self, mock_db, settings):
        """Daily drawdown > 3% -> defense-only mode."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        # -3.5% daily P&L on $50 = -$1.75
        action = await ps.check_all_layers(balance=50.0, daily_pnl=-1.75)
        assert action.action == "DEFENSE_ONLY"

    @pytest.mark.asyncio
    async def test_layer2_shutdown_on_5pct_drawdown(self, mock_db, settings):
        """Daily drawdown > 5% -> full shutdown."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=-2.75)
        assert action.action == "SHUTDOWN"

    @pytest.mark.asyncio
    async def test_layer4_shutdown_below_floor(self, mock_db, settings):
        """Balance below $10 -> permanent shutdown."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=8.0, daily_pnl=0.0)
        assert action.action == "SHUTDOWN"

    @pytest.mark.asyncio
    async def test_continue_when_all_clear(self, mock_db, settings):
        """No issues -> CONTINUE normally."""
        mock_db.get_consecutive_losses = AsyncMock(return_value=0)
        mock_db.get_weekly_drawdown = AsyncMock(return_value=2.0)
        mock_db.get_monthly_drawdown = AsyncMock(return_value=5.0)
        ps = ProtectionSystem(mock_db, settings)
        action = await ps.check_all_layers(balance=50.0, daily_pnl=0.5)
        assert action.action == "CONTINUE"
