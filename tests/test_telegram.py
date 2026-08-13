# tests/test_telegram.py
# Tests for Telegram notification formatting.
# Only tests message formatting — does NOT test actual sending
# (that would require a real bot token).

from src.notifications.telegram import TelegramNotifier


class TestTelegramFormatting:

    def test_format_daily_summary(self):
        """Daily summary should include all key metrics."""
        msg = TelegramNotifier.format_daily_summary(
            date="2024-01-15",
            total_trades=12,
            win_rate=0.75,
            net_pnl=1.23,
            balance=51.23,
            drawdown=2.5,
        )
        assert "DAILY SUMMARY" in msg
        assert "2024-01-15" in msg
        assert "12" in msg
        assert "75.0%" in msg
        assert "51.23" in msg

    def test_format_protection_alert(self):
        """Protection alert should show layer and action."""
        msg = TelegramNotifier.format_protection_alert(
            layer="Session",
            action="PAUSE",
            reason="3 consecutive losses",
        )
        assert "PROTECTION" in msg
        assert "PAUSE" in msg
        assert "Session" in msg
        assert "3 consecutive losses" in msg

    def test_format_trade_message(self):
        """Trade message should show entry, SL, TP, and strategy."""
        msg = TelegramNotifier.format_trade_message(
            pair="BTC/USDT",
            side="buy",
            entry=50000.0,
            sl=49800.0,
            tp=50400.0,
            size=10.0,
            strategy="smart_scalp",
        )
        assert "BTC/USDT" in msg
        assert "BUY" in msg
        assert "50000.00" in msg
        assert "smart_scalp" in msg
