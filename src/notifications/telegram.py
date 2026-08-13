# src/notifications/telegram.py
# Telegram bot integration for trade alerts and daily summaries.
# Sends notifications to a private chat so you can monitor the bot
# from your phone without touching the terminal.
#
# Features:
# - Trade execution alerts (entry/exit with P&L)
# - Daily performance summaries
# - Protection system alerts (when risk layers activate)
# - Error notifications (so you know if something breaks)
#
# Uses lazy loading — doesn't import telegram until first send.
# This way, if you don't set TELEGRAM_BOT_TOKEN, nothing breaks.

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trading notifications via Telegram bot.

    Lazy-loads the telegram library on first use.
    If TELEGRAM_BOT_TOKEN is not set, all sends silently skip.
    """

    def __init__(self, bot_token: Optional[str] = None,
                 chat_id: Optional[str] = None):
        # Pull from env if not provided
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._bot = None  # lazy loaded

    def _get_bot(self):
        """Lazy-load the telegram Bot instance."""
        if self._bot is None and self._token:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
            except ImportError:
                logger.warning("python-telegram-bot not installed, notifications disabled")
            except Exception as e:
                logger.error("Failed to create Telegram bot: %s", e)
        return self._bot

    async def send_message(self, text: str):
        """Send a message to the configured chat."""
        if not self._token or not self._chat_id:
            logger.debug("Telegram not configured, skipping notification")
            return

        bot = self._get_bot()
        if bot is None:
            return

        try:
            await bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("Telegram send failed: %s", e)

    # --- Formatted message builders ---

    @staticmethod
    def format_trade_message(pair: str, side: str, entry: float,
                              sl: float, tp: float, size: float,
                              strategy: str) -> str:
        """Format a trade execution notification."""
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        return (
            f"{emoji} <b>TRADE: {side.upper()} {pair}</b>\n"
            f"Entry: ${entry:.2f}\n"
            f"SL: ${sl:.2f} | TP: ${tp:.2f}\n"
            f"Size: ${size:.2f}\n"
            f"Strategy: {strategy}"
        )

    @staticmethod
    def format_close_message(pair: str, side: str, entry: float,
                              exit_price: float, pnl: float,
                              fees: float, reason: str) -> str:
        """Format a position close notification."""
        emoji = "✅" if pnl > 0 else "❌"
        return (
            f"{emoji} <b>CLOSED: {pair}</b>\n"
            f"Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
            f"P&L: ${pnl:.4f} (fees: ${fees:.4f})\n"
            f"Reason: {reason}"
        )

    @staticmethod
    def format_daily_summary(date: str, total_trades: int,
                              win_rate: float, net_pnl: float,
                              balance: float, drawdown: float) -> str:
        """Format the end-of-day performance summary."""
        emoji = "📈" if net_pnl >= 0 else "📉"
        return (
            f"{emoji} <b>DAILY SUMMARY — {date}</b>\n"
            f"Trades: {total_trades}\n"
            f"Win rate: {win_rate*100:.1f}%\n"
            f"Net P&L: ${net_pnl:.4f}\n"
            f"Balance: ${balance:.2f}\n"
            f"Max drawdown: {drawdown:.1f}%"
        )

    @staticmethod
    def format_protection_alert(layer: str, action: str,
                                 reason: str) -> str:
        """Format a protection system alert."""
        return (
            f"🛡️ <b>PROTECTION: {action.upper()}</b>\n"
            f"Layer: {layer}\n"
            f"Reason: {reason}"
        )

    @staticmethod
    def format_error(error_type: str, message: str) -> str:
        """Format an error notification."""
        return (
            f"⚠️ <b>ERROR: {error_type}</b>\n"
            f"{message}"
        )

    # --- Convenience send methods ---

    async def send_trade(self, pair: str, side: str, entry: float,
                          sl: float, tp: float, size: float,
                          strategy: str):
        """Send trade execution alert."""
        msg = self.format_trade_message(pair, side, entry, sl, tp, size, strategy)
        await self.send_message(msg)

    async def send_daily_summary(self, date: str, total_trades: int,
                                  win_rate: float, net_pnl: float,
                                  balance: float, drawdown: float):
        """Send end-of-day summary."""
        msg = self.format_daily_summary(date, total_trades, win_rate, net_pnl, balance, drawdown)
        await self.send_message(msg)

    async def send_protection_alert(self, layer: str, action: str,
                                     reason: str):
        """Send protection system alert."""
        msg = self.format_protection_alert(layer, action, reason)
        await self.send_message(msg)

    async def send_error(self, error_type: str, message: str):
        """Send error notification."""
        msg = self.format_error(error_type, message)
        await self.send_message(msg)
