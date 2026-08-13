# src/core/engine.py
# The main trading engine — orchestrates everything.
# This is the brain that ties all components together:
#
# 1. Check protection system → can we trade?
# 2. Fetch latest market data via WebSocket
# 3. Compute indicators on fresh candles
# 4. Ask each "brain" (strategy + correlation + order book) for signals
# 5. Run 3/5 consensus through the Trade Gate
# 6. Validate with Risk Manager (position sizing, SL/TP)
# 7. Execute via PaperTrader or LiveTrader
# 8. Emit events for notifications/logging
#
# The engine runs in a continuous loop with configurable intervals.
# Protection system can pause, reduce, or shut down trading at any time.

import asyncio
import logging
import time
from datetime import datetime, timezone

from src.core.config import load_settings, load_strategies, get_scaling_tier
from src.core.event_bus import EventBus
from src.data.feed import BinanceClient, PriceFeed
from src.data.indicators import IndicatorEngine
from src.data.order_book import OrderBookAnalyzer
from src.intelligence.correlation import CorrelationTracker
from src.risk.manager import RiskManager
from src.risk.protection import ProtectionSystem
from src.ai.trade_gate import TradeGate, BrainSignal
from src.strategies.smart_scalp import SmartScalpStrategy
from src.execution.paper_trader import PaperTrader
from src.notifications.telegram import TelegramNotifier
from src.storage.database import Database

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main orchestrator — connects all trading components.

    Lifecycle:
    1. start() — initialize all components
    2. run() — main trading loop
    3. stop() — graceful shutdown
    """

    def __init__(self, mode: str = "paper"):
        # Trading mode: "paper" or "live"
        self._mode = mode
        self._running = False

        # Will be initialized in start()
        self._settings = None
        self._strategies_config = None
        self._event_bus = EventBus()
        self._db = None
        self._client = None
        self._price_feed = None
        self._indicator_engine = IndicatorEngine()
        self._order_book = OrderBookAnalyzer()
        self._correlation = CorrelationTracker()
        self._protection = None
        self._risk_manager = None
        self._trade_gate = TradeGate()
        self._trader = None
        self._notifier = TelegramNotifier()

        # Strategies
        self._strategies = {
            "smart_scalp": SmartScalpStrategy(),
        }

        # Tracking
        self._cycle_count = 0
        self._last_daily_summary = None

    async def start(self):
        """Initialize all components and connect to exchange."""
        logger.info("Starting TradingEngine in %s mode...", self._mode)

        # Load configuration
        self._settings = load_settings()
        self._strategies_config = load_strategies()

        # Get risk parameters based on balance tier
        risk_cfg = self._settings.get("risk", {})

        # Initialize database
        self._db = Database()
        await self._db.initialize()

        # Initialize exchange client
        is_paper = self._mode == "paper"
        self._client = BinanceClient(paper_mode=is_paper)
        await self._client.connect()

        # Initialize protection system
        self._protection = ProtectionSystem(
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 3),
            daily_drawdown_limit=risk_cfg.get("daily_drawdown_limit", 5.0),
            weekly_drawdown_limit=risk_cfg.get("weekly_drawdown_limit", 10.0),
            monthly_drawdown_limit=risk_cfg.get("monthly_drawdown_limit", 20.0),
            balance_floor_pct=risk_cfg.get("balance_floor_pct", 50.0),
        )

        # Initialize risk manager
        self._risk_manager = RiskManager(self._settings)

        # Initialize paper trader
        balance = await self._get_balance()
        tier = get_scaling_tier(balance, self._settings)
        self._trader = PaperTrader(
            initial_balance=balance,
            maker_fee_rate=self._settings["fees"]["maker_rate"],
            min_order_size=10.0,
            max_positions=tier.get("max_positions", 3),
        )

        # Register event handlers
        self._register_events()

        self._running = True
        logger.info("TradingEngine started. Balance: $%.2f, Tier: %s", balance, tier.get("name", "micro"))

    def _register_events(self):
        """Wire up event handlers for notifications."""
        self._event_bus.on("trade_opened", self._on_trade_opened)
        self._event_bus.on("trade_closed", self._on_trade_closed)
        self._event_bus.on("protection_triggered", self._on_protection)

    async def _on_trade_opened(self, **kwargs):
        """Handle trade opened event — send Telegram notification."""
        await self._notifier.send_trade(
            pair=kwargs.get("pair", ""),
            side=kwargs.get("side", ""),
            entry=kwargs.get("entry", 0),
            sl=kwargs.get("sl", 0),
            tp=kwargs.get("tp", 0),
            size=kwargs.get("size", 0),
            strategy=kwargs.get("strategy", ""),
        )

    async def _on_trade_closed(self, **kwargs):
        """Handle trade closed event."""
        trade = kwargs.get("trade")
        if trade:
            await self._db.save_trade(trade)

    async def _on_protection(self, **kwargs):
        """Handle protection trigger — alert via Telegram."""
        await self._notifier.send_protection_alert(
            layer=kwargs.get("layer", ""),
            action=kwargs.get("action", ""),
            reason=kwargs.get("reason", ""),
        )

    async def run(self):
        """Main trading loop. Runs until stop() is called."""
        if not self._running:
            raise RuntimeError("Engine not started. Call start() first.")

        pairs = self._settings.get("pairs", ["BTC/USDT"])
        interval = self._settings.get("loop_interval_seconds", 60)

        logger.info("Entering main trading loop. Pairs: %s, Interval: %ds", pairs, interval)

        while self._running:
            try:
                self._cycle_count += 1
                logger.info("--- Cycle %d ---", self._cycle_count)

                # Step 1: Check protection system
                balance = self._trader.get_balance()
                protection_status = self._protection.check(
                    current_balance=balance,
                    initial_balance=self._trader._balance,
                )

                if protection_status.action == "SHUTDOWN":
                    logger.critical("Protection SHUTDOWN triggered: %s", protection_status.reason)
                    await self._event_bus.emit("protection_triggered",
                                               layer="system", action="SHUTDOWN",
                                               reason=protection_status.reason)
                    break

                if protection_status.action == "PAUSE":
                    logger.warning("Trading PAUSED: %s", protection_status.reason)
                    await self._event_bus.emit("protection_triggered",
                                               layer="session", action="PAUSE",
                                               reason=protection_status.reason)
                    await asyncio.sleep(interval)
                    continue

                # Step 2: Process each pair
                for pair in pairs:
                    await self._process_pair(pair, protection_status)

                # Step 3: Check for daily summary
                await self._check_daily_summary()

                # Wait for next cycle
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error("Error in trading cycle %d: %s", self._cycle_count, e)
                await self._notifier.send_error("CycleError", str(e))
                await asyncio.sleep(interval)

    async def _process_pair(self, pair: str, protection_status):
        """Process a single trading pair — the core trading logic."""
        try:
            # Fetch latest candles
            df = await self._client.get_historical_ohlcv(pair, "5m", limit=100)
            if df.empty or len(df) < 50:
                logger.warning("Not enough data for %s, skipping", pair)
                return

            # Compute indicators
            df = self._indicator_engine.compute_all(df)
            current_price = df.iloc[-1]["close"]

            # Check open positions for SL/TP
            closed = self._trader.check_open_positions({pair: current_price})
            for trade in closed:
                await self._event_bus.emit("trade_closed", trade=trade)

            # Collect brain signals from multiple sources
            brain_signals = []

            # Brain 1: Smart Scalp strategy
            scalp_config = self._strategies_config.get("smart_scalp", {})
            signal = self._strategies["smart_scalp"].evaluate(df, scalp_config)
            if signal.direction != "HOLD":
                brain_signals.append(BrainSignal(
                    brain_name="smart_scalp",
                    direction=signal.direction,
                    confidence=signal.confidence,
                    reasoning=f"RSI/MACD/Volume/EMA confirmation",
                ))

            # Brain 2: Raw indicator signal
            indicator_signal = self._indicator_engine.get_signal(df)
            if indicator_signal["direction"] != "HOLD":
                brain_signals.append(BrainSignal(
                    brain_name="indicators",
                    direction=indicator_signal["direction"],
                    confidence=indicator_signal["strength"],
                    reasoning=indicator_signal["reason"],
                ))

            # Brain 3: Correlation analysis
            self._correlation.update(pair, current_price)
            corr_signal = self._correlation.get_signal(pair)
            if corr_signal and corr_signal.get("direction") != "HOLD":
                brain_signals.append(BrainSignal(
                    brain_name="correlation",
                    direction=corr_signal["direction"],
                    confidence=corr_signal.get("confidence", 0.5),
                    reasoning=corr_signal.get("reason", "correlation signal"),
                ))

            # Brain 4: Order book analysis (needs live order book data)
            # Brain 5: AI/Gemini analysis (Phase 2 — ML after 30 days)
            # For now, these are placeholder signals that vote HOLD

            # If not enough brain signals, skip
            if len(brain_signals) < 2:
                logger.debug("Not enough brain signals for %s (%d), skipping", pair, len(brain_signals))
                return

            # Run through Trade Gate (3/5 consensus)
            gate_decision = self._trade_gate.evaluate(brain_signals)

            if not gate_decision.approved:
                logger.debug("Trade gate rejected for %s: %s", pair, gate_decision.reason)
                return

            # Risk validation
            position_size = self._risk_manager.get_position_size(
                balance=self._trader.get_balance(),
                win_rate=0.55,  # default until we have enough data
                avg_win=0.008,  # 0.8% avg win
                avg_loss=0.004,  # 0.4% avg loss
            )

            if not self._risk_manager.can_trade(self._trader.get_balance()):
                logger.info("Risk manager says no trade for %s", pair)
                return

            # Adjust position size if protection says reduce
            if protection_status.action == "REDUCE_SIZE":
                position_size *= 0.5
                position_size = max(position_size, 10.0)  # still respect $10 floor

            # Execute the trade
            trade_signal = signal  # use the strategy signal for entry/SL/TP
            trade = self._trader.execute_signal(trade_signal, pair, position_size)

            if trade:
                await self._event_bus.emit("trade_opened",
                                           pair=pair,
                                           side=trade_signal.direction,
                                           entry=trade_signal.entry_price,
                                           sl=trade_signal.stop_loss,
                                           tp=trade_signal.take_profit,
                                           size=position_size,
                                           strategy=trade_signal.strategy_name)

        except Exception as e:
            logger.error("Error processing pair %s: %s", pair, e)

    async def _check_daily_summary(self):
        """Send daily summary at end of trading day (00:00 UTC)."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if self._last_daily_summary == today:
            return

        # Only send after midnight UTC
        if now.hour == 0 and now.minute < 5:
            self._last_daily_summary = today
            balance = self._trader.get_balance()
            await self._notifier.send_daily_summary(
                date=today,
                total_trades=self._cycle_count,
                win_rate=0.0,  # TODO: calculate from DB
                net_pnl=0.0,  # TODO: calculate from DB
                balance=balance,
                drawdown=0.0,  # TODO: calculate from DB
            )

    async def _get_balance(self) -> float:
        """Get current balance from exchange or config."""
        if self._mode == "paper":
            return self._settings.get("initial_balance", 50.0)
        # Live mode would fetch from exchange
        return await self._client.get_balance("USDT")

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping TradingEngine...")
        self._running = False

        if self._client:
            await self._client.disconnect()

        if self._db:
            await self._db.close()

        logger.info("TradingEngine stopped.")
