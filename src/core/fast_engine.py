# src/core/fast_engine.py
# Fast Trading Engine — event-driven, zero-latency execution.
#
# OLD ENGINE (polling):
#   sleep 30s → fetch data → compute → decide → execute → sleep 30s
#   Latency: 0-30 seconds. Misses most opportunities.
#
# NEW ENGINE (event-driven):
#   WebSocket pushes price → engine reacts instantly
#   Two trigger types:
#   1. Candle close → full signal evaluation (new indicators)
#   2. Price tick → check open positions (update trailing stops)
#
# The engine pre-computes everything it can and only runs the
# full pipeline when a new candle closes (every 5 minutes).
# Between candles, it monitors open positions with live prices.
#
# Execution latency: under 2 seconds from signal to order.

import asyncio
import logging
import time
from datetime import datetime, timezone

from src.core.config import load_settings, load_strategies, get_scaling_tier
from src.core.event_bus import EventBus
from src.data.feed import BinanceClient
from src.data.realtime_feed import RealtimeFeed
from src.data.indicators import IndicatorEngine
from src.data.order_book import OrderBookAnalyzer
from src.intelligence.correlation import CorrelationTracker
from src.intelligence.market_regime import MarketRegimeDetector
from src.intelligence.gemini_brain import GeminiBrain
from src.intelligence.multi_timeframe import MultiTimeframeBrain
from src.intelligence.performance_tracker import PerformanceTracker
from src.intelligence.strategy_selector import StrategySelector
from src.intelligence.macro_calendar import MacroCalendar
from src.intelligence.trade_filter import TradeFilter
from src.risk.manager import RiskManager
from src.risk.protection import ProtectionSystem
from src.ai.trade_gate import TradeGate, BrainSignal
from src.strategies.smart_scalp import SmartScalpStrategy
from src.strategies.grid import GridStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.execution.paper_trader import PaperTrader
from src.execution.trailing_stop import TrailingStopManager
from src.execution.smart_exit import SmartExitManager
from src.execution.adaptive_sizer import AdaptiveSizer
from src.execution.partial_exit import PartialExitManager
from src.intelligence.funding_rate import FundingRateSignal
from src.notifications.telegram import TelegramNotifier
from src.storage.database import Database

logger = logging.getLogger(__name__)


class FastEngine:
    """Event-driven trading engine with real-time WebSocket feeds.

    Instead of polling every 30 seconds, this engine:
    1. Streams live prices via WebSocket (100ms updates)
    2. Evaluates signals on candle close (every 5 minutes)
    3. Monitors open positions with live prices (instant SL/TP)
    4. Executes trades within 1-2 seconds of signal

    Falls back to polling mode if WebSocket connection fails.
    """

    def __init__(self, mode: str = "paper"):
        self._mode = mode
        self._running = False

        # Configuration
        self._settings = None
        self._strategies_config = None

        # Core infrastructure
        self._event_bus = EventBus()
        self._db = None
        self._client = None
        self._realtime_feed = None
        self._indicator_engine = IndicatorEngine()
        self._order_book = OrderBookAnalyzer()
        self._notifier = TelegramNotifier()

        # Intelligence layer
        self._regime_detector = MarketRegimeDetector()
        self._correlation = CorrelationTracker()
        self._gemini_brain = GeminiBrain()
        self._mtf_brain = MultiTimeframeBrain()
        self._performance_tracker = PerformanceTracker()
        self._macro_calendar = MacroCalendar()
        self._trade_filter = TradeFilter()
        self._funding_signal = FundingRateSignal()

        # All 4 strategies
        self._strategies = {
            "smart_scalp": SmartScalpStrategy(),
            "grid": GridStrategy(),
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # Strategy selector
        self._strategy_selector = StrategySelector(
            self._strategies, self._performance_tracker
        )

        # Risk + protection
        self._protection = None
        self._risk_manager = None
        self._trade_gate = TradeGate()

        # Execution
        self._trader = None
        self._trailing_stops = TrailingStopManager()
        self._smart_exit = SmartExitManager()
        self._adaptive_sizer = AdaptiveSizer()
        self._partial_exit = PartialExitManager()

        # Tracking
        self._cycle_count = 0
        self._last_daily_summary = None
        self._price_history: dict[str, list[float]] = {}

        # Cached data — pre-computed, updated on candle close
        self._cached_indicators: dict[str, object] = {}
        self._cached_regime: dict[str, object] = {}
        self._last_signal_eval: dict[str, float] = {}

        # Rate limiting — don't evaluate signals faster than once per 10s per pair
        self._min_eval_interval = 10.0

        # Lock to prevent concurrent signal evaluations
        self._eval_lock = asyncio.Lock()

    async def start(self):
        """Initialize all components and connect to exchange."""
        logger.info("Starting FastEngine in %s mode...", self._mode)

        # Load configuration
        self._settings = load_settings()
        self._strategies_config = load_strategies()
        risk_cfg = self._settings.get("risk", {})

        # Initialize database
        self._db = Database()
        await self._db.initialize()

        # Initialize exchange client
        is_paper = self._mode == "paper"
        self._client = BinanceClient(paper_mode=is_paper)
        await self._client.connect()

        # Initialize real-time feed
        self._realtime_feed = RealtimeFeed(self._client._exchange)

        # Initialize protection system
        self._protection = ProtectionSystem(
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
            daily_drawdown_limit=risk_cfg.get("max_daily_drawdown_pct", 5.0),
            weekly_drawdown_limit=risk_cfg.get("weekly_drawdown_reduce_pct", 10.0),
            monthly_drawdown_limit=risk_cfg.get("monthly_drawdown_emergency_pct", 20.0),
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
        self._event_bus.on("trade_opened", self._on_trade_opened)
        self._event_bus.on("trade_closed", self._on_trade_closed)
        self._event_bus.on("protection_triggered", self._on_protection)

        self._running = True
        logger.info(
            "FastEngine started. Balance: $%.2f | Mode: %s | "
            "Strategies: %s | Brains: 5 | Feed: WebSocket",
            balance, self._mode, list(self._strategies.keys()),
        )

    async def run(self):
        """Main event loop — WebSocket driven with REST fallback."""
        if not self._running:
            raise RuntimeError("Engine not started. Call start() first.")

        pairs = self._settings.get("pairs", ["BTC/USDT"])
        timeframes = ["5m", "15m", "1h"]

        # Register WebSocket callbacks
        self._realtime_feed.on_price_update(self._on_price_tick)
        self._realtime_feed.on_candle_close(self._on_candle_close)

        # Start WebSocket streams
        ws_task = asyncio.create_task(
            self._realtime_feed.start(pairs, timeframes),
            name="realtime_feed",
        )

        # Start fallback polling loop (catches anything WebSocket misses)
        fallback_task = asyncio.create_task(
            self._fallback_loop(pairs),
            name="fallback_loop",
        )

        # Start daily summary checker
        summary_task = asyncio.create_task(
            self._daily_summary_loop(),
            name="daily_summary",
        )

        logger.info("FastEngine running. WebSocket streams active for %s", pairs)

        try:
            # Wait until stopped
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            ws_task.cancel()
            fallback_task.cancel()
            summary_task.cancel()
            await asyncio.gather(ws_task, fallback_task, summary_task,
                               return_exceptions=True)

    async def _on_price_tick(self, pair: str, price: float, spread_pct: float):
        """Called on every price update (~100ms intervals).

        FAST PATH — only checks open positions, no signal evaluation.
        This is what makes SL/TP/trailing stops react instantly.
        """
        # Update open positions with live price
        current_prices = {pair: price}

        for pos in self._trader.get_open_positions():
            if pos.pair != pair:
                continue

            pos_id = f"{pos.pair}_{pos.entry_price}"

            # Update smart exit with live price
            exit_state = self._smart_exit.update(pos_id, price)
            if exit_state and exit_state.current_sl != pos.stop_loss:
                pos.stop_loss = exit_state.current_sl

            # Backup trailing stop
            new_stop = self._trailing_stops.update(pos_id, price)
            if new_stop and new_stop > pos.stop_loss:
                pos.stop_loss = new_stop

            # Extend TP in trailing phases
            if exit_state and exit_state.phase >= 3:
                original_tp_dist = abs(pos.take_profit - pos.entry_price)
                extended_tp = (pos.entry_price + original_tp_dist * 3
                              if pos.side == "buy"
                              else pos.entry_price - original_tp_dist * 3)
                pos.take_profit = extended_tp

        # Check partial exits with live price
        for pos in self._trader.get_open_positions():
            if pos.pair != pair:
                continue
            pos_id = f"{pos.pair}_{pos.entry_price}"
            partial = self._partial_exit.check(pos_id, price)
            if partial:
                logger.info("Partial exit: %s took %.0f%% at +%.2f%%",
                           pair, partial["level_pct"], partial["profit_pct"])

        # Check for closed positions with live price
        closed = self._trader.check_open_positions(current_prices)
        for trade in closed:
            pos_id = f"{trade.pair}_{trade.entry_price}"
            self._trailing_stops.remove(pos_id)
            self._smart_exit.remove(pos_id)
            self._partial_exit.remove(pos_id)
            await self._event_bus.emit("trade_closed", trade=trade)

        # Update correlation tracker
        self._correlation.update(pair, price)

        # Track price history
        if pair not in self._price_history:
            self._price_history[pair] = []
        self._price_history[pair].append(price)
        if len(self._price_history[pair]) > 100:
            self._price_history[pair] = self._price_history[pair][-100:]

    async def _on_candle_close(self, pair: str, timeframe: str, candle_data: dict):
        """Called when a candle closes — triggers full signal evaluation.

        SLOW PATH — runs the complete analysis pipeline.
        Only fires every 5 minutes (on 5m candle close).
        This is where trading decisions happen.
        """
        # Only evaluate on primary timeframe close
        if timeframe != "5m":
            return

        # Rate limit: don't re-evaluate within 10 seconds
        now = time.time()
        last_eval = self._last_signal_eval.get(pair, 0)
        if now - last_eval < self._min_eval_interval:
            return

        # Use lock to prevent concurrent evaluations
        if self._eval_lock.locked():
            return

        async with self._eval_lock:
            self._last_signal_eval[pair] = now
            self._cycle_count += 1
            logger.info("--- Candle close: %s (cycle %d) ---", pair, self._cycle_count)

            try:
                await self._evaluate_and_execute(pair)
            except Exception as e:
                logger.error("Signal evaluation error for %s: %s", pair, e)

    async def _evaluate_and_execute(self, pair: str):
        """Full signal evaluation pipeline — called on candle close."""
        # Step 1: Protection check
        balance = self._trader.get_balance()
        protection_status = self._protection.check(
            current_balance=balance,
            initial_balance=self._trader._balance,
        )

        if protection_status.action in ("SHUTDOWN", "PAUSE", "EMERGENCY_SELL"):
            logger.warning("Protection: %s — %s", protection_status.action,
                         protection_status.reason)
            return

        # Step 2: Macro calendar check
        macro_status = self._macro_calendar.check()
        if macro_status["should_pause"]:
            logger.warning("Macro pause: %s", macro_status["reason"])
            return

        # Step 3: Fetch candle data (REST — we need full history for indicators)
        df_5m = await self._client.get_historical_ohlcv(pair, "5m", limit=100)
        if df_5m.empty or len(df_5m) < 50:
            return

        # Compute indicators
        df_5m = self._indicator_engine.compute_all(df_5m)
        current_price = df_5m.iloc[-1]["close"]
        latest = df_5m.iloc[-1]

        # Cache indicators
        self._cached_indicators[pair] = df_5m

        # Step 4: Detect market regime
        regime = self._regime_detector.detect(df_5m)
        self._cached_regime[pair] = regime

        # Step 5: Select strategy
        strategy = self._strategy_selector.select(regime)
        if strategy is None:
            return

        strategy_config = self._strategies_config.get(strategy.name, {})

        # Step 6: Collect 5 brain signals
        brain_signals = {}

        # Brain 1: Strategy signal (pair passed for per-pair cooldown)
        signal = strategy.evaluate(df_5m, strategy_config, pair=pair)
        if signal.direction != "HOLD":
            brain_signals["strategy"] = BrainSignal(
                direction=signal.direction,
                confidence=signal.confidence,
                source=f"strategy:{strategy.name}",
            )

        # Brain 2: Order book (use cached spread from WebSocket)
        try:
            order_book_data = await self._client.get_order_book(pair)
            if order_book_data:
                ob_signal = self._order_book.analyze(order_book_data)
                if ob_signal.is_liquid:
                    ob_direction = "HOLD"
                    ob_confidence = 0.5
                    if ob_signal.imbalance > 0.3:
                        ob_direction = "BUY"
                        ob_confidence = 0.5 + ob_signal.imbalance * 0.3
                    elif ob_signal.imbalance < -0.3:
                        ob_direction = "SELL"
                        ob_confidence = 0.5 + abs(ob_signal.imbalance) * 0.3
                    if ob_signal.whale_detected:
                        if ob_signal.whale_side == "bid" and ob_direction == "BUY":
                            ob_confidence = min(ob_confidence + 0.15, 1.0)
                        elif ob_signal.whale_side == "ask" and ob_direction == "SELL":
                            ob_confidence = min(ob_confidence + 0.15, 1.0)
                    brain_signals["order_book"] = BrainSignal(
                        direction=ob_direction,
                        confidence=ob_confidence,
                        source="order_book",
                    )
        except Exception as e:
            logger.debug("Order book error: %s", e)

        # Brain 3: Gemini AI
        try:
            indicators_dict = {
                "rsi": round(float(latest.get("rsi", 0)), 2),
                "macd_histogram": round(float(latest.get("macd_histogram", 0)), 4),
                "bb_width": round(float(latest.get("bb_width", 0)), 4),
                "volume_ratio": round(float(latest.get("volume_ratio", 0)), 2),
                "ema_fast": round(float(latest.get("ema_fast", 0)), 2),
                "ema_slow": round(float(latest.get("ema_slow", 0)), 2),
                "atr": round(float(latest.get("atr", 0)), 2),
            }
            gemini_result = await self._gemini_brain.analyze(
                pair, current_price, indicators_dict,
                regime.regime, self._price_history.get(pair, []),
            )
            if gemini_result["direction"] != "HOLD":
                brain_signals["gemini_ai"] = BrainSignal(
                    direction=gemini_result["direction"],
                    confidence=gemini_result["confidence"],
                    source="gemini_ai",
                )
        except Exception as e:
            logger.debug("Gemini error: %s", e)

        # Brain 4: Multi-timeframe
        try:
            tf_data = {"5m": df_5m}
            df_15m = await self._client.get_historical_ohlcv(pair, "15m", limit=100)
            if not df_15m.empty and len(df_15m) >= 30:
                tf_data["15m"] = df_15m
            df_1h = await self._client.get_historical_ohlcv(pair, "1h", limit=100)
            if not df_1h.empty and len(df_1h) >= 30:
                tf_data["1h"] = df_1h

            mtf_result = self._mtf_brain.analyze(tf_data)
            if mtf_result["direction"] != "HOLD":
                brain_signals["multi_timeframe"] = BrainSignal(
                    direction=mtf_result["direction"],
                    confidence=mtf_result["confidence"],
                    source="multi_timeframe",
                )
        except Exception as e:
            logger.debug("Multi-timeframe error: %s", e)

        # Brain 5: Correlation
        corr_signal = self._correlation.get_brain_signal(pair)
        if corr_signal and corr_signal["direction"] != "HOLD":
            brain_signals["correlation"] = BrainSignal(
                direction=corr_signal["direction"],
                confidence=corr_signal.get("confidence", 0.5),
                source="correlation",
            )

        # Brain 6 (bonus): Funding rate — contrarian signal from derivatives
        try:
            funding_rate = await self._funding_signal.fetch_funding_rate(
                self._client._exchange, pair
            )
            if funding_rate is not None:
                fr_result = self._funding_signal.analyze(pair, funding_rate)
                if fr_result["direction"] != "HOLD":
                    brain_signals["funding_rate"] = BrainSignal(
                        direction=fr_result["direction"],
                        confidence=fr_result["confidence"],
                        source="funding_rate",
                    )
        except Exception as e:
            logger.debug("Funding rate error: %s", e)

        # Step 7: Trade Gate — 4/5 consensus (funding rate is bonus, not required)
        if len(brain_signals) < 2:
            return

        gate_decision = self._trade_gate.evaluate(brain_signals)
        if not gate_decision.approved:
            return

        # Step 8: Trade Quality Filter
        spread_pct = self._realtime_feed.get_spread_pct(pair)
        filter_result = self._trade_filter.check(
            confidence=gate_decision.confidence,
            spread_pct=spread_pct,
            open_positions=self._trader.get_open_positions(),
            pair=pair,
        )
        if not filter_result["pass"]:
            logger.info("Filter blocked %s: %s", pair, filter_result["reason"])
            return

        # Step 9: Risk validation
        if not self._risk_manager.can_trade(self._trader.get_balance()):
            return

        position_size = self._risk_manager.get_position_size(
            balance=self._trader.get_balance(),
            win_rate=self._performance_tracker.get_win_rate(strategy.name),
            avg_win=0.008,
            avg_loss=0.004,
        )

        # Apply adaptive sizing
        position_size = self._adaptive_sizer.adjust_size(position_size)

        # Reduce if protection says so
        if protection_status.action == "REDUCE_SIZE":
            position_size *= 0.5
            position_size = max(position_size, 10.0)

        # Step 10: Execute
        trade = self._trader.execute_signal(signal, pair, position_size)

        if trade:
            pos_id = f"{pair}_{signal.entry_price}"
            self._smart_exit.register(
                pos_id, pair, signal.direction.lower(),
                signal.entry_price, signal.stop_loss, signal.take_profit,
            )
            self._trailing_stops.register(
                pos_id, pair, signal.direction.lower(),
                signal.entry_price, signal.stop_loss,
            )
            self._partial_exit.register(
                pos_id, pair, signal.direction.lower(),
                signal.entry_price, position_size,
            )

            await self._event_bus.emit(
                "trade_opened",
                pair=pair, side=signal.direction,
                entry=signal.entry_price,
                sl=signal.stop_loss, tp=signal.take_profit,
                size=position_size, strategy=strategy.name,
            )

            logger.info(
                "TRADE: %s %s via %s | $%.2f @ $%.2f | "
                "Gate: %d brains (%.0f%%) | Regime: %s | Latency: REALTIME",
                signal.direction, pair, strategy.name,
                position_size, signal.entry_price,
                gate_decision.agreeing_brains,
                gate_decision.confidence * 100, regime.regime,
            )

    async def _fallback_loop(self, pairs: list[str]):
        """Fallback polling loop — runs every 60s as safety net.

        If WebSocket disconnects or misses a candle close,
        this loop catches it. Not the primary execution path.
        """
        while self._running:
            try:
                await asyncio.sleep(60)

                for pair in pairs:
                    # Only run if WebSocket data is stale
                    if self._realtime_feed.is_stale(pair, max_age=30.0):
                        logger.warning("WebSocket stale for %s, running fallback",
                                      pair)
                        await self._evaluate_and_execute(pair)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Fallback loop error: %s", e)

    async def _daily_summary_loop(self):
        """Send daily summary at midnight UTC."""
        while self._running:
            try:
                await asyncio.sleep(60)  # check every minute
                now = datetime.now(timezone.utc)
                today = now.strftime("%Y-%m-%d")

                if self._last_daily_summary == today:
                    continue

                if now.hour == 0 and now.minute < 5:
                    self._last_daily_summary = today
                    balance = self._trader.get_balance()
                    rankings = self._performance_tracker.get_rankings()

                    # Get upcoming macro events for the summary
                    upcoming = self._macro_calendar.get_upcoming(days=3)
                    if upcoming:
                        logger.info("Upcoming macro events: %s", upcoming)

                    await self._notifier.send_daily_summary(
                        date=today,
                        total_trades=self._cycle_count,
                        win_rate=0.0,
                        net_pnl=0.0,
                        balance=balance,
                        drawdown=0.0,
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _on_trade_opened(self, **kwargs):
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
        trade = kwargs.get("trade")
        if trade:
            await self._db.save_trade(trade)
            self._performance_tracker.record_trade(trade.strategy, trade.pnl)
            # Feed protection system — tracks consecutive losses + daily P&L
            self._protection.record_trade_result(trade.pnl)
            if trade.pnl > 0:
                self._adaptive_sizer.record_win()
            else:
                self._adaptive_sizer.record_loss()
                self._trade_filter.record_loss()

    async def _on_protection(self, **kwargs):
        await self._notifier.send_protection_alert(
            layer=kwargs.get("layer", ""),
            action=kwargs.get("action", ""),
            reason=kwargs.get("reason", ""),
        )

    async def _get_balance(self) -> float:
        if self._mode == "paper":
            return self._settings.get("initial_balance", 50.0)
        balances = await self._client.get_balance()
        return balances.get("USDT", 0.0)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping FastEngine...")
        self._running = False
        if self._realtime_feed:
            await self._realtime_feed.stop()
        if self._client:
            await self._client.disconnect()
        if self._db:
            await self._db.close()
        logger.info("FastEngine stopped.")
