# src/core/engine.py
# The main trading engine — the brain that orchestrates EVERYTHING.
#
# Flow per cycle:
# 1. Protection check → can we trade at all?
# 2. Detect market regime → trending, ranging, or volatile?
# 3. Select strategy → pick the best one for this regime
# 4. Fetch data → candles from Binance for all timeframes
# 5. Compute indicators → RSI, MACD, BB, EMA, ATR, volume
# 6. Collect 5 brain signals:
#    Brain 1: Strategy signal (smart_scalp / grid / momentum / mean_reversion)
#    Brain 2: Order book analysis (imbalance, whales, liquidity)
#    Brain 3: Gemini AI analysis (market conditions, sentiment)
#    Brain 4: Multi-timeframe alignment (5m + 15m + 1h agree?)
#    Brain 5: Cross-asset correlation (BTC trend affects alts)
# 7. Trade Gate → 60% of voting brains must agree, veto on strong opposing
# 8. Risk validation → position sizing, SL/TP, balance check
# 9. Adaptive sizing → increase on win streak, decrease on loss streak
# 10. Execute → PaperTrader or live
# 11. Smart exit → 4-phase exit (normal → breakeven → trail → tight trail)
# 12. Track performance → auto-disable losing strategies
#
# The engine runs in a continuous loop until stopped.

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
from src.strategies.real_grid import GridCalculator
from src.execution.grid_executor import GridExecutor
from src.intelligence.adaptive_params import AdaptiveParams
from src.intelligence.trade_journal import TradeJournal
from src.notifications.telegram import TelegramNotifier
from src.storage.database import Database
from src.ai.embeddings import EmbeddingEngine, MarketSnapshot
from src.ai.rag_memory import RAGMemory
from src.ai.ml_model import MLModel
from src.ai.ml_features import FeatureEngineer
from src.ai.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main orchestrator — connects all trading components.

    Lifecycle:
    1. start() — initialize all components
    2. run() — main trading loop
    3. stop() — graceful shutdown
    """

    def __init__(self, mode: str = "paper"):
        self._mode = mode
        self._running = False

        # Configuration (loaded in start())
        self._settings = None
        self._strategies_config = None

        # Core infrastructure
        self._event_bus = EventBus()
        self._db = None
        self._client = None
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

        # All 4 strategies
        self._strategies = {
            "smart_scalp": SmartScalpStrategy(),
            "grid": GridStrategy(),
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

        # Strategy selector (initialized after performance tracker)
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

        # Adaptive intelligence — ATR-based dynamic SL/TP/sizing
        self._adaptive_params = AdaptiveParams()
        self._trade_journal = TradeJournal()

        # Real grid engine — separate from directional strategies
        self._grid_calculator = None   # initialized in start() from config
        self._grid_executor = None     # initialized in start() from config
        self._max_concurrent_grids = 3

        # Phase 4: ML pipeline
        self._ml_model = MLModel()
        self._feature_engineer = FeatureEngineer()

        # Phase 5: RAG memory + embeddings
        self._embedding_engine = EmbeddingEngine()
        self._rag_memory = RAGMemory()

        # Phase 6: Agentic AI orchestrator
        self._agent_orchestrator = AgentOrchestrator()

        # Tracking
        self._cycle_count = 0
        self._last_daily_summary = None
        self._price_history: dict[str, list[float]] = {}
        # P0.1: Last known prices per pair — used for equity calculation
        # so protection system sees true account value, not just cash
        self._last_prices: dict[str, float] = {}

    async def start(self):
        """Initialize all components and connect to exchange."""
        logger.info("Starting TradingEngine in %s mode...", self._mode)

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

        # Initialize protection system — TIGHTENED defaults
        # Old: 5 losses / 8% daily / 50% floor
        # New: 4 losses / 3% daily / 70% floor
        self._protection = ProtectionSystem(
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 4),
            daily_drawdown_limit=risk_cfg.get("max_daily_drawdown_pct", 3.0),
            weekly_drawdown_limit=risk_cfg.get("weekly_drawdown_reduce_pct", 7.0),
            monthly_drawdown_limit=risk_cfg.get("monthly_drawdown_emergency_pct", 12.0),
            balance_floor_pct=risk_cfg.get("balance_floor_pct", 70.0),
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
            time_exit_hours=self._settings.get("risk", {}).get("time_exit_hours", 1),
        )

        # Initialize real grid engine from config
        grid_cfg = self._strategies_config.get("real_grid", {})
        if grid_cfg.get("enabled", False):
            fee_rate = self._settings["fees"]["maker_rate"]
            self._grid_calculator = GridCalculator(
                num_levels=grid_cfg.get("num_levels", 10),
                max_grid_exposure_pct=grid_cfg.get("max_exposure_pct", 35.0),
                max_bb_width_pct=grid_cfg.get("max_bb_width_pct", 5.5),
                exit_atr_multiplier=grid_cfg.get("exit_atr_multiplier", 1.0),
                fee_rate=fee_rate,
                min_profit_per_fill=grid_cfg.get("min_profit_per_fill", 20.0),
            )
            self._grid_executor = GridExecutor(
                fee_rate=fee_rate,
                max_loss_pct=grid_cfg.get("max_grid_loss_pct", 40.0),
            )
            self._max_concurrent_grids = grid_cfg.get("max_concurrent_grids", 3)
            logger.info("Real grid engine initialized: max %d levels (dynamic), %.1f%% exposure, $%.0f min profit/fill, max %d concurrent",
                       grid_cfg.get("num_levels", 10), grid_cfg.get("max_exposure_pct", 35.0),
                       grid_cfg.get("min_profit_per_fill", 20.0), self._max_concurrent_grids)

        # Phase 5: Initialize RAG memory (starts collecting from day 1)
        self._rag_memory.initialize()

        # Phase 4: Initialize ML model (loads saved model if exists)
        self._ml_model.initialize()

        # Phase 6: Initialize agent orchestrator
        self._agent_orchestrator.initialize()

        # Register event handlers
        self._register_events()

        # Re-register restored positions with smart exit and trailing stop
        for pos in self._trader.get_open_positions():
            pos_id = f"{pos.pair}_{pos.entry_price}"
            self._smart_exit.register(
                pos_id, pos.pair, pos.side,
                pos.entry_price, pos.stop_loss, pos.take_profit,
            )
            self._trailing_stops.register(
                pos_id, pos.pair, pos.side,
                pos.entry_price, pos.stop_loss,
            )

        self._running = True
        logger.info(
            "TradingEngine started. Balance: $%.2f | Mode: %s | "
            "Strategies: %s | Brains: 5 | Restored positions: %d",
            self._trader.get_balance(), self._mode,
            list(self._strategies.keys()),
            len(self._trader.get_open_positions()),
        )

    def _register_events(self):
        """Wire up event handlers for notifications and tracking."""
        self._event_bus.on("trade_opened", self._on_trade_opened)
        self._event_bus.on("trade_closed", self._on_trade_closed)
        self._event_bus.on("protection_triggered", self._on_protection)

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
            # Record in performance tracker
            self._performance_tracker.record_trade(trade.strategy, trade.pnl)
            # Feed protection system — tracks consecutive losses + daily P&L
            self._protection.record_trade_result(trade.pnl)
            # Feed adaptive sizer — bigger bets when winning, smaller when losing
            if trade.pnl > 0:
                self._adaptive_sizer.record_win()
            else:
                self._adaptive_sizer.record_loss()
                # Tell trade filter to start cooldown (avoid revenge trades)
                self._trade_filter.record_loss()

            # Phase 5: Store trade outcome in RAG memory
            self._store_trade_memory(trade, kwargs.get("market_snapshot"))

            # Phase 4: Record ML model outcome for drift detection
            if self._ml_model.is_trained:
                self._ml_model.record_outcome(trade.pnl > 0)

            # Phase 6: Update agent accuracy tracking
            self._agent_orchestrator.record_outcomes(trade.pnl > 0)

    async def _on_protection(self, **kwargs):
        await self._notifier.send_protection_alert(
            layer=kwargs.get("layer", ""),
            action=kwargs.get("action", ""),
            reason=kwargs.get("reason", ""),
        )

    async def _manage_open_positions(self):
        """P0.1: Manage ALL open positions independently of the signal pipeline.

        This runs EVERY cycle, even when:
        - Protection is PAUSED (cash looks low because capital is in positions)
        - Macro calendar is pausing new trades
        - Binance candle fetch fails for new signals
        - No new strategy signals exist

        Position exits are RISK MANAGEMENT, not trade-entry logic.
        They must never be blocked by the new-trade pipeline.
        """
        positions = self._trader.get_open_positions()
        if not positions:
            return

        # Get unique pairs that have open positions
        pairs_with_positions = sorted({pos.pair for pos in positions})

        for pair in pairs_with_positions:
            try:
                # Fetch fresh 1-minute candle for accurate high/low
                df = await self._client.get_historical_ohlcv(pair, "1m", limit=2)
                if df.empty:
                    logger.warning("EXIT CHECK: no price data for %s, skipping", pair)
                    continue

                latest = df.iloc[-1]
                current_price = float(latest["close"])
                candle_high = float(latest["high"])
                candle_low = float(latest["low"])

                # Cache price for equity calculation
                self._last_prices[pair] = current_price

                # P0.7: Diagnostic logging for every position on this pair
                pair_positions = [p for p in positions if p.pair == pair]
                for pos in pair_positions:
                    pos_id = f"{pos.pair}_{pos.entry_price}"
                    age_hours = (time.time() - pos.opened_at) / 3600

                    # Calculate unrealized P&L for this position
                    if pos.side == "buy":
                        unrealized = (current_price - pos.entry_price) * pos.quantity
                    else:
                        unrealized = (pos.entry_price - current_price) * pos.quantity

                    logger.info(
                        "EXIT CHECK | %s %s | entry=%.2f current=%.2f | "
                        "low=%.2f high=%.2f | SL=%.2f TP=%.2f | "
                        "age=%.1fh | unrealizedPnL=$%.2f",
                        pos.side.upper(), pos.pair,
                        pos.entry_price, current_price,
                        candle_low, candle_high,
                        pos.stop_loss, pos.take_profit,
                        age_hours, unrealized,
                    )

                    # Update smart exit (4-phase: normal -> breakeven -> trail -> tight)
                    exit_state = self._smart_exit.update(pos_id, current_price)
                    if exit_state:
                        if exit_state.current_sl != pos.stop_loss:
                            old_sl = pos.stop_loss
                            pos.stop_loss = exit_state.current_sl
                            logger.info(
                                "SMART EXIT | %s %s | phase=%d | SL moved %.2f -> %.2f",
                                pos.side.upper(), pos.pair, exit_state.phase,
                                old_sl, pos.stop_loss,
                            )

                        # P0.5: Fix short trailing stop — use min() for sells, max() for buys
                        new_stop = self._trailing_stops.update(pos_id, current_price)
                        if new_stop is not None:
                            if pos.side == "buy" and new_stop > pos.stop_loss:
                                pos.stop_loss = new_stop
                            elif pos.side == "sell" and new_stop < pos.stop_loss:
                                pos.stop_loss = new_stop

                        # Once trailing is active, remove TP ceiling entirely.
                        # The trailing stop IS the exit — a fixed TP caps winners.
                        if exit_state.phase >= 3:
                            original_tp_dist = abs(exit_state.original_tp - exit_state.entry_price)
                            extended_tp = (pos.entry_price + original_tp_dist * 10
                                          if pos.side == "buy"
                                          else pos.entry_price - original_tp_dist * 10)
                            pos.take_profit = extended_tp

                    # Update trade journal MFE/MAE
                    self._trade_journal.update_price(pos_id, current_price)

                # P0.4: Pass candle high/low so SL/TP checks catch intra-candle wicks
                current_prices = {pair: current_price}
                highs = {pair: candle_high}
                lows = {pair: candle_low}
                closed = self._trader.check_open_positions(current_prices, highs=highs, lows=lows)

                for trade in closed:
                    pos_id = f"{trade.pair}_{trade.entry_price}"
                    close_reason = getattr(trade, "close_reason", "unknown")
                    trade_status = getattr(trade, "status", "closed")

                    # P0.6: Only remove exit managers for FULLY closed trades
                    # Partial closes must keep their smart exit and trailing stop active
                    if trade_status == "closed":
                        self._trailing_stops.remove(pos_id)
                        self._smart_exit.remove(pos_id)

                        # Close in trade journal and feed adaptive learning
                        journal_record = self._trade_journal.close_trade(
                            pos_id, trade.exit_price, trade.pnl, close_reason,
                        )
                        if journal_record:
                            self._adaptive_params.record_outcome(journal_record)

                        # PAIR COOLDOWN: register stop losses for escalating cooldown
                        # After stop: 30min, 2nd: 60min, 3rd+: 120min
                        # Only reset on fully profitable close, not partials
                        if close_reason == "stop_loss" and trade.pnl < 0:
                            self._risk_manager.register_stop_loss(trade.pair)
                        elif trade.pnl > 0:
                            self._risk_manager.register_profitable_close(trade.pair)

                        logger.info(
                            "TRADE CLOSED | %s %s | reason=%s | pnl=$%.4f | balance=$%.2f",
                            trade.side.upper(), trade.pair, close_reason,
                            trade.pnl, self._trader.get_balance(),
                        )
                    else:
                        logger.info(
                            "PARTIAL CLOSE | %s %s | reason=%s | pnl=$%.4f",
                            trade.side.upper(), trade.pair, close_reason, trade.pnl,
                        )

                    await self._event_bus.emit("trade_closed", trade=trade)

            except Exception as e:
                logger.error("Position management error for %s: %s", pair, e)

        # Manage active grid positions (separate from directional)
        await self._manage_grid_positions()

    async def _manage_grid_positions(self):
        """Update all active grid positions with current prices.

        Grids run independently from directional trades.
        Each cycle: check for fills, check for range breakout.
        """
        if not self._grid_executor or not self._grid_executor.active_pairs:
            return

        for pair in list(self._grid_executor.active_pairs):
            try:
                df = await self._client.get_historical_ohlcv(pair, "1m", limit=2)
                if df.empty:
                    continue

                latest = df.iloc[-1]
                current_price = float(latest["close"])
                candle_high = float(latest["high"])
                candle_low = float(latest["low"])
                self._last_prices[pair] = current_price

                # Update grid — execute any buy/sell fills
                fills = self._grid_executor.update(pair, current_price, candle_high, candle_low)
                for fill in fills:
                    logger.info(
                        "GRID FILL | %s | buy=$%.2f sell=$%.2f | net=$%.4f | total_fills=%d",
                        pair, fill.buy_price, fill.sell_price, fill.net_pnl,
                        self._grid_executor.total_fills,
                    )

                # Check if price has broken out of the range (trend starting)
                grid = self._grid_executor.get_grid(pair)
                if grid:
                    # Use 1h ATR for breakout detection
                    df_1h = await self._client.get_historical_ohlcv(pair, "1h", limit=20)
                    atr = current_price * 0.01  # fallback 1%
                    if not df_1h.empty and len(df_1h) >= 15:
                        high = df_1h["high"].values
                        low = df_1h["low"].values
                        close = df_1h["close"].values
                        trs = []
                        for i in range(1, len(high)):
                            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                            trs.append(tr)
                        if len(trs) >= 14:
                            atr = sum(trs[-14:]) / 14

                    # Check all exit conditions: ATR breakout, max loss, fully loaded,
                    # staleness (BB drift), idle timeout
                    should_close, reason = self._grid_calculator.should_close_grid(current_price, grid, atr)

                    if not should_close:
                        force_close, force_reason = self._grid_executor.should_force_close(pair, current_price)
                        if force_close:
                            should_close = True
                            reason = f"MAX LOSS: {force_reason}"

                    if not should_close and self._grid_executor.is_fully_loaded(pair):
                        if current_price <= grid.lower_bound:
                            should_close = True
                            reason = "fully loaded + price at/below lower bound"

                    # Check if grid is stale (BB has drifted away from grid range)
                    if not should_close:
                        try:
                            df_4h = await self._client.get_historical_ohlcv(pair, "4h", limit=50)
                            if not df_4h.empty and len(df_4h) >= 25:
                                df_4h = self._indicator_engine.compute_all(df_4h)
                                latest_4h = df_4h.iloc[-1]
                                new_bb_upper = float(latest_4h.get("bb_upper", 0))
                                new_bb_lower = float(latest_4h.get("bb_lower", 0))
                                if new_bb_upper > 0 and new_bb_lower > 0:
                                    stale, stale_reason = self._grid_calculator.is_grid_stale(
                                        grid, new_bb_lower, new_bb_upper,
                                    )
                                    if stale:
                                        should_close = True
                                        reason = f"STALE: {stale_reason}"
                        except Exception as e:
                            logger.debug("Staleness check failed for %s: %s", pair, e)

                    # Check idle timeout (no fills after 8 hours)
                    if not should_close:
                        idle, idle_reason = self._grid_calculator.is_grid_idle(grid)
                        if idle:
                            should_close = True
                            reason = f"IDLE: {idle_reason}"

                    if should_close:
                        balance_return, summary = self._grid_executor.deactivate_grid(pair, current_price)
                        self._trader._balance += balance_return
                        logger.info("GRID DEACTIVATED | %s | %s | returned $%.2f to paper balance",
                                   pair, reason, balance_return)
                        await self._notifier.send_trade(
                            pair=pair, side="GRID_CLOSE", entry=0,
                            sl=0, tp=0, size=balance_return,
                            strategy=f"real_grid: {summary}",
                        )
                    else:
                        status = self._grid_executor.get_status(pair)
                        unrealized = self._grid_executor.get_unrealized_pnl(pair, current_price)
                        loss_pct = self._grid_executor.get_loss_pct(pair, current_price)
                        logger.info(
                            "GRID STATUS | %s | holding=%d/%d | fills=%d | "
                            "realized=$%.4f | unrealized=$%.4f | loss=%.1f%% | cash=$%.2f",
                            pair, status["holding"], status["levels"],
                            status["fills"], status["pnl"], unrealized,
                            loss_pct, status["grid_cash"],
                        )

            except Exception as e:
                logger.error("Grid management error for %s: %s", pair, e)

    async def run(self):
        """Main trading loop. Runs until stop() is called."""
        if not self._running:
            raise RuntimeError("Engine not started. Call start() first.")

        pairs = self._settings.get("pairs", ["BTC/USDT"])
        interval = self._settings.get("loop_interval_seconds", 60)

        logger.info("Entering main loop. Pairs: %s, Interval: %ds", pairs, interval)

        while self._running:
            try:
                self._cycle_count += 1
                logger.info("--- Cycle %d ---", self._cycle_count)

                # P0.1: ALWAYS manage open positions FIRST — before protection.
                # Position exits are risk management and must never be blocked
                # by protection/macro gates. A PAUSE should stop NEW trades,
                # not prevent existing positions from hitting SL/TP/time exit.
                await self._manage_open_positions()

                # P0.2: Use EQUITY (cash + position value) not just cash balance.
                # Opening 4 × $20K positions drops cash to $20K, but equity is
                # still ~$100K. Without this, protection sees a fake 80% drawdown
                # and triggers PAUSE, which blocks position management (now fixed
                # above), but also blocks new trades unnecessarily.
                equity = self._trader.get_equity(self._last_prices)
                # Grid balance was deducted from trader._balance on activation.
                # Add back: grid cash (unspent) + exposure (filled positions) + unrealized P&L.
                if self._grid_executor:
                    equity += self._grid_executor.total_grid_cash
                    equity += self._grid_executor.get_open_exposure()
                    for gp in self._grid_executor.active_pairs:
                        equity += self._grid_executor.get_unrealized_pnl(
                            gp, self._last_prices.get(gp, 0),
                        )
                cash = self._trader.get_balance()
                logger.info(
                    "ACCOUNT | cash=$%.2f | equity=$%.2f | positions=%d | "
                    "grids=%d | unrealizedPnL=$%.2f | gridPnL=$%.2f",
                    cash, equity,
                    len(self._trader.get_open_positions()),
                    len(self._grid_executor.active_pairs) if self._grid_executor else 0,
                    self._trader.get_unrealized_pnl(self._last_prices),
                    self._grid_executor.total_pnl if self._grid_executor else 0,
                )

                # Step 1: Check protection system — using EQUITY not cash
                protection_status = self._protection.check(
                    current_balance=equity,
                    initial_balance=self._trader.get_initial_balance(),
                )

                if protection_status.action == "SHUTDOWN":
                    logger.critical("SHUTDOWN: %s", protection_status.reason)
                    await self._event_bus.emit("protection_triggered",
                                               layer="system", action="SHUTDOWN",
                                               reason=protection_status.reason)
                    break

                if protection_status.action in ("PAUSE", "EMERGENCY_SELL"):
                    logger.warning("PAUSED (new trades only): %s", protection_status.reason)
                    await self._event_bus.emit("protection_triggered",
                                               layer="session", action=protection_status.action,
                                               reason=protection_status.reason)
                    await asyncio.sleep(interval)
                    continue

                # Step 1b: Check macro event calendar (Fed, CPI, NFP)
                macro_status = self._macro_calendar.check()
                if macro_status["should_pause"]:
                    logger.warning("MACRO PAUSE (new trades only): %s", macro_status["reason"])
                    await self._event_bus.emit("protection_triggered",
                                               layer="macro", action="PAUSE",
                                               reason=macro_status["reason"])
                    await asyncio.sleep(interval)
                    continue

                # Step 2: Process each pair (signal generation + new trade entry only)
                for pair in pairs:
                    await self._process_pair(pair, protection_status)

                # Step 3: Daily summary check
                await self._check_daily_summary()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error("Cycle %d error: %s", self._cycle_count, e)
                await self._notifier.send_error("CycleError", str(e))
                await asyncio.sleep(interval)

    async def _process_pair(self, pair: str, protection_status):
        """Process a single trading pair — the FULL intelligence pipeline."""
        try:
            # --- Step 1: Fetch candle data for primary timeframe ---
            df_5m = await self._client.get_historical_ohlcv(pair, "5m", limit=100)
            if df_5m.empty or len(df_5m) < 50:
                logger.warning("Not enough 5m data for %s", pair)
                return

            # Check candle freshness — reject stale data
            import time as _time
            latest_ts = df_5m.iloc[-1]["timestamp"]
            if hasattr(latest_ts, "timestamp"):
                candle_age = _time.time() - latest_ts.timestamp()
                if candle_age > 600:  # older than 10 minutes
                    logger.warning("Stale candle data for %s (%.0fs old), skipping",
                                  pair, candle_age)
                    return

            # Compute indicators on primary timeframe
            df_5m = self._indicator_engine.compute_all(df_5m)
            current_price = df_5m.iloc[-1]["close"]
            latest = df_5m.iloc[-1]

            # Track price history for Gemini brain
            if pair not in self._price_history:
                self._price_history[pair] = []
            self._price_history[pair].append(current_price)
            if len(self._price_history[pair]) > 100:
                self._price_history[pair] = self._price_history[pair][-100:]

            # Update correlation tracker
            self._correlation.update(pair, current_price)

            # Position management now handled by _manage_open_positions()
            # which runs EVERY cycle before protection checks.
            # Cache current price for other uses in this method.
            self._last_prices[pair] = current_price

            # --- Step 3: Detect market regime ---
            regime = self._regime_detector.detect(df_5m)
            logger.info("Regime: %s (%.0f%% confidence, %s volatility)",
                       regime.regime, regime.confidence * 100, regime.volatility)

            # --- Step 3b: GRID MODE — activate/skip if RANGING ---
            # Real grid engine handles ranging markets separately from
            # the directional 5-brain pipeline. If grid activates,
            # skip the entire directional flow for this pair.
            if self._grid_calculator and self._grid_executor:
                grid_handled = await self._try_grid_mode(
                    pair, regime, df_5m, current_price,
                )
                if grid_handled:
                    return

            # --- Step 4: Select best strategy for this regime ---
            strategy = self._strategy_selector.select(regime)
            if strategy is None:
                logger.info("No strategy available for %s, skipping", pair)
                return

            strategy_config = self._strategies_config.get(strategy.name, {})

            # --- Step 5: Collect all 5 brain signals ---
            brain_signals = {}

            # Brain 1: Selected strategy signal (pair passed for per-pair cooldown)
            signal = strategy.evaluate(df_5m, strategy_config, pair=pair)
            if signal.direction != "HOLD":
                brain_signals["strategy"] = BrainSignal(
                    direction=signal.direction,
                    confidence=signal.confidence,
                    source=f"strategy:{strategy.name}",
                )

            # Brain 2: Order book analysis
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
                        # Whale detection boosts confidence
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
                logger.debug("Order book fetch failed for %s: %s", pair, e)

            # Brain 3: Gemini AI analysis (now with RAG memory — Phase 5)
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

                # Phase 5: Build RAG context from similar historical trades
                market_snapshot = MarketSnapshot(
                    pair=pair, price=current_price,
                    rsi=indicators_dict["rsi"],
                    macd_histogram=indicators_dict["macd_histogram"],
                    bb_width=indicators_dict["bb_width"],
                    volume_ratio=indicators_dict["volume_ratio"],
                    ema_fast=indicators_dict["ema_fast"],
                    ema_slow=indicators_dict["ema_slow"],
                    atr=indicators_dict["atr"],
                    regime=regime.regime,
                    regime_confidence=regime.confidence,
                )
                rag_context = self._get_rag_context(pair, market_snapshot)

                gemini_result = await self._gemini_brain.analyze(
                    pair, current_price, indicators_dict,
                    regime.regime, self._price_history.get(pair, []),
                    rag_context=rag_context,
                )
                if gemini_result["direction"] != "HOLD":
                    brain_signals["gemini_ai"] = BrainSignal(
                        direction=gemini_result["direction"],
                        confidence=gemini_result["confidence"],
                        source="gemini_ai",
                    )
            except Exception as e:
                logger.info("Gemini brain skipped for %s: %s", pair, e)

            # Brain 4: Multi-timeframe alignment
            tf_data = {"5m": df_5m}  # init outside try so filter can use it
            try:
                # Fetch 15m and 1h candles
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

            # Brain 5: Cross-asset correlation
            corr_signal = self._correlation.get_brain_signal(pair)
            if corr_signal and corr_signal["direction"] != "HOLD":
                brain_signals["correlation"] = BrainSignal(
                    direction=corr_signal["direction"],
                    confidence=corr_signal.get("confidence", 0.5),
                    source="correlation",
                )

            # --- Step 6: Trade Gate — percentage-based consensus ---
            # Log brain signal summary for diagnostics
            brain_summary = {name: f"{s.direction}({s.confidence:.2f})"
                            for name, s in brain_signals.items()}
            logger.info("Brains for %s: %s | Strategy: %s",
                       pair, brain_summary, strategy.name)

            gate_decision = self._trade_gate.evaluate(brain_signals)

            if not gate_decision.approved:
                logger.info("Gate rejected %s: %s", pair, gate_decision.reasons[0]
                           if gate_decision.reasons else "no reason")
                return

            # Require at least one PRIMARY brain (strategy or gemini_ai)
            # Order book + multi_timeframe alone are support signals, not entries
            primary_brains = {"strategy", "gemini_ai"}
            has_primary = any(name in brain_signals for name in primary_brains)
            if not has_primary:
                logger.info("Skipping %s: no primary brain (strategy/gemini) voted", pair)
                return

            # --- Step 6b: Trade Quality Filter ---
            # Gate says direction is correct, filter checks conditions are right

            # Feed 1h trend to filter (PROVEN: trading with trend wins 55-65%)
            try:
                if "1h" in tf_data:
                    df_1h_local = tf_data["1h"]
                    if len(df_1h_local) >= 20:
                        ema_fast_1h = df_1h_local["close"].ewm(span=8).mean().iloc[-1]
                        ema_slow_1h = df_1h_local["close"].ewm(span=21).mean().iloc[-1]
                        close_1h = df_1h_local["close"].iloc[-1]
                        if close_1h > ema_fast_1h > ema_slow_1h:
                            self._trade_filter.set_hourly_trend(pair, "UP")
                        elif close_1h < ema_fast_1h < ema_slow_1h:
                            self._trade_filter.set_hourly_trend(pair, "DOWN")
                        else:
                            self._trade_filter.set_hourly_trend(pair, "NEUTRAL")
            except Exception:
                pass

            # Feed ATR ratio to filter (PROVEN: dead markets = random walks)
            try:
                atr_col = df_5m.get("atr")
                if atr_col is not None and len(atr_col.dropna()) >= 20:
                    current_atr = atr_col.iloc[-1]
                    avg_atr = atr_col.iloc[-20:].mean()
                    if avg_atr > 0:
                        self._trade_filter.set_atr_ratio(pair, current_atr / avg_atr)
            except Exception:
                pass

            spread_pct = 0.0
            try:
                order_book_data = await self._client.get_order_book(pair)
                if order_book_data and order_book_data.get("bids") and order_book_data.get("asks"):
                    best_bid = float(order_book_data["bids"][0][0])
                    best_ask = float(order_book_data["asks"][0][0])
                    spread_pct = self._trade_filter.get_spread_pct(best_bid, best_ask)
            except Exception:
                pass

            filter_result = self._trade_filter.check(
                confidence=gate_decision.confidence,
                spread_pct=spread_pct,
                open_positions=self._trader.get_open_positions(),
                pair=pair,
                trade_direction=gate_decision.direction,
            )
            if not filter_result["pass"]:
                logger.info("Trade filter blocked %s: %s", pair, filter_result["reason"])
                return

            # --- Step 7: Risk validation ---
            if not self._risk_manager.can_trade(self._trader.get_balance()):
                logger.info("Risk manager blocked trade for %s", pair)
                return

            # PAIR COOLDOWN: block re-entry after recent stop losses
            # Escalating: 30m → 60m → 120m after consecutive stops on same pair
            # Prevents churn like POL getting stopped 3 times in 45 minutes
            if self._risk_manager.pair_cooldown_active(pair):
                return

            # --- Step 8: ADAPTIVE INTELLIGENCE — ATR-based SL/TP/sizing ---
            # Use 1-HOUR ATR (not 5m) because trades hold up to 12 hours.
            # 5m ATR is tiny noise ($0.30 for BNB) → everything hits minimum floor.
            # 1h ATR captures actual volatility range we're trading within.
            atr_value = 0
            if "1h" in tf_data:
                try:
                    h = tf_data["1h"]
                    if len(h) >= 15:
                        # Compute ATR directly from 1h OHLCV
                        high = h["high"].values
                        low = h["low"].values
                        close_arr = h["close"].values
                        trs = []
                        for i in range(1, len(high)):
                            tr = max(
                                high[i] - low[i],
                                abs(high[i] - close_arr[i - 1]),
                                abs(low[i] - close_arr[i - 1]),
                            )
                            trs.append(tr)
                        if len(trs) >= 14:
                            atr_value = sum(trs[-14:]) / 14
                except Exception:
                    pass
            if atr_value <= 0:
                # Fallback: 5m ATR exists but is too small for SL/TP
                atr_5m = float(latest.get("atr", 0))
                # Scale up: 1h ≈ 4-5× the 5m ATR (volatility scaling)
                atr_value = atr_5m * 5 if atr_5m > 0 else current_price * 0.008

            # P0.3: TradeGate is AUTHORITATIVE on direction.
            # The old code let the strategy signal override the gate's consensus,
            # which meant the gate could say SELL but execution would use BUY.
            direction = gate_decision.direction
            if signal.direction != direction:
                logger.warning(
                    "DIRECTION MISMATCH | %s | strategy=%s gate=%s | using gate",
                    pair, signal.direction, direction,
                )

            # Get volume ratio for momentum multiplier
            vol_ratio = float(latest.get("volume_ratio", 1.0))

            dynamic = self._adaptive_params.calculate(
                entry_price=current_price,
                direction=direction,
                atr=atr_value,
                balance=self._trader.get_balance(),
                regime=regime.regime,
                strategy=strategy.name,
                confidence=gate_decision.confidence,
                volume_ratio=vol_ratio,
            )

            # MINIMUM R:R GATE — only enter trades with asymmetric upside
            # At 1.5:1 minimum with 5-brain consensus, even 35% WR is profitable:
            # 35 × 1.5R - 65 × 1R = 52.5R - 65R = -12.5R... but trailing stop
            # extends winners well beyond initial TP, so actual avg win >> 1.5R
            if dynamic.risk_reward < 1.5:
                logger.info(
                    "R:R REJECTED %s: %.1f:1 < 1.5:1 minimum (SL=%.2f%% TP=%.2f%%)",
                    pair, dynamic.risk_reward,
                    dynamic.sl_distance_pct, dynamic.tp_distance_pct,
                )
                return

            # Use adaptive position size instead of Kelly
            position_size = dynamic.position_size

            # Apply adaptive sizing multiplier (win/loss streaks)
            position_size = self._adaptive_sizer.adjust_size(position_size)

            # Reduce size if protection says so
            if protection_status.action == "REDUCE_SIZE":
                position_size *= 0.5
                position_size = max(position_size, 10.0)

            # PORTFOLIO EXPOSURE CHECK: max 50% margin, max 5% total initial risk
            if not self._risk_manager.check_portfolio_exposure(
                balance=self._trader.get_balance(),
                open_positions=self._trader.get_open_positions(),
                new_position_size=position_size,
                new_sl_pct=dynamic.sl_distance_pct,
                leverage=dynamic.leverage,
            ):
                logger.info("Portfolio exposure limit blocked %s", pair)
                return

            # Build the executable signal with ATR-based SL/TP
            from src.strategies.base import StrategySignal as SigType
            signal = SigType(
                direction=direction,
                confidence=gate_decision.confidence,
                entry_price=current_price,
                stop_loss=dynamic.stop_loss,
                take_profit=dynamic.take_profit,
                strategy_name=strategy.name,
                reasons=gate_decision.reasons,
            )

            # --- Step 9: Execute the trade (with futures leverage) ---
            trade = self._trader.execute_signal(signal, pair, position_size,
                                                leverage=dynamic.leverage)

            if trade:
                pos_id = f"{pair}_{signal.entry_price}"

                # Register with smart exit using R-based thresholds
                # Thresholds are multiples of initial SL distance (1R),
                # not raw ATR. This fixes the 0.29R breakeven bug.
                exit_thresholds = self._adaptive_params.get_smart_exit_thresholds(
                    atr=atr_value, entry_price=current_price,
                    regime=regime.regime, strategy=strategy.name,
                )
                self._smart_exit.register(
                    pos_id, pair, signal.direction.lower(),
                    signal.entry_price, signal.stop_loss, signal.take_profit,
                    breakeven_pct=exit_thresholds["breakeven_pct"],
                    trail_activate_pct=exit_thresholds["trail_activate_pct"],
                    tight_activate_pct=exit_thresholds["tight_activate_pct"],
                    trail_distance_pct=exit_thresholds["trail_distance_pct"],
                    tight_trail_pct=exit_thresholds["tight_trail_pct"],
                )
                self._trailing_stops.register(
                    pos_id, pair, signal.direction.lower(),
                    signal.entry_price, signal.stop_loss,
                )

                # Register with trade journal for MFE/MAE tracking
                self._trade_journal.open_trade(
                    pos_id=pos_id, pair=pair,
                    side=signal.direction.lower(),
                    strategy=strategy.name, regime=regime.regime,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=position_size,
                    confidence=gate_decision.confidence,
                    atr=atr_value,
                )

                await self._event_bus.emit(
                    "trade_opened",
                    pair=pair, side=signal.direction,
                    entry=signal.entry_price,
                    sl=signal.stop_loss, tp=signal.take_profit,
                    size=position_size, strategy=strategy.name,
                )

                logger.info(
                    "TRADE EXECUTED: %s %s via %s | Entry: $%.2f | "
                    "SL: $%.2f (%.2f%%) | TP: $%.2f (%.2f%%) | "
                    "Notional: $%.0f | Margin: $%.0f | Leverage: %dx | "
                    "R:R=%.1f:1 | Gate: %d brains (%.0f%% conf) | Regime: %s",
                    signal.direction, pair, strategy.name,
                    signal.entry_price,
                    signal.stop_loss, dynamic.sl_distance_pct,
                    signal.take_profit, dynamic.tp_distance_pct,
                    position_size, dynamic.margin_required, dynamic.leverage,
                    dynamic.risk_reward,
                    gate_decision.agreeing_brains,
                    gate_decision.confidence * 100, regime.regime,
                )

        except Exception as e:
            logger.error("Error processing %s: %s", pair, e)

    async def _try_grid_mode(self, pair: str, regime, df_5m, current_price: float) -> bool:
        """Try to activate or manage grid mode for a ranging pair.

        Returns True if grid is handling this pair (skip directional pipeline).
        Returns False if grid can't activate (fall through to directional).
        """
        grid_active = pair in self._grid_executor.active_pairs

        # If regime changed AWAY from RANGING and grid is active → deactivate
        if grid_active and regime.regime != "RANGING":
            balance_return, summary = self._grid_executor.deactivate_grid(pair, current_price)
            self._trader._balance += balance_return
            logger.info("GRID → TREND switch | %s | regime=%s | %s | returned $%.2f",
                       pair, regime.regime, summary, balance_return)
            return False  # let directional pipeline take over

        # If grid is already active for this pair, it's being managed
        # by _manage_grid_positions() — skip directional pipeline
        if grid_active:
            return True

        # Not RANGING → no grid activation
        if regime.regime != "RANGING":
            return False

        # Cap concurrent grids to prevent over-allocation
        active_count = len(self._grid_executor.active_pairs)
        if active_count >= self._max_concurrent_grids:
            return False

        # Use 4-HOUR BB bands for grid range.
        # 1h BB(20) = only 20 hours of data → ~$100-300 BTC range → unprofitable.
        # 4h BB(20) = 80 hours (3.3 days) → ~$1000-3000 range → profitable spacing.
        try:
            df_4h = await self._client.get_historical_ohlcv(pair, "4h", limit=50)
            if df_4h.empty or len(df_4h) < 25:
                return False
            df_4h = self._indicator_engine.compute_all(df_4h)
            latest_4h = df_4h.iloc[-1]
            bb_upper = float(latest_4h.get("bb_upper", 0))
            bb_lower = float(latest_4h.get("bb_lower", 0))
            bb_width = float(latest_4h.get("bb_width", 0))
        except Exception as e:
            logger.debug("Failed to fetch 4h BB for grid %s: %s", pair, e)
            return False

        if bb_upper <= 0 or bb_lower <= 0:
            return False

        # Log BB values for diagnosis
        bb_range = bb_upper - bb_lower
        bb_mid = (bb_upper + bb_lower) / 2
        bb_range_pct = (bb_range / bb_mid * 100) if bb_mid > 0 else 0
        logger.info(
            "GRID 4h BB | %s | $%.2f-$%.2f (range $%.2f = %.2f%%) | width=%.4f",
            pair, bb_lower, bb_upper, bb_range, bb_range_pct, bb_width,
        )

        can_activate, reason = self._grid_calculator.can_activate(bb_width, regime.regime)
        if not can_activate:
            logger.debug("Grid rejected for %s: %s", pair, reason)
            return False

        # Calculate grid levels from 4-hour Bollinger Bands
        grid_state = self._grid_calculator.calculate_levels(
            bb_lower=bb_lower,
            bb_upper=bb_upper,
            current_price=current_price,
            balance=self._trader.get_balance(),
        )
        if grid_state is None:
            return False

        # Activate the grid — reserves balance from paper trader
        ok, msg = self._grid_executor.activate_grid(pair, grid_state, self._trader.get_balance())
        if not ok:
            logger.info("Grid activation failed for %s: %s", pair, msg)
            return False

        # Deduct reserved balance from paper trader
        self._trader._balance -= grid_state.reserved_balance

        # Log grid activation details
        spacing_pct = self._grid_calculator.get_grid_spacing_pct(grid_state)
        profit_per_fill = self._grid_calculator.get_profit_per_fill(grid_state)
        logger.info(
            "GRID ACTIVATED | %s | range $%.2f-$%.2f | %d levels | "
            "spacing=%.3f%% | est_profit/fill=$%.4f | reserved=$%.2f",
            pair, grid_state.lower_bound, grid_state.upper_bound,
            len(grid_state.levels), spacing_pct, profit_per_fill,
            grid_state.reserved_balance,
        )
        await self._notifier.send_trade(
            pair=pair, side="GRID_OPEN", entry=current_price,
            sl=grid_state.lower_bound, tp=grid_state.upper_bound,
            size=grid_state.reserved_balance, strategy="real_grid",
        )

        return True  # grid is handling this pair

    async def _check_daily_summary(self):
        """Send daily summary at midnight UTC."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        if self._last_daily_summary == today:
            return

        if now.hour == 0 and now.minute < 5:
            self._last_daily_summary = today
            balance = self._trader.get_balance()

            # Get strategy rankings for the summary
            rankings = self._performance_tracker.get_rankings()
            logger.info("Daily strategy rankings: %s", rankings)

            await self._notifier.send_daily_summary(
                date=today,
                total_trades=self._cycle_count,
                win_rate=0.0,
                net_pnl=0.0,
                balance=balance,
                drawdown=0.0,
            )

    def _store_trade_memory(self, trade, market_snapshot=None):
        """Phase 5: Store a completed trade in RAG memory with full context."""
        if not self._rag_memory.is_available:
            return

        try:
            # Build a market snapshot for embedding
            snapshot = MarketSnapshot(
                pair=trade.pair,
                price=trade.entry_price,
                regime=getattr(trade, "_regime", "RANGING"),
            )

            # Generate the embedding vector
            embedding = self._embedding_engine.embed(snapshot)

            # Build metadata
            pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity) * 100
                       if trade.quantity > 0 else 0)
            metadata = {
                "pair": trade.pair,
                "strategy": trade.strategy,
                "side": trade.side,
                "outcome": "WIN" if trade.pnl > 0 else "LOSS",
                "pnl_pct": round(pnl_pct, 4),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
            }

            # Human-readable summary for Gemini context
            summary = (
                f"{trade.pair} {trade.side} via {trade.strategy}: "
                f"{'WIN' if trade.pnl > 0 else 'LOSS'} "
                f"({pnl_pct:+.2f}%) — "
                f"entry ${trade.entry_price:.2f} → exit ${trade.exit_price:.2f}"
            )

            # Store in ChromaDB
            trade_id = f"{trade.pair}_{trade.timestamp.strftime('%Y%m%d_%H%M%S')}"
            self._rag_memory.store_trade(trade_id, embedding, metadata, summary)

        except Exception as e:
            logger.debug("Failed to store trade memory: %s", e)

    def _get_rag_context(self, pair: str, snapshot: MarketSnapshot) -> str:
        """Phase 5: Retrieve similar historical scenarios for Gemini context."""
        if not self._rag_memory.is_available or self._rag_memory.memory_count < 5:
            return ""

        try:
            embedding = self._embedding_engine.embed(snapshot)
            memories = self._rag_memory.retrieve_similar(
                embedding, n_results=5, pair_filter=pair
            )
            return self._rag_memory.format_for_gemini(memories)
        except Exception as e:
            logger.debug("RAG retrieval failed: %s", e)
            return ""

    async def _get_balance(self) -> float:
        if self._mode == "paper":
            return self._settings.get("initial_balance", 50.0)
        balances = await self._client.get_balance()
        return balances.get("USDT", 0.0)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping TradingEngine...")
        self._running = False
        if self._client:
            await self._client.disconnect()
        if self._db:
            await self._db.close()
        logger.info("TradingEngine stopped.")
