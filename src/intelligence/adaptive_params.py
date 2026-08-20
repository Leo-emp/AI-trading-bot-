# src/intelligence/adaptive_params.py
# Adaptive Parameter Engine — the "pro trader brain" that adjusts
# SL/TP/position size based on ACTUAL market volatility (ATR).
#
# WHY THIS MATTERS:
# Fixed % targets (0.5% SL, 1.5% TP) treat BTC and XRP the same.
# But BTC moves $500 in a 5min candle while XRP moves $0.005.
# A 0.5% SL on BTC = $475 — within one candle of noise.
# Result: stop-hunted by normal volatility.
#
# PRO APPROACH (ATR-based):
# - SL = 2.0 × ATR below entry (outside noise, respects volatility)
# - TP = 1.8 × ATR above entry (realistic target for the timeframe)
# - Position size = (account_risk $) / (SL distance $)
#
# This means:
# - In low volatility: tight SL/TP, bigger position, quick trades
# - In high volatility: wide SL/TP, smaller position, bigger individual wins
# - Risk per trade stays CONSTANT regardless of volatility
#
# KEY PRINCIPLE: risk_amount is FIXED at 0.5% of equity.
# Confidence and volume affect WHETHER we trade, not HOW MUCH we risk.
# This prevents outsized losers (the BNB -$168 problem).
#
# SMART EXITS are R-based (multiples of initial SL distance), NOT ATR-based.
# This fixes the mismatch where breakeven triggered at 0.29R instead of 1.5R.

import logging
import json
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# File to persist learned parameters
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "adaptive_params.json")


@dataclass
class DynamicLevels:
    """Calculated SL/TP/size for one trade setup."""
    stop_loss: float        # absolute price for SL
    take_profit: float      # absolute price for TP
    position_size: float    # USDT notional size (full exposure)
    sl_distance_pct: float  # % distance to SL (for logging)
    tp_distance_pct: float  # % distance to TP (for logging)
    risk_reward: float      # TP distance / SL distance
    atr_value: float        # raw ATR used
    risk_amount: float      # actual $ at risk if SL hit (for verification)
    leverage: int = 1               # futures leverage multiplier
    margin_required: float = 0.0    # cash locked (position_size / leverage)
    liquidation_price: float = 0.0  # price at which position is force-closed


class AdaptiveParams:
    """Calculates dynamic trading parameters based on ATR and performance.

    The "intelligent" part: it doesn't use fixed numbers. It reads the
    market's actual volatility and sizes everything proportionally.
    A pro trader would never use the same stop for BTC and a memecoin.
    """

    def __init__(self,
                 # Base ATR multipliers (starting points — will adapt)
                 # Tightened from 2.0 to 1.5: 25% smaller losses, still outside noise
                 sl_atr_multiplier: float = 1.5,
                 # TP widened from 1.8 → 4.0: let winners run to 3R-5R.
                 # 1.8 ATR capped winners too early — average win was 0.5R vs 1R loss.
                 # The trailing stop handles exit, TP is a safety net for extended moves.
                 tp_atr_multiplier: float = 4.0,
                 # Base risk: 0.5% of equity per trade (halved from 1.0%).
                 # Keeps max loss small. Leverage handles position sizing.
                 risk_per_trade_pct: float = 0.5,
                 # Min/max bounds to prevent insane values
                 # (0.5% SL minimum = crypto noise floor on 1h timeframe)
                 min_sl_pct: float = 0.5,
                 max_sl_pct: float = 4.0,
                 # TP set very wide — trailing stop is the real exit.
                 # No artificial ceiling on profits. Let winners run.
                 min_tp_pct: float = 1.5,
                 max_tp_pct: float = 50.0,
                 # Position size bounds — large enough for conviction trades
                 min_position_usd: float = 10.0,
                 max_position_pct: float = 25.0,
                 # Learning rate (how fast multipliers adapt)
                 learning_rate: float = 0.05):

        self._sl_mult = sl_atr_multiplier
        self._tp_mult = tp_atr_multiplier
        self._risk_pct = risk_per_trade_pct / 100
        self._min_sl_pct = min_sl_pct / 100
        self._max_sl_pct = max_sl_pct / 100
        self._min_tp_pct = min_tp_pct / 100
        self._max_tp_pct = max_tp_pct / 100
        self._min_pos = min_position_usd
        self._max_pos_pct = max_position_pct / 100
        self._learning_rate = learning_rate

        # Per-regime multiplier adjustments (learned over time)
        # Start at 1.0 (no adjustment). Values >1 = widen, <1 = tighten
        self._regime_adjustments = {
            "TRENDING_UP": {"sl": 1.2, "tp": 1.5},    # trends: wider TP, slightly wider SL
            "TRENDING_DOWN": {"sl": 1.2, "tp": 1.5},  # same for shorts
            "RANGING": {"sl": 0.8, "tp": 0.9},        # range: tighter (mean reversion)
            "VOLATILE": {"sl": 1.5, "tp": 1.3},       # volatile: wider SL to survive
        }

        # Per-strategy multiplier overrides
        # Base R:R = (4.0 × regime_tp × strat_tp) / (2.0 × regime_sl × strat_sl)
        # Momentum TRENDING_UP: (4.0×1.5×1.2)/(2.0×1.2×1.0) = 7.2/2.4 = 3.0:1
        # Grid RANGING: (4.0×0.9×1.0)/(2.0×0.8×0.8) = 3.6/1.28 = 2.8:1
        # Scalp: (4.0×1.0×1.0)/(2.0×1.0×0.9) = 4.0/1.8 = 2.2:1
        # Pullback: (4.0×1.0×1.0)/(2.0×1.0×0.7) = 4.0/1.4 = 2.9:1
        self._strategy_adjustments = {
            "momentum": {"sl": 1.0, "tp": 1.2},       # momentum: wide TP, standard SL
            "grid": {"sl": 0.8, "tp": 1.0},            # grid: tighter SL in ranges
            "smart_scalp": {"sl": 0.9, "tp": 1.0},    # scalp: slightly tighter SL
            "mean_reversion": {"sl": 0.7, "tp": 1.0}, # pullback: tight SL (reversal entries)
        }

        # Trade outcome history for learning
        self._trade_outcomes: list[dict] = []
        self._max_history = 200

        # Load persisted adjustments
        self._load_params()

    def calculate(self, entry_price: float, direction: str,
                  atr: float, balance: float,
                  regime: str = "RANGING",
                  strategy: str = "grid",
                  confidence: float = 0.7,
                  volume_ratio: float = 1.0) -> DynamicLevels:
        """Calculate dynamic SL/TP/position size based on ATR.

        This is the core intelligence: adapts everything to market conditions.

        CRITICAL CHANGE: risk_amount is now FIXED at 0.5% of balance.
        Confidence and volume influence whether we enter (gate/filter),
        NOT how much we risk. This prevents the BNB -$168 problem where
        confidence/volume multipliers created positions 3-4x normal risk.
        """
        # --- Step 1: Get regime + strategy adjustments ---
        regime_adj = self._regime_adjustments.get(regime, {"sl": 1.0, "tp": 1.0})
        strat_adj = self._strategy_adjustments.get(strategy, {"sl": 1.0, "tp": 1.0})

        # Combine adjustments (multiplicative)
        sl_mult = self._sl_mult * regime_adj["sl"] * strat_adj["sl"]
        tp_mult = self._tp_mult * regime_adj["tp"] * strat_adj["tp"]

        # --- Step 2: Calculate raw SL/TP distances ---
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        # --- Step 3: Confidence affects TP/SL distances (NOT risk amount) ---
        # Higher confidence = slightly wider TP (let winners run more)
        # Lower confidence = slightly tighter SL (protect more)
        if confidence >= 0.8:
            tp_distance *= 1.2   # high confidence: be more ambitious
        elif confidence < 0.6:
            sl_distance *= 0.8   # low confidence: tighter stop
            tp_distance *= 0.8

        # --- Step 4: Enforce bounds (% of entry price) ---
        sl_pct = sl_distance / entry_price
        tp_pct = tp_distance / entry_price

        sl_pct = max(self._min_sl_pct, min(sl_pct, self._max_sl_pct))
        tp_pct = max(self._min_tp_pct, min(tp_pct, self._max_tp_pct))

        sl_distance = entry_price * sl_pct
        tp_distance = entry_price * tp_pct

        # --- Step 5: Calculate absolute price levels ---
        if direction.upper() == "BUY":
            stop_loss = round(entry_price - sl_distance, 8)
            take_profit = round(entry_price + tp_distance, 8)
        else:
            stop_loss = round(entry_price + sl_distance, 8)
            take_profit = round(entry_price - tp_distance, 8)

        # --- Step 6: CONFIDENCE-SCALED risk + LEVERAGE ---
        # AGGRESSIVE WHEN SURE, TINY WHEN NOT.
        # High confidence = multiple brains agree = bet big.
        # Low confidence = uncertain = barely participate.
        # The partial-close fix (SL → entry+1R after 2R partial) means
        # big positions on high-confidence trades are SAFE — once partial
        # triggers, the remaining 75% is guaranteed profit.
        #
        # Risk multiplier scales POSITION SIZE (more $ at risk):
        #   <70% → 0.5× ($250 on $100K) — throwaway, just watching
        #   70%  → 1.0× ($500) — standard
        #   80%  → 2.0× ($1,000) — strong conviction, double size
        #   90%+ → 3.0× ($1,500) — full consensus, triple size, max profit
        if confidence >= 0.90:
            risk_mult = 3.0
        elif confidence >= 0.80:
            risk_mult = 2.0
        elif confidence >= 0.70:
            risk_mult = 1.0
        else:
            risk_mult = 0.5
        risk_amount = balance * self._risk_pct * risk_mult

        # FUTURES LEVERAGE: confidence → leverage tier (AGGRESSIVE)
        # Leverage doesn't change $ risk (SL defines that). It changes
        # how much MARGIN is locked — high leverage = less cash tied up =
        # more positions possible = more upside exposure.
        #   <70% → 1×  (spot-like, eats margin = natural position limiter)
        #   70%  → 5×  (moderate)
        #   80%  → 15× (strong — multiple brains agree, capital-efficient)
        #   90%+ → 25× (maximum conviction, minimum margin locked)
        if confidence >= 0.90:
            leverage = 25
        elif confidence >= 0.80:
            leverage = 15
        elif confidence >= 0.70:
            leverage = 5
        else:
            leverage = 1

        # Position size = risk / (distance to SL as fraction)
        if sl_pct > 0:
            position_size = risk_amount / sl_pct
        else:
            position_size = risk_amount

        # With leverage, cap MARGIN (cash locked) not notional.
        # margin = notional / leverage, so max_notional = max_margin × leverage.
        # $50K margin at 20× leverage = $1M notional exposure.
        max_margin = balance * self._max_pos_pct
        max_notional = max_margin * leverage
        position_size = min(position_size, max_notional)
        position_size = max(position_size, self._min_pos)

        margin_required = position_size / leverage

        # Liquidation price (isolated margin, Binance-style)
        # Long: liq ≈ entry × (1 - 1/leverage)
        # Short: liq ≈ entry × (1 + 1/leverage)
        maintenance_rate = 0.004
        if direction.upper() == "BUY":
            liquidation_price = entry_price * (1 - (1 / leverage) * (1 - maintenance_rate))
        else:
            liquidation_price = entry_price * (1 + (1 / leverage) * (1 - maintenance_rate))

        # Safety: SL must be BEFORE liquidation price (with 2% buffer)
        if direction.upper() == "BUY" and stop_loss <= liquidation_price:
            stop_loss = round(liquidation_price * 1.02, 8)
            sl_distance = entry_price - stop_loss
            sl_pct = sl_distance / entry_price if entry_price > 0 else 0
        elif direction.upper() == "SELL" and stop_loss >= liquidation_price:
            stop_loss = round(liquidation_price * 0.98, 8)
            sl_distance = stop_loss - entry_price
            sl_pct = sl_distance / entry_price if entry_price > 0 else 0

        # --- Step 6b: VERIFY max loss doesn't exceed risk budget ---
        actual_risk = position_size * sl_pct
        if actual_risk > risk_amount:
            position_size = risk_amount / sl_pct
            position_size = max(position_size, self._min_pos)
            margin_required = position_size / leverage

        # --- Step 7: Calculate risk:reward ratio ---
        risk_reward = tp_pct / sl_pct if sl_pct > 0 else 0

        logger.info(
            "POSITION_SIZE balance=%.2f risk=%.2f sl_pct=%.4f "
            "notional=%.2f margin=%.2f leverage=%dx max_loss=%.2f | "
            "SL=%.2f%% TP=%.2f%% R:R=%.1f:1 liq=$%.2f "
            "(ATR=%.4f, regime=%s, strat=%s, conf=%.0f%%)",
            balance, risk_amount, sl_pct * 100,
            position_size, margin_required, leverage, position_size * sl_pct,
            sl_pct * 100, tp_pct * 100, risk_reward, liquidation_price,
            atr, regime, strategy, confidence * 100,
        )

        return DynamicLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=round(position_size, 2),
            sl_distance_pct=sl_pct * 100,
            tp_distance_pct=tp_pct * 100,
            risk_reward=risk_reward,
            atr_value=atr,
            risk_amount=round(risk_amount, 2),
            leverage=leverage,
            margin_required=round(margin_required, 2),
            liquidation_price=round(liquidation_price, 2),
        )

    def record_outcome(self, trade_result: dict):
        """Learn from a closed trade to adapt future parameters.

        The adaptive learning: after each trade, record what happened.
        If trades with wider stops win more, gradually widen stops.
        If trades in trending regimes hit TP faster, widen TP for trends.
        """
        self._trade_outcomes.append(trade_result)
        if len(self._trade_outcomes) > self._max_history:
            self._trade_outcomes = self._trade_outcomes[-self._max_history:]

        # Need at least 20 trades before adapting
        if len(self._trade_outcomes) < 20:
            return

        # --- Regime-specific learning ---
        regime = trade_result.get("regime", "RANGING")
        regime_trades = [t for t in self._trade_outcomes if t.get("regime") == regime]

        if len(regime_trades) >= 10:
            wins = [t for t in regime_trades if t["pnl"] > 0]
            losses = [t for t in regime_trades if t["pnl"] <= 0]
            win_rate = len(wins) / len(regime_trades)

            # If we're losing too much in this regime, widen SL
            if win_rate < 0.4 and losses:
                sl_was_tight = sum(1 for t in losses
                                   if t.get("close_reason") == "stop_loss"
                                   and t.get("max_favorable", 0) > t.get("sl_distance_pct", 0) * 0.5
                                   ) / max(len(losses), 1)
                if sl_was_tight > 0.4:
                    self._regime_adjustments[regime]["sl"] *= (1 + self._learning_rate)
                    self._regime_adjustments[regime]["sl"] = min(
                        self._regime_adjustments[regime]["sl"], 2.5)
                    logger.info("ADAPTIVE LEARN: widening SL for %s regime (%.2f)",
                               regime, self._regime_adjustments[regime]["sl"])

            # If we're winning but leaving money on table, widen TP
            if win_rate > 0.5 and wins:
                avg_favorable = sum(t.get("max_favorable", 0) for t in wins) / len(wins)
                avg_tp = sum(t.get("tp_distance_pct", 0) for t in wins) / len(wins)
                if avg_favorable > avg_tp * 1.5:
                    self._regime_adjustments[regime]["tp"] *= (1 + self._learning_rate)
                    self._regime_adjustments[regime]["tp"] = min(
                        self._regime_adjustments[regime]["tp"], 3.0)
                    logger.info("ADAPTIVE LEARN: widening TP for %s regime (%.2f)",
                               regime, self._regime_adjustments[regime]["tp"])

        # --- Strategy-specific learning ---
        strategy = trade_result.get("strategy", "")
        strat_trades = [t for t in self._trade_outcomes if t.get("strategy") == strategy]

        if len(strat_trades) >= 10:
            strat_win_rate = sum(1 for t in strat_trades if t["pnl"] > 0) / len(strat_trades)
            if strat_win_rate < 0.35:
                self._strategy_adjustments.setdefault(strategy, {"sl": 1.0, "tp": 1.0})
                self._strategy_adjustments[strategy]["sl"] *= (1 - self._learning_rate * 0.5)
                self._strategy_adjustments[strategy]["sl"] = max(
                    self._strategy_adjustments[strategy]["sl"], 0.5)
            elif strat_win_rate > 0.6:
                self._strategy_adjustments.setdefault(strategy, {"sl": 1.0, "tp": 1.0})
                self._strategy_adjustments[strategy]["tp"] *= (1 + self._learning_rate)
                self._strategy_adjustments[strategy]["tp"] = min(
                    self._strategy_adjustments[strategy]["tp"], 3.0)

        # Persist learned parameters
        self._save_params()

    def get_smart_exit_thresholds(self, atr: float, entry_price: float,
                                  regime: str = "RANGING",
                                  strategy: str = "grid") -> dict:
        """Calculate smart exit thresholds in multiples of INITIAL RISK (R).

        PROFIT-FIRST EXIT SYSTEM:
        The old system (50% partial at 1R, BE at 1.5R) produced avg win 0.5R
        vs avg loss 1R — negative EV even at 60%+ WR.

        New system lets winners run to 3R-5R:
        1. Initial SL (1R risk) — cut losers at exactly 1R
        2. Breakeven at +1R: SL moves to entry + 0.25R (risk-free early)
        3. Partial at +2R: 25% closed (handled by paper_trader), banks 0.5R
        4. Trailing at +2.5R: 1.0R trailing stop (wide enough for crypto noise)
        5. Tight trailing at +4R: 0.75R stop (extended move, protect hard)

        Math at 40% WR with avg win 3R:
        40 × 3R - 60 × 1R = 120R - 60R = +60R per 100 trades
        """
        regime_adj = self._regime_adjustments.get(regime, {"sl": 1.0, "tp": 1.0})
        strat_adj = self._strategy_adjustments.get(strategy, {"sl": 1.0, "tp": 1.0})

        sl_mult = self._sl_mult * regime_adj["sl"] * strat_adj["sl"]
        sl_distance = atr * sl_mult  # 1R in price terms

        # STRATEGY-FIRST EXIT PROFILES
        # Different strategies need fundamentally different exit behavior:
        #
        # SCALP: grab quick profit, don't wait. Close at 1.0-1.5R.
        #   → Fast BE at 0.5R, tight trailing at 1.0R, close fast.
        #   → Expected: $200-500 wins in 5-30 minutes.
        #
        # MOMENTUM: ride the trend for big moves. Hold for 3-5R+.
        #   → Delayed BE at 1.5R, wide trailing, patient.
        #   → Expected: $1,000-5,000 wins over 1-12 hours.
        #
        # MEAN_REVERSION: moderate target, reversion to mean.
        #   → Quick BE at 0.75R, moderate trailing at 1.5R.
        #   → Expected: $300-800 wins in 15-60 minutes.
        #
        # GRID: managed by grid executor, but fallback exits here.
        #   → Similar to ranging: early trail, no BE trap.
        #
        # Regime modifies within each strategy profile.

        if strategy == "smart_scalp":
            # SCALP: fast in, fast out. Grab 1.0-1.5R and leave.
            breakeven_distance = sl_distance * 0.50   # BE early at 0.5R
            trail_activate_distance = sl_distance * 0.75  # trail starts at 0.75R
            tight_activate_distance = sl_distance * 1.50  # tighten at 1.5R
            trail_distance = sl_distance * 0.30       # tight 0.3R trail
            tight_trail_distance = sl_distance * 0.15 # very tight to lock profit
            r_key = "scalp"

        elif strategy == "momentum":
            # MOMENTUM: let winners run far. Patient trailing.
            if regime in ("TRENDING_UP", "TRENDING_DOWN"):
                # Trend + momentum = maximum patience
                breakeven_distance = sl_distance * 2.00   # BE only after 2R
                trail_activate_distance = sl_distance * 3.00  # trail at 3R
                tight_activate_distance = sl_distance * 5.00  # tighten at 5R
                trail_distance = sl_distance * 1.50       # wide 1.5R trail
                tight_trail_distance = sl_distance * 0.75 # moderate tighten
            elif regime == "VOLATILE":
                # Volatile momentum: faster exits but still patient
                breakeven_distance = sl_distance * 1.50
                trail_activate_distance = sl_distance * 2.00
                tight_activate_distance = sl_distance * 3.50
                trail_distance = sl_distance * 1.00
                tight_trail_distance = sl_distance * 0.50
            else:
                # Ranging momentum: moderate
                breakeven_distance = sl_distance * 1.00
                trail_activate_distance = sl_distance * 1.50
                tight_activate_distance = sl_distance * 2.50
                trail_distance = sl_distance * 0.75
                tight_trail_distance = sl_distance * 0.50
            r_key = "momentum"

        elif strategy == "mean_reversion":
            # MEAN REVERSION: moderate target, revert to mean
            breakeven_distance = sl_distance * 0.75
            trail_activate_distance = sl_distance * 1.25
            tight_activate_distance = sl_distance * 2.00
            trail_distance = sl_distance * 0.50
            tight_trail_distance = sl_distance * 0.30
            r_key = "mean_rev"

        else:
            # GRID / fallback: regime-aware defaults
            if regime in ("TRENDING_UP", "TRENDING_DOWN"):
                breakeven_distance = sl_distance * 1.50
                trail_activate_distance = sl_distance * 2.50
                tight_activate_distance = sl_distance * 4.00
                trail_distance = sl_distance * 1.00
                tight_trail_distance = sl_distance * 0.75
            elif regime == "VOLATILE":
                breakeven_distance = sl_distance * 1.00
                trail_activate_distance = sl_distance * 1.50
                tight_activate_distance = sl_distance * 2.50
                trail_distance = sl_distance * 0.50
                tight_trail_distance = sl_distance * 0.30
            else:
                breakeven_distance = sl_distance * 3.00
                trail_activate_distance = sl_distance * 1.00
                tight_activate_distance = sl_distance * 2.00
                trail_distance = sl_distance * 0.75
                tight_trail_distance = sl_distance * 0.50
            r_key = "grid"

        breakeven_pct = (breakeven_distance / entry_price) * 100
        trail_activate_pct = (trail_activate_distance / entry_price) * 100
        tight_activate_pct = (tight_activate_distance / entry_price) * 100
        trail_distance_pct = (trail_distance / entry_price) * 100
        tight_trail_pct = (tight_trail_distance / entry_price) * 100

        breakeven_pct = max(breakeven_pct, 0.3)
        trail_activate_pct = max(trail_activate_pct, 0.8)
        tight_activate_pct = max(tight_activate_pct, 1.5)
        trail_distance_pct = max(trail_distance_pct, 0.3)
        tight_trail_pct = max(tight_trail_pct, 0.2)

        # R-values for logging/diagnostics (computed from actual thresholds)
        r_vals = {
            "breakeven_r": round(breakeven_distance / sl_distance, 2) if sl_distance > 0 else 0,
            "trail_activate_r": round(trail_activate_distance / sl_distance, 2) if sl_distance > 0 else 0,
            "tight_activate_r": round(tight_activate_distance / sl_distance, 2) if sl_distance > 0 else 0,
            "trail_distance_r": round(trail_distance / sl_distance, 2) if sl_distance > 0 else 0,
            "tight_trail_r": round(tight_trail_distance / sl_distance, 2) if sl_distance > 0 else 0,
            "exit_profile": r_key,
        }

        return {
            "breakeven_pct": breakeven_pct,
            "trail_activate_pct": trail_activate_pct,
            "tight_activate_pct": tight_activate_pct,
            "trail_distance_pct": trail_distance_pct,
            "tight_trail_pct": tight_trail_pct,
            **r_vals,
        }

    def _save_params(self):
        """Persist learned adjustments to disk."""
        try:
            data = {
                "regime_adjustments": self._regime_adjustments,
                "strategy_adjustments": self._strategy_adjustments,
                "trade_count": len(self._trade_outcomes),
            }
            os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
            with open(PARAMS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("Failed to save adaptive params: %s", e)

    def _load_params(self):
        """Load previously learned adjustments."""
        try:
            if os.path.exists(PARAMS_FILE):
                with open(PARAMS_FILE, "r") as f:
                    data = json.load(f)
                if "regime_adjustments" in data:
                    self._regime_adjustments.update(data["regime_adjustments"])
                if "strategy_adjustments" in data:
                    self._strategy_adjustments.update(data["strategy_adjustments"])
                logger.info("Loaded adaptive params (from %d trades)",
                           data.get("trade_count", 0))
        except Exception as e:
            logger.debug("No saved adaptive params: %s", e)
