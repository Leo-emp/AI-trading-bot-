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
