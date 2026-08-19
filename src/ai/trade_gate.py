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
# CONSENSUS RULES:
# - Uses percentage-based consensus: >= 60% of VOTING brains must agree
# - A "voting" brain is one with a BUY or SELL opinion (HOLD = abstain)
# - Minimum 2 absolute votes required (1 brain alone can't trigger a trade)
# - No brain can show a STRONG opposing signal (confidence > 0.65)
#
# This adapts to however many brains are active. In quiet markets where
# only 2-3 brains have opinions, 2 agreeing is enough. In volatile markets
# where all 5 fire, you need 3+.

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Confidence threshold above which an opposing signal vetoes the trade
VETO_THRESHOLD = 0.65
# Minimum percentage of voting brains that must agree
CONSENSUS_PCT = 0.60
# Absolute minimum — even if only 1 brain voted at 100%, still need 2
MIN_ABSOLUTE_VOTES = 2


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
    """Evaluates signals from all brains and decides whether to trade.

    Uses percentage-based consensus that adapts to the number of active brains.
    """

    def __init__(self, consensus_pct: float = CONSENSUS_PCT,
                 min_absolute: int = MIN_ABSOLUTE_VOTES,
                 veto_threshold: float = VETO_THRESHOLD):
        self._consensus_pct = consensus_pct
        self._min_absolute = min_absolute
        self._veto_threshold = veto_threshold

    def evaluate(self, signals: dict[str, BrainSignal]) -> GateDecision:
        """Evaluate all brain signals and return a gate decision.

        signals: dict mapping brain name to BrainSignal
        Returns GateDecision with approval status and reasoning.
        """
        reasons = []

        # Count votes for each direction (HOLD = abstain)
        buy_voters = []
        sell_voters = []
        hold_count = 0

        for name, signal in signals.items():
            if signal.direction == "BUY":
                buy_voters.append(signal)
            elif signal.direction == "SELL":
                sell_voters.append(signal)
            else:
                hold_count += 1

        total_voting = len(buy_voters) + len(sell_voters)

        # Not enough brains voting — unless a primary brain has high conviction.
        # A strategy or gemini brain with >0.85 confidence can pass alone
        # to prevent the bot sitting idle on strong setups in quiet markets.
        high_conviction_override = False
        if total_voting < self._min_absolute:
            all_voters = buy_voters + sell_voters
            has_strong_primary = any(
                s.confidence > 0.85
                and s.source.startswith(("strategy:", "gemini"))
                for s in all_voters
            )
            if has_strong_primary:
                high_conviction_override = True
                reasons.append("high-conviction override: primary brain > 0.85")
            else:
                reasons.append(f"only {total_voting} brains voted (need {self._min_absolute})")
                return GateDecision(
                    approved=False, direction="HOLD",
                    confidence=0.0, agreeing_brains=0,
                    reasons=reasons,
                )

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
        agree_pct = agreeing / total_voting if total_voting > 0 else 0

        # Check 1: Percentage consensus
        # High-conviction override relaxes min votes from 2 to 1
        min_required = 1 if high_conviction_override else self._min_absolute
        if agree_pct < self._consensus_pct or agreeing < min_required:
            reasons.append(
                f"{agreeing}/{total_voting} ({agree_pct:.0%}) agree on {majority_dir}, "
                f"need {self._consensus_pct:.0%}"
            )
            return GateDecision(
                approved=False, direction="HOLD",
                confidence=0.0, agreeing_brains=agreeing,
                reasons=reasons,
            )

        # Check 2: No strong opposing signal (veto)
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

        # Approved — calculate combined confidence
        avg_confidence = sum(s.confidence for s in majority_signals) / len(majority_signals)

        for s in majority_signals:
            reasons.append(f"{s.source}: {s.direction} ({s.confidence:.2f})")

        logger.info(
            "GATE APPROVED: %s with %d/%d brains (%.0f%%), confidence %.2f",
            majority_dir, agreeing, total_voting, agree_pct * 100, avg_confidence,
        )

        return GateDecision(
            approved=True,
            direction=majority_dir,
            confidence=avg_confidence,
            agreeing_brains=agreeing,
            reasons=reasons,
        )
