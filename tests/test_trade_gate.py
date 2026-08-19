# tests/test_trade_gate.py
# Updated for stricter 4/5 consensus requirement.

import pytest
from src.ai.trade_gate import TradeGate, BrainSignal, GateDecision


class TestTradeGate:

    def test_approves_when_4_of_5_agree_buy(self):
        """4+ brains agreeing on BUY -> approved."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("BUY", 0.6, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.direction == "BUY"
        assert decision.agreeing_brains >= 4

    def test_rejects_when_split_votes(self):
        """Split votes: 2 BUY vs 3 SELL -> BUY rejected (40% < 60%)."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("SELL", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("SELL", 0.6, "multi_timeframe"),
            "correlation": BrainSignal("SELL", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is False

    def test_rejects_when_strong_opposing_signal(self):
        """Even with 4 agreeing, a strong OPPOSING signal vetoes."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("BUY", 0.6, "multi_timeframe"),
            "correlation": BrainSignal("SELL", 0.9, "correlation"),  # strong opposing
        }
        decision = gate.evaluate(signals)
        assert decision.approved is False

    def test_confidence_is_average_of_agreeing_brains(self):
        """Overall confidence = average of agreeing brains' confidence."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("SELL", 0.8, "technical"),
            "order_flow": BrainSignal("SELL", 0.6, "order_flow"),
            "sentiment": BrainSignal("SELL", 0.7, "sentiment"),
            "multi_timeframe": BrainSignal("SELL", 0.5, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.3, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.direction == "SELL"
        # Average of 0.8, 0.6, 0.7, 0.5 = 0.65
        assert abs(decision.confidence - 0.65) < 0.01

    def test_hold_signals_dont_count_as_opposing(self):
        """HOLD is neutral -- doesn't count for or against."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.7, "order_flow"),
            "sentiment": BrainSignal("BUY", 0.6, "sentiment"),
            "multi_timeframe": BrainSignal("BUY", 0.65, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True  # HOLDs don't veto

    def test_rejects_single_low_confidence_brain(self):
        """1 brain with low confidence (<0.85) -> rejected."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.6, "technical"),
            "sentiment": BrainSignal("HOLD", 0.3, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.3, "multi_timeframe"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is False

    def test_high_conviction_override_approves(self):
        """Single primary brain with >0.85 confidence passes alone."""
        gate = TradeGate()
        signals = {
            "strategy": BrainSignal("BUY", 0.90, "strategy:momentum"),
            "order_book": BrainSignal("HOLD", 0.5, "order_book"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.direction == "BUY"

    def test_two_agreeing_brains_approve(self):
        """2 brains agreeing on same direction -> approved (meets 60% + min 2)."""
        gate = TradeGate()
        signals = {
            "technical": BrainSignal("BUY", 0.8, "technical"),
            "order_flow": BrainSignal("BUY", 0.6, "order_flow"),
            "sentiment": BrainSignal("HOLD", 0.3, "sentiment"),
            "multi_timeframe": BrainSignal("HOLD", 0.3, "multi_timeframe"),
            "correlation": BrainSignal("HOLD", 0.5, "correlation"),
        }
        decision = gate.evaluate(signals)
        assert decision.approved is True
        assert decision.agreeing_brains == 2
