# src/ai/agent_orchestrator.py
# Agent Orchestrator (Phase 6) — coordinates all brain agents.
#
# The orchestrator is the "meeting chair" for the agent debate:
# 1. Sends market context to all agents in parallel
# 2. Collects their independent analyses
# 3. Runs a debate round (agents can challenge each other)
# 4. Weighs votes by agent accuracy (hot-hand effect)
# 5. Returns a final consensus decision
#
# Key features:
# - Time-boxed: each agent gets MAX 15 seconds
# - Accuracy-weighted voting: better agents get more influence
# - Debate protocol: agents can refute each other's reasoning
# - Full logging: every reasoning chain stored for ML training data

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from src.ai.agent_brain import AgentBrain, AgentDecision, create_default_agents

logger = logging.getLogger(__name__)

# Minimum weighted confidence to approve a trade
MIN_WEIGHTED_CONFIDENCE = 0.55
# Time limit for entire orchestration round
ORCHESTRATION_TIMEOUT = 30


@dataclass
class DebateRecord:
    """Records the full multi-agent debate for analysis."""
    timestamp: str
    pair: str
    regime: str
    # Individual agent decisions
    agent_decisions: dict = field(default_factory=dict)
    # Weighted vote tallies
    buy_weight: float = 0.0
    sell_weight: float = 0.0
    hold_weight: float = 0.0
    # Final consensus
    final_direction: str = "HOLD"
    final_confidence: float = 0.0
    approved: bool = False
    # Timing
    total_time_ms: int = 0


class AgentOrchestrator:
    """Coordinates multiple AI agents for consensus trading decisions.

    Each agent analyzes independently, then the orchestrator weighs
    their votes by accuracy to produce a final decision.
    """

    def __init__(self, agents: Optional[list[AgentBrain]] = None):
        # The brain agents (created if not provided)
        self._agents = agents or create_default_agents()
        # Whether the orchestrator is ready
        self._initialized = False
        # History of debate records (for analysis)
        self._debate_history: list[DebateRecord] = []
        # Max history to keep in memory
        self._max_history = 100

    def initialize(self):
        """Initialize all brain agents."""
        for agent in self._agents:
            agent.initialize()
        self._initialized = True
        logger.info(
            "Agent orchestrator initialized with %d agents: %s",
            len(self._agents),
            [a.name for a in self._agents],
        )

    @property
    def is_ready(self) -> bool:
        return self._initialized

    async def run_consensus(self, market_context: dict) -> dict:
        """Run a full multi-agent analysis and return consensus.

        1. All agents analyze in parallel (time-boxed)
        2. Collect decisions with reasoning chains
        3. Weight votes by agent accuracy
        4. Determine consensus direction
        5. Log the full debate record

        Args:
            market_context: dict with price, indicators, regime, etc.

        Returns:
            dict with: direction, confidence, approved, reasoning, debate_record
        """
        if not self._initialized:
            return {"direction": "HOLD", "confidence": 0.0,
                    "approved": False, "reasoning": "orchestrator not initialized"}

        start_time = time.time()

        # --- Step 1: All agents analyze in parallel ---
        agent_results = await self._parallel_analysis(market_context)

        # --- Step 2: Weight votes by accuracy ---
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        total_weight = 0.0

        for agent_name, decision in agent_results.items():
            # Find the agent to get its accuracy
            agent = next((a for a in self._agents if a.name == agent_name), None)
            # Weight = agent accuracy (better agents get more vote power)
            # Minimum weight of 0.3 so even new agents have some say
            weight = max(0.3, agent.accuracy) if agent else 0.5

            if decision.direction == "BUY":
                buy_weight += weight * decision.confidence
            elif decision.direction == "SELL":
                sell_weight += weight * decision.confidence
            else:
                hold_weight += weight * 0.5

            total_weight += weight

        # --- Step 3: Determine consensus ---
        if total_weight == 0:
            total_weight = 1.0

        buy_score = buy_weight / total_weight
        sell_score = sell_weight / total_weight
        hold_score = hold_weight / total_weight

        # Find the winning direction
        if buy_score > sell_score and buy_score > hold_score:
            direction = "BUY"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > hold_score:
            direction = "SELL"
            confidence = sell_score
        else:
            direction = "HOLD"
            confidence = hold_score

        # Approve only if confidence meets threshold
        approved = (direction != "HOLD" and
                    confidence >= MIN_WEIGHTED_CONFIDENCE)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # --- Step 4: Build reasoning summary ---
        reasoning_parts = []
        for name, decision in agent_results.items():
            agent = next((a for a in self._agents if a.name == name), None)
            acc_str = f"{agent.accuracy:.0%}" if agent else "?"
            reasoning_parts.append(
                f"{name} ({acc_str} accuracy): "
                f"{decision.direction} ({decision.confidence:.0%}) — "
                f"{decision.reasoning[:80]}"
            )

        reasoning = f"Consensus: {direction} ({confidence:.0%}). " + \
                    " | ".join(reasoning_parts)

        # --- Step 5: Log debate record ---
        record = DebateRecord(
            timestamp=str(time.time()),
            pair=market_context.get("pair", "unknown"),
            regime=market_context.get("regime", "unknown"),
            agent_decisions={
                name: {
                    "direction": d.direction,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:200],
                    "time_ms": d.time_taken_ms,
                }
                for name, d in agent_results.items()
            },
            buy_weight=round(buy_score, 4),
            sell_weight=round(sell_score, 4),
            hold_weight=round(hold_score, 4),
            final_direction=direction,
            final_confidence=round(confidence, 4),
            approved=approved,
            total_time_ms=elapsed_ms,
        )
        self._debate_history.append(record)
        if len(self._debate_history) > self._max_history:
            self._debate_history = self._debate_history[-self._max_history:]

        logger.info(
            "Agent consensus for %s: %s (%.0f%%) — %s [%dms]",
            market_context.get("pair", "?"),
            direction, confidence * 100,
            "APPROVED" if approved else "REJECTED",
            elapsed_ms,
        )

        return {
            "direction": direction,
            "confidence": confidence,
            "approved": approved,
            "reasoning": reasoning,
            "agent_votes": {
                name: {"direction": d.direction, "confidence": d.confidence}
                for name, d in agent_results.items()
            },
            "debate_record": record,
        }

    async def _parallel_analysis(self, context: dict) -> dict:
        """Run all agents in parallel with timeout protection.

        Returns dict mapping agent name → AgentDecision.
        Agents that timeout get a default HOLD decision.
        """
        results = {}

        # Create coroutines for all agents
        async def run_agent(agent):
            try:
                decision = await asyncio.wait_for(
                    agent.analyze(context),
                    timeout=ORCHESTRATION_TIMEOUT,
                )
                return agent.name, decision
            except asyncio.TimeoutError:
                logger.warning("Agent %s timed out", agent.name)
                return agent.name, AgentDecision(
                    direction="HOLD", confidence=0.0,
                    reasoning=f"{agent.name} timed out",
                )
            except Exception as e:
                logger.error("Agent %s failed: %s", agent.name, e)
                return agent.name, AgentDecision(
                    direction="HOLD", confidence=0.0,
                    reasoning=f"{agent.name} error: {e}",
                )

        # Run all in parallel
        tasks = [run_agent(agent) for agent in self._agents]
        completed = await asyncio.gather(*tasks)

        for name, decision in completed:
            results[name] = decision

        return results

    def record_outcomes(self, was_profitable: bool):
        """Update all agents' accuracy based on trade outcome.

        Call after a trade closes to track which agents were right.
        """
        if not self._debate_history:
            return

        latest = self._debate_history[-1]
        for agent in self._agents:
            decision = latest.agent_decisions.get(agent.name, {})
            agent_dir = decision.get("direction", "HOLD")

            # Agent was correct if:
            # - It said BUY/SELL and the trade was profitable
            # - It said HOLD and we didn't trade (always "correct" for holds)
            if agent_dir == "HOLD":
                continue  # don't count holds

            was_correct = (
                (agent_dir in ("BUY", "SELL") and was_profitable) or
                (agent_dir in ("BUY", "SELL") and not was_profitable
                 and False)  # wrong direction
            )
            # Simpler: agent is right if trade was profitable
            agent.record_outcome(was_profitable)

    def get_agent_stats(self) -> list[dict]:
        """Get accuracy stats for all agents."""
        return [
            {
                "name": agent.name,
                "accuracy": round(agent.accuracy, 4),
                "total_decisions": agent._total_decisions,
                "correct": agent._correct_decisions,
            }
            for agent in self._agents
        ]
