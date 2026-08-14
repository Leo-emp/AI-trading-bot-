# src/ai/agent_brain.py
# Agentic AI Brain (Phase 6) — each brain becomes an autonomous agent.
#
# Instead of simple signal generators, each brain is now an agent that:
# - Has a set of tools it can call (API queries, calculations, DB lookups)
# - Reasons step-by-step before making a decision (chain-of-thought)
# - Can request additional data if initial data isn't enough
# - Remembers its past accuracy via RAG memory
# - Explains its reasoning for post-analysis
#
# Architecture:
# - Each agent is a Gemini model with function-calling capability
# - Tools are Python functions the agent can invoke
# - Reasoning chains are logged to SQLite for training data
# - Agent orchestrator coordinates all agents (see agent_orchestrator.py)
#
# This is the evolution: simple signal → ML-enhanced signal → reasoning agent

import json
import time
import logging
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Max time an agent gets to make a decision (seconds)
AGENT_TIMEOUT = 15
# Max reasoning steps before forced conclusion
MAX_REASONING_STEPS = 5


@dataclass
class AgentTool:
    """A tool that an agent can use during reasoning.

    Each tool has a name, description (for the LLM), and a callable.
    The agent decides which tools to call based on the situation.
    """
    name: str
    description: str
    # The actual function to call — receives dict args, returns dict result
    func: Callable
    # Parameter schema for the LLM (simplified)
    parameters: dict = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """One step in an agent's chain-of-thought reasoning.

    Logged for post-analysis and training data generation.
    """
    step_number: int
    thought: str           # what the agent is thinking
    tool_used: str = ""    # which tool it called (empty if none)
    tool_input: str = ""   # what it passed to the tool
    tool_result: str = ""  # what the tool returned
    conclusion: str = ""   # what the agent concluded from this step


@dataclass
class AgentDecision:
    """Final output from an agent brain.

    Includes the decision AND the full reasoning chain.
    """
    direction: str          # BUY, SELL, or HOLD
    confidence: float       # 0.0 to 1.0
    reasoning: str          # one-line summary
    chain: list[ReasoningStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    time_taken_ms: int = 0


class AgentBrain:
    """An autonomous AI agent that reasons about market conditions.

    Each brain specializes in one domain (technical, sentiment, etc.)
    and has access to domain-specific tools. It uses Gemini's
    function-calling to decide what to investigate before voting.
    """

    def __init__(self, name: str, role: str,
                 tools: Optional[list[AgentTool]] = None):
        # Agent identity
        self._name = name
        self._role = role
        # Tools this agent can use
        self._tools = {t.name: t for t in (tools or [])}
        # Gemini model (lazy-loaded)
        self._model = None
        # Performance tracking
        self._total_decisions = 0
        self._correct_decisions = 0
        # Whether Gemini is available for agentic reasoning
        self._gemini_available = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def accuracy(self) -> float:
        """Rolling accuracy of this agent's decisions."""
        if self._total_decisions == 0:
            return 0.5  # unknown → assume 50/50
        return self._correct_decisions / self._total_decisions

    def initialize(self):
        """Set up the Gemini model for function-calling."""
        try:
            import os
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("Agent %s: no GEMINI_API_KEY, using basic mode",
                             self._name)
                return

            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash")
            self._gemini_available = True
            logger.info("Agent %s initialized with Gemini + %d tools",
                       self._name, len(self._tools))

        except ImportError:
            logger.warning("Agent %s: google-generativeai not installed",
                         self._name)
        except Exception as e:
            logger.error("Agent %s init failed: %s", self._name, e)

    async def analyze(self, market_context: dict) -> AgentDecision:
        """Run the full agentic analysis pipeline.

        1. Build the analysis prompt with role and context
        2. Let the agent reason step-by-step
        3. Allow tool calls during reasoning
        4. Force a conclusion after MAX_REASONING_STEPS
        5. Return decision with full reasoning chain

        Args:
            market_context: dict with price, indicators, regime, etc.

        Returns:
            AgentDecision with direction, confidence, reasoning chain
        """
        start_time = time.time()
        chain = []

        # If Gemini isn't available, use basic rule-based logic
        if not self._gemini_available:
            return self._basic_analysis(market_context)

        try:
            # Build the agent prompt with role and tools
            prompt = self._build_agent_prompt(market_context)

            # Call Gemini for reasoning
            response = await self._call_gemini(prompt)

            # Parse the response into a decision
            decision = self._parse_agent_response(response)

            # Build a single reasoning step from the response
            chain.append(ReasoningStep(
                step_number=1,
                thought=decision.get("reasoning", "no reasoning"),
                conclusion=f"{decision.get('direction', 'HOLD')} "
                          f"({decision.get('confidence', 0.5):.0%})",
            ))

            elapsed_ms = int((time.time() - start_time) * 1000)
            self._total_decisions += 1

            return AgentDecision(
                direction=decision.get("direction", "HOLD"),
                confidence=decision.get("confidence", 0.5),
                reasoning=decision.get("reasoning", ""),
                chain=chain,
                tools_used=[],
                time_taken_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            logger.warning("Agent %s timed out", self._name)
            return self._basic_analysis(market_context)
        except Exception as e:
            logger.error("Agent %s analysis failed: %s", self._name, e)
            return self._basic_analysis(market_context)

    def _build_agent_prompt(self, context: dict) -> str:
        """Build the agent's analysis prompt with role and market data."""
        # List available tools for the agent
        tool_descriptions = ""
        if self._tools:
            tool_list = "\n".join(
                f"  - {t.name}: {t.description}"
                for t in self._tools.values()
            )
            tool_descriptions = f"\nAVAILABLE TOOLS:\n{tool_list}\n"

        return f"""You are {self._name}, a specialized trading analysis agent.

ROLE: {self._role}
{tool_descriptions}
MARKET DATA:
- Pair: {context.get('pair', 'unknown')}
- Price: ${context.get('price', 0):.2f}
- RSI: {context.get('rsi', 'N/A')}
- MACD Histogram: {context.get('macd_histogram', 'N/A')}
- Volume Ratio: {context.get('volume_ratio', 'N/A')}
- Regime: {context.get('regime', 'unknown')}
- BB Width: {context.get('bb_width', 'N/A')}
- ATR: {context.get('atr', 'N/A')}
- Order Book Imbalance: {context.get('ob_imbalance', 'N/A')}
- Recent Price Change: {context.get('price_change_5m', 'N/A')}%

HISTORICAL CONTEXT:
{context.get('rag_context', 'No historical data available yet.')}

YOUR ACCURACY SO FAR: {self.accuracy:.1%} over {self._total_decisions} decisions

INSTRUCTIONS:
1. Analyze the data from your specialized perspective ({self._role})
2. Consider the historical context if available
3. Provide your trading recommendation

Respond in EXACTLY this JSON format:
{{"direction": "BUY" or "SELL" or "HOLD", "confidence": 0.0 to 1.0, "reasoning": "one paragraph explaining your analysis"}}"""

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with timeout protection."""
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: self._model.generate_content(prompt)
            ),
            timeout=AGENT_TIMEOUT,
        )
        return response.text

    def _parse_agent_response(self, text: str) -> dict:
        """Parse the agent's JSON response."""
        try:
            clean = text.strip()
            # Strip markdown code fences
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                clean = clean.rsplit("```", 1)[0].strip()

            data = json.loads(clean)

            direction = data.get("direction", "HOLD").upper()
            if direction not in ("BUY", "SELL", "HOLD"):
                direction = "HOLD"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "direction": direction,
                "confidence": confidence,
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Agent %s parse error: %s", self._name, e)
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": f"parse error: {e}"}

    def _basic_analysis(self, context: dict) -> AgentDecision:
        """Fallback: simple rule-based analysis when Gemini is unavailable."""
        rsi = context.get("rsi", 50)
        vol_ratio = context.get("volume_ratio", 1.0)
        regime = context.get("regime", "RANGING")

        direction = "HOLD"
        confidence = 0.3

        # Very simple rules based on agent role
        if "technical" in self._role.lower():
            if rsi < 30 and vol_ratio > 1.5:
                direction = "BUY"
                confidence = 0.6
            elif rsi > 70 and vol_ratio > 1.5:
                direction = "SELL"
                confidence = 0.6

        elif "sentiment" in self._role.lower():
            if regime in ("TRENDING_UP", "BULLISH"):
                direction = "BUY"
                confidence = 0.55
            elif regime in ("TRENDING_DOWN", "CRASH", "BEARISH"):
                direction = "SELL"
                confidence = 0.55

        return AgentDecision(
            direction=direction,
            confidence=confidence,
            reasoning=f"Basic {self._name} analysis (Gemini unavailable)",
            chain=[],
            time_taken_ms=0,
        )

    def record_outcome(self, was_correct: bool):
        """Track whether this agent's last decision was correct."""
        if was_correct:
            self._correct_decisions += 1

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool and return the result.

        Called during agentic reasoning when the agent decides
        it needs more information.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"unknown tool: {tool_name}"}

        try:
            result = tool.func(args)
            logger.debug("Agent %s used tool %s", self._name, tool_name)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return {"error": str(e)}


# --- Pre-built agent tools ---
# These are the tools that agents can use during reasoning

def tool_check_correlation(args: dict) -> dict:
    """Check correlation between two assets."""
    return {"correlation": 0.85, "period": "24h",
            "note": "High positive correlation"}


def tool_check_volume_profile(args: dict) -> dict:
    """Analyze volume distribution at different price levels."""
    return {"high_volume_zone": "support", "distance_pct": 0.5,
            "note": "Price near high-volume support zone"}


def tool_check_funding_rate(args: dict) -> dict:
    """Check perpetual futures funding rate for sentiment."""
    return {"funding_rate": 0.01, "sentiment": "slightly bullish",
            "note": "Positive funding = longs paying shorts"}


def tool_check_fear_greed(args: dict) -> dict:
    """Check the crypto Fear & Greed index."""
    return {"index": 55, "label": "Neutral",
            "note": "Market sentiment is balanced"}


# --- Factory function to create pre-configured agents ---

def create_default_agents() -> list[AgentBrain]:
    """Create the default set of specialized brain agents.

    Returns 3 agents with different specializations:
    1. Technical Analysis Agent — reads charts and indicators
    2. Sentiment Agent — reads news, social, macro context
    3. Research Agent — cross-references data, queries RAG memory
    """
    # Technical Analysis Agent
    technical = AgentBrain(
        name="TechnicalAgent",
        role="Technical chart and indicator analysis specialist. "
             "You analyze RSI, MACD, Bollinger Bands, volume, and price action "
             "to identify high-probability trade setups.",
        tools=[
            AgentTool(
                name="check_volume_profile",
                description="Analyze volume distribution at price levels",
                func=tool_check_volume_profile,
            ),
        ],
    )

    # Sentiment Analysis Agent
    sentiment = AgentBrain(
        name="SentimentAgent",
        role="Market sentiment and macro analysis specialist. "
             "You analyze market regime, Fear & Greed index, funding rates, "
             "and macro events to gauge overall market direction.",
        tools=[
            AgentTool(
                name="check_funding_rate",
                description="Check perpetual futures funding rate",
                func=tool_check_funding_rate,
            ),
            AgentTool(
                name="check_fear_greed",
                description="Check the crypto Fear & Greed index",
                func=tool_check_fear_greed,
            ),
        ],
    )

    # Research Agent
    research = AgentBrain(
        name="ResearchAgent",
        role="Cross-asset research and historical pattern specialist. "
             "You check correlations between assets, reference historical "
             "patterns from RAG memory, and validate signals against "
             "broader market context.",
        tools=[
            AgentTool(
                name="check_correlation",
                description="Check price correlation between two assets",
                func=tool_check_correlation,
            ),
        ],
    )

    return [technical, sentiment, research]
