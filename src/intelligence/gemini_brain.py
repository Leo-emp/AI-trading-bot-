# src/intelligence/gemini_brain.py
# Gemini AI Brain — the smartest analyst in the room.
#
# Sends market data to Google's Gemini API and asks for analysis.
# This is Brain 3 of the 5-brain consensus system.
#
# What it analyzes:
# - Current price action and indicators
# - Market regime context
# - Recent price history pattern
# - Volume analysis
#
# Returns: BUY, SELL, or HOLD with confidence and reasoning.
#
# Rate limited: calls Gemini at most once every 15 minutes
# to stay within free tier (15 RPM on gemini-2.0-flash).
#
# SECURITY: API key loaded from environment variable only.

import os
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiBrain:
    """AI-powered market analysis using Google Gemini.

    Provides the "AI brain" signal for the 5-brain consensus.
    Lazy-loads the Gemini SDK on first use.
    """

    def __init__(self, min_interval_seconds: int = 900):
        # Minimum time between API calls (15 min default)
        self._min_interval = min_interval_seconds
        self._last_call_time = 0
        self._last_result: Optional[dict] = None
        self._model = None

    def _init_model(self):
        """Lazy-load the Gemini model. Only imports when first needed."""
        if self._model is not None:
            return True

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, AI brain disabled")
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Gemini AI brain initialized")
            return True
        except ImportError:
            logger.warning("google-generativeai not installed, AI brain disabled")
            return False
        except Exception as e:
            logger.error("Failed to initialize Gemini: %s", e)
            return False

    async def analyze(self, pair: str, current_price: float,
                      indicators: dict, regime: str,
                      recent_prices: list[float]) -> dict:
        """Ask Gemini to analyze current market conditions.

        Returns dict with: direction, confidence, reasoning
        Rate-limited to avoid exceeding API quota.
        """
        # Rate limiting — return cached result if too soon
        now = time.time()
        if now - self._last_call_time < self._min_interval and self._last_result:
            logger.debug("Gemini rate limited, using cached result")
            return self._last_result

        if not self._init_model():
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": "Gemini not available"}

        try:
            prompt = self._build_prompt(pair, current_price, indicators,
                                         regime, recent_prices)
            response = await self._call_gemini(prompt)
            result = self._parse_response(response)

            self._last_call_time = now
            self._last_result = result
            logger.info("Gemini analysis for %s: %s (%.2f confidence)",
                       pair, result["direction"], result["confidence"])
            return result

        except Exception as e:
            logger.error("Gemini analysis failed: %s", e)
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": f"analysis failed: {e}"}

    def _build_prompt(self, pair: str, price: float, indicators: dict,
                      regime: str, recent_prices: list[float]) -> str:
        """Build the analysis prompt for Gemini."""
        # Calculate simple trend from recent prices
        if len(recent_prices) >= 2:
            change_pct = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
        else:
            change_pct = 0

        return f"""You are a crypto trading analyst. Analyze this data and give a trading recommendation.

PAIR: {pair}
CURRENT PRICE: ${price:.2f}
PRICE CHANGE (recent): {change_pct:+.2f}%
MARKET REGIME: {regime}

INDICATORS:
- RSI: {indicators.get('rsi', 'N/A')}
- MACD Histogram: {indicators.get('macd_histogram', 'N/A')}
- Bollinger Band Width: {indicators.get('bb_width', 'N/A')}
- Volume Ratio: {indicators.get('volume_ratio', 'N/A')}
- EMA Fast vs Slow: {indicators.get('ema_fast', 'N/A')} vs {indicators.get('ema_slow', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}

RECENT PRICES (last 10): {recent_prices[-10:] if len(recent_prices) >= 10 else recent_prices}

RULES:
- This is for a small account ($20-100). Capital preservation is #1 priority.
- Only recommend BUY or SELL if you are at least 60% confident.
- Consider fees: 0.15% round trip. Net profit must exceed fees.
- If uncertain, recommend HOLD.

Respond in EXACTLY this JSON format, nothing else:
{{"direction": "BUY" or "SELL" or "HOLD", "confidence": 0.0 to 1.0, "reasoning": "one sentence explanation"}}"""

    async def _call_gemini(self, prompt: str) -> str:
        """Call the Gemini API. Runs sync call in thread to avoid blocking."""
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self._model.generate_content(prompt)
        )
        return response.text

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini's JSON response. Falls back to HOLD on parse error."""
        try:
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()

            data = json.loads(clean)

            direction = data.get("direction", "HOLD").upper()
            if direction not in ("BUY", "SELL", "HOLD"):
                direction = "HOLD"

            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(data.get("reasoning", "no reasoning provided"))

            return {"direction": direction, "confidence": confidence,
                    "reasoning": reasoning}

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse Gemini response: %s | Raw: %s", e, text[:200])
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": f"parse error: {e}"}
