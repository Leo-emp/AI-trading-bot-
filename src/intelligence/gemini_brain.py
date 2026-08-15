# src/intelligence/gemini_brain.py
# Gemini AI Brain — the smartest analyst in the room.
#
# Sends market data to Google's Gemini API and asks for analysis.
# This is Brain 3 of the 5-brain consensus system.
#
# Per-pair caching: each pair gets its own cached result with a 3-minute
# cooldown. Gemini 2.5 Flash free tier allows 10 RPM — with 12 pairs
# on a 3-minute per-pair cooldown, we average ~4 RPM (safe headroom).
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
    Per-pair caching prevents stale cross-pair analysis contamination.
    """

    def __init__(self, min_interval_seconds: int = 600):
        # Minimum time between API calls PER PAIR (10 min default)
        self._min_interval = min_interval_seconds
        # Per-pair cache: {"BTC/USDT": {"time": ..., "result": ...}}
        self._cache: dict[str, dict] = {}
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
            # gemini-2.5-flash-lite: lighter model with higher free tier quota
            self._model = genai.GenerativeModel("gemini-2.5-flash-lite")
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
                      recent_prices: list[float],
                      rag_context: str = "") -> dict:
        """Ask Gemini to analyze current market conditions.

        Returns dict with: direction, confidence, reasoning
        Per-pair rate limiting to avoid exceeding API quota.
        """
        now = time.time()

        # Per-pair rate limiting — return cached result for THIS pair if too soon
        cached = self._cache.get(pair)
        if cached and (now - cached["time"] < self._min_interval):
            return cached["result"]

        if not self._init_model():
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": "Gemini not available"}

        try:
            prompt = self._build_prompt(pair, current_price, indicators,
                                         regime, recent_prices, rag_context)
            response = await self._call_gemini(prompt)
            result = self._parse_response(response)

            # Cache per pair
            self._cache[pair] = {"time": now, "result": result}
            logger.info("Gemini analysis for %s: %s (%.2f confidence) — %s",
                       pair, result["direction"], result["confidence"],
                       result["reasoning"][:80])
            return result

        except Exception as e:
            logger.error("Gemini analysis failed for %s: %s", pair, e)
            return {"direction": "HOLD", "confidence": 0.0,
                    "reasoning": f"analysis failed: {e}"}

    def _build_prompt(self, pair: str, price: float, indicators: dict,
                      regime: str, recent_prices: list[float],
                      rag_context: str = "") -> str:
        """Build the analysis prompt for Gemini."""
        if len(recent_prices) >= 2:
            change_pct = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
        else:
            change_pct = 0

        history_section = ""
        if rag_context:
            history_section = f"\n{rag_context}\n"

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
{history_section}
RULES:
- This is for a small account ($20-100). Capital preservation is #1 priority.
- Only recommend BUY or SELL if you are at least 60% confident.
- Consider fees: 0.15% round trip. Net profit must exceed fees.
- If historical context shows similar scenarios mostly lost money, be extra cautious.
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
