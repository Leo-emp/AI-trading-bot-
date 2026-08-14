# src/ai/rag_memory.py
# RAG Memory System — the bot's long-term market memory.
#
# Uses ChromaDB (lightweight vector database) to store every trade
# outcome alongside the full market context that led to it.
#
# When Gemini analyzes a new situation, RAG retrieves the 5 most
# similar historical scenarios and includes them in the prompt:
# "Last 5 times conditions looked like this, here's what happened"
#
# This transforms Gemini from a stateless analyzer into one with
# memory — it learns from the bot's own trading history.
#
# Storage: each trade becomes a "document" in ChromaDB with:
# - embedding: numerical market state vector (from EmbeddingEngine)
# - metadata: pair, regime, indicators, outcome, P&L
# - document: human-readable summary for Gemini's context
#
# Retrieval: given current market state, find the K nearest neighbors
# and format them as context for Gemini's analysis prompt.

import json
import logging
import os
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RAGMemory:
    """Vector-based trade memory using ChromaDB.

    Stores trade outcomes with their market context as embeddings.
    Retrieves similar historical scenarios for AI-enhanced decisions.
    """

    def __init__(self, persist_dir: str = "data/rag_memory",
                 collection_name: str = "trade_outcomes"):
        # Where ChromaDB stores its data on disk
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        # ChromaDB client and collection (lazy-loaded)
        self._client = None
        self._collection = None
        # Track how many memories we've stored
        self._memory_count = 0
        # Whether ChromaDB is available
        self._available = False

    def initialize(self):
        """Set up ChromaDB. Safe to call if chromadb isn't installed."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Use persistent storage so memories survive restarts
            os.makedirs(self._persist_dir, exist_ok=True)

            # Create ChromaDB client with persistent storage
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create the collection for trade memories
            # Using cosine similarity — best for normalized feature vectors
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._memory_count = self._collection.count()
            self._available = True
            logger.info(
                "RAG memory initialized: %d memories in %s",
                self._memory_count, self._persist_dir,
            )

        except ImportError:
            logger.warning("chromadb not installed — RAG memory disabled. "
                          "Install with: pip install chromadb")
            self._available = False
        except Exception as e:
            logger.error("Failed to initialize RAG memory: %s", e)
            self._available = False

    @property
    def is_available(self) -> bool:
        """Whether RAG memory is ready to use."""
        return self._available

    @property
    def memory_count(self) -> int:
        """How many trade memories are stored."""
        if self._collection:
            return self._collection.count()
        return 0

    def store_trade(self, trade_id: str, embedding: list[float],
                    metadata: dict, summary: str):
        """Store a completed trade with its market context.

        Args:
            trade_id: unique identifier (e.g. "BTC_USDT_20260815_143022")
            embedding: market state vector from EmbeddingEngine
            metadata: dict with pair, regime, indicators, outcome, pnl
            summary: human-readable description for Gemini context
        """
        if not self._available:
            return

        try:
            # ChromaDB metadata only supports str, int, float, bool
            # Convert any complex types to strings
            clean_meta = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)

            # Add to collection
            self._collection.add(
                ids=[trade_id],
                embeddings=[embedding],
                metadatas=[clean_meta],
                documents=[summary],
            )

            self._memory_count += 1
            logger.debug(
                "Stored trade memory: %s (total: %d)",
                trade_id, self._memory_count,
            )

        except Exception as e:
            logger.error("Failed to store trade memory: %s", e)

    def retrieve_similar(self, current_embedding: list[float],
                         n_results: int = 5,
                         pair_filter: Optional[str] = None) -> list[dict]:
        """Find the N most similar historical market conditions.

        Args:
            current_embedding: current market state vector
            n_results: how many similar scenarios to retrieve
            pair_filter: optional — only retrieve from same pair

        Returns:
            List of dicts with: id, similarity, metadata, summary
        """
        if not self._available or self._memory_count == 0:
            return []

        try:
            # Build query filters
            where_filter = None
            if pair_filter:
                where_filter = {"pair": pair_filter}

            # Query ChromaDB for nearest neighbors
            results = self._collection.query(
                query_embeddings=[current_embedding],
                n_results=min(n_results, self._memory_count),
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )

            # Format results into clean dicts
            memories = []
            if results and results["ids"] and results["ids"][0]:
                for i, trade_id in enumerate(results["ids"][0]):
                    # ChromaDB returns distance (lower = more similar)
                    # Convert to similarity score (higher = more similar)
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # cosine distance → similarity

                    memories.append({
                        "id": trade_id,
                        "similarity": round(similarity, 4),
                        "metadata": results["metadatas"][0][i],
                        "summary": results["documents"][0][i],
                    })

            logger.debug(
                "Retrieved %d similar memories (top similarity: %.3f)",
                len(memories),
                memories[0]["similarity"] if memories else 0,
            )
            return memories

        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            return []

    def format_for_gemini(self, memories: list[dict]) -> str:
        """Format retrieved memories into a context block for Gemini's prompt.

        Takes the similar historical scenarios and creates a structured
        text block that Gemini can use to make better-informed decisions.
        """
        if not memories:
            return ""

        lines = ["HISTORICAL CONTEXT (similar past scenarios):"]
        lines.append("=" * 50)

        for i, mem in enumerate(memories, 1):
            meta = mem["metadata"]
            outcome = meta.get("outcome", "unknown")
            pnl = meta.get("pnl_pct", "?")
            regime = meta.get("regime", "?")
            strategy = meta.get("strategy", "?")
            similarity = mem["similarity"]

            # Color-code outcomes for clarity
            outcome_label = "WIN" if outcome == "WIN" else "LOSS"

            lines.append(
                f"\nScenario {i} (similarity: {similarity:.1%}):"
            )
            lines.append(f"  Outcome: {outcome_label} | P&L: {pnl}%")
            lines.append(f"  Regime: {regime} | Strategy: {strategy}")
            lines.append(f"  Context: {mem['summary']}")

        lines.append("=" * 50)

        # Add instruction for Gemini
        lines.append(
            "\nUse these historical outcomes to inform your analysis. "
            "If most similar scenarios resulted in losses, be more cautious. "
            "If most were wins, consider the pattern that led to success."
        )

        return "\n".join(lines)

    def get_win_rate_for_similar(self, embedding: list[float],
                                 n_results: int = 10,
                                 pair: Optional[str] = None) -> dict:
        """Compute win rate from similar historical scenarios.

        Returns a quick statistical summary without needing Gemini.
        Useful as a standalone confidence signal.
        """
        memories = self.retrieve_similar(embedding, n_results, pair)
        if not memories:
            return {"win_rate": 0.5, "sample_size": 0, "avg_pnl": 0.0}

        wins = sum(1 for m in memories if m["metadata"].get("outcome") == "WIN")
        total = len(memories)
        avg_pnl = sum(
            float(m["metadata"].get("pnl_pct", 0)) for m in memories
        ) / total

        return {
            "win_rate": wins / total,
            "sample_size": total,
            "avg_pnl": round(avg_pnl, 4),
            "avg_similarity": round(
                sum(m["similarity"] for m in memories) / total, 4
            ),
        }

    def store_market_snapshot(self, snapshot_id: str,
                              embedding: list[float],
                              metadata: dict):
        """Store a market snapshot even without a trade outcome.

        These 'observation' memories help build a richer context
        even when the bot doesn't trade. Useful for learning what
        conditions looked like when the bot correctly chose NOT to trade.
        """
        if not self._available:
            return

        try:
            clean_meta = {
                k: v if isinstance(v, (str, int, float, bool)) else str(v)
                for k, v in metadata.items()
            }
            clean_meta["type"] = "observation"  # distinguish from trades

            self._collection.add(
                ids=[snapshot_id],
                embeddings=[embedding],
                metadatas=[clean_meta],
                documents=[f"Market observation: {metadata.get('pair', '?')} "
                          f"at {metadata.get('price', '?')} in "
                          f"{metadata.get('regime', '?')} regime"],
            )

        except Exception as e:
            logger.debug("Failed to store snapshot: %s", e)
