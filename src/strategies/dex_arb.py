# src/strategies/dex_arb.py
# DEX Arbitrage Scanner (Phase 8) — cross-pool price difference trading.
#
# Monitors Solana DEX pools for the same token at different prices:
# - Jupiter aggregator for best route pricing
# - Raydium AMM pools for direct pool prices
# - Orca Whirlpools for concentrated liquidity pricing
#
# Arbitrage flow:
# 1. Scan all pools for the same token pair
# 2. Find price differences > gas + slippage threshold
# 3. Execute atomic swap: buy cheap on Pool A → sell on Pool B
# 4. Profit = price gap - gas fees - slippage
#
# Requirements:
# - Solana wallet with SOL for gas
# - RPC endpoint (free tier: ~25 req/sec)
# - Fast execution (MEV competition is fierce on Solana)
#
# SECURITY: Private key loaded from env var, never hardcoded.
# This module is NOT active until explicitly enabled in config.

import os
import json
import time
import logging
import asyncio
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum profit to execute an arb (must exceed gas + slippage)
MIN_PROFIT_BPS = 30  # 0.30% minimum profit after all costs
# Solana transaction fee (approximately)
SOLANA_TX_FEE_SOL = 0.000005
# Maximum slippage tolerance
MAX_SLIPPAGE_BPS = 50  # 0.50%
# Scan interval between arb checks
SCAN_INTERVAL_SECONDS = 2


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity between two DEX pools."""
    token_pair: str            # e.g. "SOL/USDC"
    buy_pool: str              # where to buy cheap (e.g. "raydium")
    sell_pool: str             # where to sell expensive (e.g. "orca")
    buy_price: float           # price on buy pool
    sell_price: float          # price on sell pool
    spread_bps: float          # price difference in basis points
    estimated_profit_usd: float  # after gas + slippage
    max_size_usd: float        # max trade size before impact
    timestamp: float


@dataclass
class ArbResult:
    """Result of an executed arbitrage trade."""
    opportunity: ArbOpportunity
    executed: bool
    actual_profit_usd: float = 0.0
    gas_cost_usd: float = 0.0
    slippage_bps: float = 0.0
    tx_signature: str = ""
    error: str = ""


class DexArbScanner:
    """Scans Solana DEX pools for cross-pool arbitrage opportunities.

    This scanner compares prices across Jupiter, Raydium, and Orca
    to find profitable atomic swap opportunities.
    """

    def __init__(self):
        # Solana RPC client (lazy-loaded)
        self._rpc_client = None
        # Wallet keypair (from env)
        self._wallet = None
        # Whether the scanner is ready
        self._available = False
        # Token pairs to scan
        self._scan_pairs = [
            "SOL/USDC", "SOL/USDT",
            "BONK/SOL", "JTO/SOL",
            "WIF/SOL", "JUP/SOL",
        ]
        # Pool price caches
        self._jupiter_prices: dict = {}
        self._raydium_prices: dict = {}
        self._orca_prices: dict = {}
        # Stats
        self._total_scans = 0
        self._opportunities_found = 0
        self._trades_executed = 0
        self._total_profit = 0.0

    def initialize(self) -> bool:
        """Set up Solana connection and wallet.

        Requires:
        - SOLANA_RPC_URL env var (e.g. https://api.mainnet-beta.solana.com)
        - SOLANA_PRIVATE_KEY env var (base58 encoded)
        """
        rpc_url = os.getenv("SOLANA_RPC_URL", "")
        private_key = os.getenv("SOLANA_PRIVATE_KEY", "")

        if not rpc_url:
            logger.info("DEX arb disabled: SOLANA_RPC_URL not set")
            return False

        if not private_key:
            logger.info("DEX arb disabled: SOLANA_PRIVATE_KEY not set")
            return False

        try:
            from solders.keypair import Keypair
            from solana.rpc.async_api import AsyncClient

            # Initialize RPC client
            self._rpc_client = AsyncClient(rpc_url)

            # Load wallet from private key
            self._wallet = Keypair.from_base58_string(private_key)
            logger.info("DEX arb scanner initialized. Wallet: %s",
                       str(self._wallet.pubkey())[:8] + "...")
            self._available = True
            return True

        except ImportError:
            logger.warning("DEX arb disabled: solders/solana-py not installed. "
                          "Install with: pip install solders solana")
            return False
        except Exception as e:
            logger.error("DEX arb init failed: %s", e)
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    async def scan_opportunities(self) -> list[ArbOpportunity]:
        """Scan all DEX pools for arbitrage opportunities.

        Compares prices across Jupiter, Raydium, and Orca
        for each monitored token pair.

        Returns list of profitable opportunities sorted by profit.
        """
        if not self._available:
            return []

        self._total_scans += 1
        opportunities = []

        for pair in self._scan_pairs:
            try:
                # Fetch prices from all three DEXes in parallel
                prices = await self._fetch_all_prices(pair)

                if len(prices) < 2:
                    continue  # need at least 2 pools to compare

                # Compare all pool pairs for arb
                pool_names = list(prices.keys())
                for i in range(len(pool_names)):
                    for j in range(i + 1, len(pool_names)):
                        pool_a = pool_names[i]
                        pool_b = pool_names[j]
                        price_a = prices[pool_a]
                        price_b = prices[pool_b]

                        if price_a <= 0 or price_b <= 0:
                            continue

                        # Calculate spread in basis points
                        spread_bps = abs(price_a - price_b) / min(price_a, price_b) * 10000

                        if spread_bps < MIN_PROFIT_BPS:
                            continue  # not profitable enough

                        # Determine buy/sell direction
                        if price_a < price_b:
                            buy_pool, sell_pool = pool_a, pool_b
                            buy_price, sell_price = price_a, price_b
                        else:
                            buy_pool, sell_pool = pool_b, pool_a
                            buy_price, sell_price = price_b, price_a

                        # Estimate profit after costs
                        estimated_profit = self._estimate_profit(
                            buy_price, sell_price, trade_size_usd=50.0
                        )

                        if estimated_profit > 0:
                            opp = ArbOpportunity(
                                token_pair=pair,
                                buy_pool=buy_pool,
                                sell_pool=sell_pool,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                spread_bps=round(spread_bps, 2),
                                estimated_profit_usd=round(estimated_profit, 4),
                                max_size_usd=50.0,
                                timestamp=time.time(),
                            )
                            opportunities.append(opp)
                            self._opportunities_found += 1

            except Exception as e:
                logger.debug("Error scanning %s: %s", pair, e)

        # Sort by estimated profit (highest first)
        opportunities.sort(key=lambda o: o.estimated_profit_usd, reverse=True)

        if opportunities:
            logger.info(
                "Found %d arb opportunities. Best: %s %.2f bps ($%.4f profit)",
                len(opportunities),
                opportunities[0].token_pair,
                opportunities[0].spread_bps,
                opportunities[0].estimated_profit_usd,
            )

        return opportunities

    async def execute_arb(self, opportunity: ArbOpportunity) -> ArbResult:
        """Execute an arbitrage trade.

        Buys on the cheap pool, sells on the expensive pool.
        Uses Jupiter aggregator for best execution.

        This is a placeholder — real implementation needs:
        - Jupiter swap API integration
        - Transaction building with compute budget
        - Priority fee for faster inclusion
        - Slippage protection
        """
        if not self._available:
            return ArbResult(opportunity=opportunity, executed=False,
                           error="scanner not available")

        try:
            logger.info(
                "Executing arb: %s — buy on %s ($%.4f) → sell on %s ($%.4f) — %.2f bps",
                opportunity.token_pair,
                opportunity.buy_pool, opportunity.buy_price,
                opportunity.sell_pool, opportunity.sell_price,
                opportunity.spread_bps,
            )

            # TODO: Implement actual Solana transaction
            # 1. Build Jupiter swap instruction (buy side)
            # 2. Build sell instruction on target pool
            # 3. Bundle into single transaction (atomic)
            # 4. Add compute budget and priority fee
            # 5. Sign and send transaction
            # 6. Confirm and check actual fill prices

            # For now, log the opportunity
            self._trades_executed += 1

            return ArbResult(
                opportunity=opportunity,
                executed=False,
                error="DEX execution not yet implemented — logging opportunity",
            )

        except Exception as e:
            logger.error("Arb execution failed: %s", e)
            return ArbResult(opportunity=opportunity, executed=False,
                           error=str(e))

    async def _fetch_all_prices(self, pair: str) -> dict:
        """Fetch prices from all DEX sources for a token pair.

        Returns dict mapping pool name → price.
        """
        prices = {}

        # Fetch in parallel from all sources
        tasks = [
            self._fetch_jupiter_price(pair),
            self._fetch_raydium_price(pair),
            self._fetch_orca_price(pair),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        if isinstance(results[0], (int, float)) and results[0] > 0:
            prices["jupiter"] = results[0]
        if isinstance(results[1], (int, float)) and results[1] > 0:
            prices["raydium"] = results[1]
        if isinstance(results[2], (int, float)) and results[2] > 0:
            prices["orca"] = results[2]

        return prices

    async def _fetch_jupiter_price(self, pair: str) -> float:
        """Fetch price from Jupiter aggregator API."""
        try:
            # Jupiter V6 Quote API
            # In production: use httpx to call Jupiter's API
            # https://quote-api.jup.ag/v6/quote?inputMint=...&outputMint=...
            return 0.0  # placeholder
        except Exception:
            return 0.0

    async def _fetch_raydium_price(self, pair: str) -> float:
        """Fetch price from Raydium AMM pool."""
        try:
            # Raydium SDK or direct pool account deserialization
            return 0.0  # placeholder
        except Exception:
            return 0.0

    async def _fetch_orca_price(self, pair: str) -> float:
        """Fetch price from Orca Whirlpool."""
        try:
            # Orca Whirlpool SDK
            return 0.0  # placeholder
        except Exception:
            return 0.0

    def _estimate_profit(self, buy_price: float, sell_price: float,
                         trade_size_usd: float) -> float:
        """Estimate net profit after gas and slippage."""
        if buy_price <= 0:
            return 0.0

        # Gross profit from price difference
        gross_profit_pct = (sell_price - buy_price) / buy_price
        gross_profit_usd = trade_size_usd * gross_profit_pct

        # Estimated costs
        gas_cost_usd = 0.01    # ~0.000005 SOL × $150/SOL × 2 txs
        slippage_cost = trade_size_usd * (MAX_SLIPPAGE_BPS / 10000 / 2)  # half of max

        net_profit = gross_profit_usd - gas_cost_usd - slippage_cost
        return net_profit

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            "total_scans": self._total_scans,
            "opportunities_found": self._opportunities_found,
            "trades_executed": self._trades_executed,
            "total_profit_usd": round(self._total_profit, 4),
            "scan_pairs": self._scan_pairs,
            "available": self._available,
        }

    async def run_scanner_loop(self, on_opportunity=None):
        """Run continuous scanning loop.

        Scans every SCAN_INTERVAL_SECONDS and calls on_opportunity
        callback when a profitable arb is found.
        """
        logger.info("Starting DEX arb scanner loop (%ds interval)",
                   SCAN_INTERVAL_SECONDS)

        while True:
            try:
                opportunities = await self.scan_opportunities()

                if opportunities and on_opportunity:
                    # Execute the best opportunity
                    best = opportunities[0]
                    result = await self.execute_arb(best)
                    if on_opportunity:
                        await on_opportunity(result)

                await asyncio.sleep(SCAN_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info("DEX arb scanner stopped")
                break
            except Exception as e:
                logger.error("Scanner loop error: %s", e)
                await asyncio.sleep(10)
