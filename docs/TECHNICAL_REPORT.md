# Multi-Agent AI Trading System: Technical Report

## Problem Statement

Cryptocurrency markets operate 24/7 with extreme volatility, making manual trading impractical for retail investors. Existing automated solutions suffer from three core limitations:

1. **Single-signal dependency** — most bots use one indicator (e.g., RSI crossover) which fails when market conditions change
2. **No memory** — bots analyze each moment in isolation, unable to learn from their own trading history
3. **Brittle risk management** — a single stop-loss layer cannot protect against cascading failures (flash crashes, API outages, liquidity crises)

This project addresses all three by combining multiple AI/ML techniques into a unified decision pipeline with autonomous protection.

## Approach

### Multi-Brain Consensus Architecture

Rather than relying on a single signal source, the system implements a **5-brain voting mechanism** inspired by ensemble methods in machine learning. Each brain analyzes market conditions independently:

| Brain | Input | Method | Output |
|-------|-------|--------|--------|
| Technical Signals | OHLCV + indicators | Rule-based (RSI, MACD, BB) | BUY/SELL/HOLD + confidence |
| Order Flow | Order book depth | Statistical (imbalance ratio) | Direction + whale detection |
| AI Sentiment | Price + indicators + history | Gemini LLM + RAG context | Direction + reasoning |
| Multi-Timeframe | 5m, 15m, 1h candles | Cross-timeframe alignment | Agree/disagree signal |
| Cross-Asset | BTC, ETH, altcoin prices | Correlation tracking | Systemic risk signal |

A trade executes only when **4 of 5 brains agree** on direction, AND no brain shows a strong opposing signal (confidence > 0.65). This eliminates low-conviction trades and dramatically improves win rate at the cost of trade frequency.

### Retrieval-Augmented Generation (RAG)

The system maintains a **vector database** (ChromaDB) of every completed trade alongside its full market context. When analyzing a new trading opportunity:

1. The current market state is encoded as a **16-dimensional embedding vector** using handcrafted features (RSI, MACD, volume, regime, price momentum, order book imbalance)
2. ChromaDB performs **cosine similarity search** to find the 5 most similar historical scenarios
3. These historical outcomes are injected into the Gemini LLM prompt as context
4. Gemini analyzes current conditions informed by "what happened last time conditions looked like this"

This transforms the LLM from a stateless analyzer into one with **experiential memory** — the system literally learns from its own trading history without requiring model retraining.

**Embedding design choice:** We use handcrafted numerical feature vectors rather than transformer-based text embeddings. For structured financial data, domain-specific features (RSI normalized to [-1,1], log-scaled volume ratios, cyclical time encoding) outperform general-purpose text embeddings while running on minimal hardware (1 GB RAM server).

### Machine Learning Pipeline

An XGBoost binary classifier predicts whether a trade signal will be profitable, using a **30-dimensional feature vector**:

- **Price features (7):** Multi-window returns, momentum at 5/10/20 periods, price vs EMA distance
- **Volume features (4):** Volume ratio, trend, spike detection, consistency
- **Indicator features (8):** RSI + slope, MACD + crossover, Bollinger position + width, ATR%, ADX
- **Order book features (3):** Imbalance, spread, whale pressure
- **Regime features (3):** Encoded regime, confidence, stability
- **RAG features (3):** Historical win rate, average P&L, similarity score from similar scenarios
- **Time features (2):** Cyclical hour encoding (sin/cos to preserve circular nature)

Training uses **walk-forward validation**: train on the oldest 80% of data, validate on the newest 20%. This prevents look-ahead bias that plagues most backtesting. The model only deploys if validation accuracy exceeds 55%.

**Drift detection:** After deployment, the system tracks live prediction accuracy. If it drops more than 5% below the training baseline, the model auto-reverts to rule-based signals and flags for retraining.

### Agentic AI Layer

Three specialized AI agents reason autonomously about market conditions:

1. **Technical Agent** — analyzes chart patterns and indicator setups
2. **Sentiment Agent** — evaluates market regime, funding rates, fear/greed index
3. **Research Agent** — cross-references assets, queries RAG memory for historical patterns

Each agent:
- Has access to domain-specific **tools** (API queries, calculations, database lookups)
- Produces a **chain-of-thought reasoning** trace (logged for analysis)
- Votes independently with a confidence score
- Is **weighted by historical accuracy** — agents that have been more correct get more voting power

The **orchestrator** runs all agents in parallel (time-boxed to 15 seconds), collects votes, applies accuracy weighting, and produces a consensus decision.

### 5-Layer Autonomous Protection

The system operates without human intervention through layered defense:

| Layer | Scope | Response Time | Action |
|-------|-------|---------------|--------|
| L1: Per-Trade | Single position | Milliseconds | Hard stop-loss, trailing stop, time-based exit |
| L2: Session | Trading session | Minutes | Reduce size after 3 losses, pause after 5 |
| L3: Portfolio | Weekly/monthly | Hours | Scale down on drawdown, switch to defense mode |
| L4: Black Swan | Market-wide | Instant | Exit all positions on >5% move in 1 minute |
| L5: Infrastructure | System | Automatic | Auto-reconnect, crash recovery, graceful degradation |

Each layer operates independently — a failure in one does not compromise others. The system degrades gracefully: if Gemini API is down, it continues with technical signals only. If WebSocket disconnects, it falls back to REST polling. If the process crashes, systemd auto-restarts within 30 seconds.

## Results

### Backtesting Performance

Tested on simulated market data across 5 conditions (normal, bull, bear, volatile, sideways) with $100 starting balance and realistic Binance fee simulation:

| Metric | Value |
|--------|-------|
| Strategies tested | 4 |
| Market seeds | 5 |
| Pass rate | 19/20 (95%) |
| Average P&L per strategy | $4.00 - $5.70 |
| Average win rate | 45% - 60% |
| Maximum drawdown | 0.7% - 1.1% |
| Fee model | Binance maker 0.075% (with BNB discount) |
| Order type | Limit only (never market) |

### System Specifications

| Metric | Value |
|--------|-------|
| Total codebase | 16,800+ lines of Python |
| Source files | 51 Python modules |
| Test files | 18 |
| Trading pairs | 12 (BTC, ETH, SOL, XRP, DOGE, BNB, ADA, AVAX, POL, LINK, DOT, NEAR) |
| Cycle interval | 30 seconds |
| Timeframes analyzed | 5m, 15m, 1h |
| Feature vector dimensions | 30 (ML) + 16 (embedding) |
| Deployment | Oracle Cloud Free Tier (Ubuntu 22.04, systemd) |
| Uptime target | 24/7 continuous operation |

## Conclusions

The multi-brain consensus approach eliminates the single-signal failure mode that plagues most trading bots. By requiring agreement from independent analysis methods (technical, order flow, AI sentiment, multi-timeframe, cross-asset), the system only trades on high-conviction opportunities.

RAG memory provides a practical alternative to model retraining — the system learns from its own history without GPU compute or training pipelines. This is particularly valuable for small-scale deployment where compute resources are limited.

The 5-layer protection system ensures autonomous operation. In 6+ hours of initial deployment, the system correctly identified market regimes, generated Gemini AI analysis, and made conservative HOLD decisions during low-conviction periods — exactly the intended behavior for capital preservation.

## Future Work

- **Phase 7:** Distill Gemini's reasoning chains into a smaller fine-tuned model (Gemma 2B) for faster, cheaper inference
- **Phase 8:** Solana DEX cross-pool arbitrage scanner (Jupiter, Raydium, Orca)
- **Performance dashboard:** Real-time web interface showing live P&L, trade history, and agent reasoning chains
- **30-day live results:** Comprehensive metrics report after paper trading validation period
