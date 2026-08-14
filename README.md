# Multi-Agent AI Trading System

An autonomous cryptocurrency trading system that combines **6 AI/ML techniques** into a single decision pipeline: LLM analysis (Gemini), XGBoost ML signals, Retrieval-Augmented Generation (RAG), market state embeddings, multi-agent consensus, and self-learning parameter optimization.

Deployed 24/7 on Oracle Cloud, trading 12 pairs on Binance with 5-layer autonomous protection.

---

## Architecture

```
                              ┌──────────────────────────┐
                              │     TRADE GATE           │
                              │  (4/5 consensus + veto)  │
                              └────────────┬─────────────┘
               ┌──────────┬───────┬────────┼────────┬────────────┐
               ▼          ▼       ▼        ▼        ▼            ▼
          ┌─────────┐ ┌───────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌───────┐
          │Strategy │ │Order  │ │Gemini│ │Multi-│ │Cross- │ │  ML   │
          │Signal   │ │Book   │ │AI +  │ │Time- │ │Asset  │ │Signal │
          │(4 strats)│ │Flow  │ │RAG   │ │frame │ │Corr.  │ │(XGB)  │
          └─────────┘ └───────┘ └──────┘ └──────┘ └───────┘ └───────┘
           Brain 1    Brain 2   Brain 3  Brain 4   Brain 5   Brain 6
                                   │
                              ┌────┴────┐
                              │ChromaDB │
                              │(vector  │
                              │ memory) │
                              └─────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │               5-LAYER PROTECTION SYSTEM                     │
    │  L1: Per-trade (SL/TP/trailing/time-exit)                  │
    │  L2: Session (consecutive loss pause, drawdown defense)     │
    │  L3: Portfolio (weekly/monthly drawdown limits)              │
    │  L4: Black swan (flash crash instant exit)                  │
    │  L5: Infrastructure (auto-reconnect, self-healing)          │
    └─────────────────────────────────────────────────────────────┘
```

## AI/ML Techniques Used

| # | Technique | Implementation | Purpose |
|---|-----------|---------------|---------|
| 1 | **LLM Integration** | Gemini 2.5 Flash API | Market regime analysis, sentiment, reasoning |
| 2 | **Machine Learning** | XGBoost classifier, 30 features | Predict trade profitability from indicators |
| 3 | **RAG** | ChromaDB + custom embeddings | Retrieve similar historical trades to augment Gemini |
| 4 | **Embeddings** | 16-dim market state vectors | Encode market snapshots for similarity search |
| 5 | **Multi-Agent AI** | 3 specialized agents + orchestrator | Parallel reasoning with accuracy-weighted voting |
| 6 | **Self-Learning** | Rolling 50-trade scoring + adaptive tuning | Auto-disable losing strategies, adjust parameters |

## How It Works

### Decision Pipeline (every 30 seconds)

1. **Fetch Data** — OHLCV candles from Binance for 12 pairs across 3 timeframes
2. **Compute Indicators** — RSI, MACD, Bollinger Bands, ATR, EMA, volume profiles
3. **Detect Regime** — classify market as trending/ranging/volatile/crash
4. **Select Strategy** — route to the best strategy for current conditions
5. **Collect Brain Signals** — 5 independent analysis layers vote on direction
6. **RAG Retrieval** — find 5 most similar historical scenarios from ChromaDB
7. **Gemini Analysis** — LLM analyzes with current data + historical context
8. **Trade Gate** — require 4/5 brain consensus, veto on strong opposing signals
9. **Risk Validation** — position sizing (quarter-Kelly), balance checks, limits
10. **Execute** — limit orders only (never market orders), maker fees

### Trading Strategies

| Strategy | Market Condition | Win Rate | R:R |
|----------|-----------------|----------|-----|
| Smart Scalp | All regimes (background) | 56-60% | 3:1 |
| Grid Trading | Ranging/sideways | 56-60% | 3:1 |
| Trend Momentum | Trending markets | 45-55% | 4:1 |
| Mean Reversion | Extreme moves | 55-60% | 3:1 |

### Backtest Results

Tested across 5 market conditions (normal, bull, bear, volatile, sideways):

| Strategy | Avg P&L | Win Rate | Max Drawdown | Pass Rate |
|----------|---------|----------|--------------|-----------|
| Volume Breakout | $5.70 | 60.0% | 0.7% | 5/5 |
| Vol Squeeze | $5.29 | 56.6% | 1.0% | 4/5 |
| Trend Momentum | $5.00 | 45.3% | 1.1% | 5/5 |
| Trend Pullback | $4.54 | 56.6% | 1.0% | 5/5 |

**Overall: 19/20 strategy-seed combinations pass** ($100 starting balance, fees included)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ (async/await) |
| Exchange | Binance via ccxt (REST + WebSocket) |
| LLM | Google Gemini 2.5 Flash |
| ML | XGBoost, scikit-learn |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Database | SQLite (WAL mode, async via aiosqlite) |
| Indicators | ta (RSI, MACD, BB, ATR, ADX, EMA) |
| Deployment | Oracle Cloud (Ubuntu 22.04, systemd) |
| Monitoring | Telegram alerts, structured logging |

## Project Structure

```
ai-trading-bot/                    # 16,800+ lines of Python
├── src/
│   ├── core/
│   │   ├── engine.py              # Main trading loop (585 lines)
│   │   ├── event_bus.py           # Async event system
│   │   ├── config.py              # YAML config loader
│   │   └── fast_engine.py         # Low-latency engine variant
│   ├── ai/                        # AI/ML Layer
│   │   ├── trade_gate.py          # Multi-brain consensus (4/5 vote)
│   │   ├── rag_memory.py          # ChromaDB RAG memory system
│   │   ├── embeddings.py          # Market state vector encoding
│   │   ├── ml_model.py            # XGBoost training + prediction
│   │   ├── ml_features.py         # 30-feature engineering pipeline
│   │   ├── agent_brain.py         # Autonomous brain agents
│   │   └── agent_orchestrator.py  # Multi-agent coordination
│   ├── strategies/                # Trading Strategies
│   │   ├── smart_scalp.py         # Fee-aware micro-scalping
│   │   ├── grid.py                # Volatility squeeze grid
│   │   ├── momentum.py            # Trend following (ADX + EMA)
│   │   ├── mean_reversion.py      # RSI extreme bounce
│   │   └── dex_arb.py             # Solana DEX arbitrage (Phase 8)
│   ├── intelligence/              # Market Intelligence
│   │   ├── gemini_brain.py        # Gemini LLM analysis + RAG
│   │   ├── market_regime.py       # Regime detection
│   │   ├── correlation.py         # Cross-asset correlation
│   │   ├── multi_timeframe.py     # Multi-TF alignment
│   │   ├── strategy_selector.py   # Dynamic strategy routing
│   │   └── macro_calendar.py      # Fed/CPI event awareness
│   ├── risk/                      # Risk Management
│   │   ├── manager.py             # Position sizing (Kelly criterion)
│   │   ├── protection.py          # 5-layer autonomous protection
│   │   └── position_sizer.py      # Dynamic sizing
│   ├── execution/                 # Order Execution
│   │   ├── paper_trader.py        # Simulated trading (fee-accurate)
│   │   ├── smart_exit.py          # 4-phase exit (normal→BE→trail→tight)
│   │   ├── trailing_stop.py       # Dynamic trailing stops
│   │   └── adaptive_sizer.py      # Win/loss streak sizing
│   ├── data/                      # Market Data
│   │   ├── feed.py                # Binance client (REST + WS)
│   │   ├── indicators.py          # Technical indicator engine
│   │   └── order_book.py          # Order book + whale detection
│   └── storage/
│       ├── database.py            # Async SQLite (trades, AI decisions)
│       └── models.py              # Data models
├── backtest/                      # Backtesting Framework
│   ├── engine.py                  # Walk-forward backtest engine
│   └── data_loader.py            # Historical data fetcher
├── config/
│   ├── settings.yaml              # Risk params, pairs, scaling tiers
│   └── strategies.yaml            # Per-strategy parameters
├── tests/                         # 18 test files
└── main.py                        # Entry point
```

## Evolution Roadmap

| Phase | Status | What |
|-------|--------|------|
| 0 | Done | Backtesting framework, historical data, walk-forward validation |
| 1 | Done | Foundation, indicators, risk manager, smart scalp strategy |
| 2 | Done | Grid strategy, Gemini AI brain, strategy selector |
| 3 | **Active** | Paper trading 24/7 on Oracle Cloud, collecting data |
| 4 | Code ready | XGBoost ML model — activates after 200+ trades |
| 5 | **Active** | RAG memory — storing every trade in ChromaDB from day 1 |
| 6 | Code ready | Agentic AI — 3 autonomous brain agents with tool use |
| 7 | Planned | Fine-tuned model — distill Gemini into custom Gemma 2B |
| 8 | Code ready | Solana DEX arbitrage — Jupiter/Raydium/Orca scanner |

## Key Engineering Decisions

**Why limit orders only?** Market orders pay taker fees (0.1%), limit orders pay maker fees (0.075%). On 20 trades/day, that's $0.50/day saved — 15% of daily target profit.

**Why 4/5 consensus?** Requiring 4 of 5 brains to agree eliminates low-confidence trades. Higher threshold = fewer trades but much better win rate.

**Why RAG over fine-tuning first?** RAG gives memory with zero training cost. The bot learns from its own history immediately. Fine-tuning needs 90+ days of data and GPU compute.

**Why quarter-Kelly sizing?** Full Kelly criterion is mathematically optimal but assumes perfect probability estimates. Quarter-Kelly (25% of optimal) provides 75% of the growth with much lower variance.

**Why 5 protection layers?** Each layer catches what the others miss. Per-trade stops catch single bad trades. Session protection catches losing streaks. Portfolio protection catches regime shifts. Black swan protection catches flash crashes. Infrastructure protection catches technical failures.

## Security

- API keys stored in `.env` only (never in code, config, or commits)
- Binance API: read + spot trading only, **withdrawal disabled**
- IP whitelist on API key (server IP only)
- `.env` is `chmod 600` on server
- No secrets in logs or error messages

## Setup

```bash
# Clone
git clone https://github.com/Leo-emp/AI-trading-bot-.git
cd AI-trading-bot-

# Install
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Binance API keys and Gemini API key

# Backtest first
python -m backtest.engine

# Paper trade
python main.py --slow
```

## License

MIT
