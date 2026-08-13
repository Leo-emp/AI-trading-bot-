# AI Trading Bot — Design Spec

**Date:** 2026-08-13
**Status:** Draft → Audited (v2) → Autonomous Intelligence (v3)
**Target:** 2% daily return with ultra-conservative risk management
**Audit:** 4 critical bugs found and fixed (fee math, min order size, ML cold-start, missing backtest)
**v3:** Fully autonomous operation — zero human intervention required

## Overview

A modular, event-driven AI trading bot for cryptocurrency markets. Targets Binance CEX as the primary exchange with DEX arbitrage (Solana) planned for Phase 2. Uses a hybrid AI layer — Gemini for sentiment/regime detection, local ML (XGBoost) for real-time trade signals. Designed for micro accounts ($20-100 starting capital) with aggressive compounding.

**Core principle:** No single trade can hurt you. Profits come from volume × slight edge × compounding.

## Performance Target

| Metric | Target |
|--------|--------|
| Daily return | 2% net (after fees) |
| Trades per day | 15-25 high-quality (not 50 low-quality) |
| Win rate | 55-60%+ |
| Risk/reward ratio | 2:1 per trade (was 1.5:1 — increased to overcome fees) |
| Max risk per trade | 0.4% of portfolio ($0.20 on $50) |
| Max daily drawdown | 5% (auto-shutdown) |
| Order type | LIMIT ONLY (maker fee 0.02-0.1%, never taker 0.1%) |
| BNB holding | Required (25% fee discount) |

### Fee-Aware Profit Math

```
Binance maker fee (with BNB): 0.075% per side → 0.15% round-trip
Target profit per winning trade: 0.8-1.0% gross → 0.65-0.85% net
Stop-loss per losing trade: 0.4% gross → 0.55% net loss (incl fees)

With 57% win rate, 20 trades/day:
  Wins:  11.4 trades × 0.75% net = +8.55%
  Losses: 8.6 trades × 0.55% net = -4.73%
  Daily net: +8.55% - 4.73% = +3.82% gross
  Per-trade position size (~$12): actual daily P&L ≈ +2.0% of portfolio
```

### Realistic Projected Growth from $50

(Assumes 1.5% average daily — some days 0% or negative, some days 3%+)

| Timeline | Conservative (1%) | Target (1.5%) | Optimistic (2%) |
|----------|-------------------|---------------|-----------------|
| Month 1 | $67 | $78 | $90 |
| Month 3 | $122 | $172 | $296 |
| Month 6 | $301 | $596 | $1,763 |
| Month 12 | $1,813 | $7,126 | $62,264 |

## Architecture

Event-driven pipeline with pluggable strategy modules:

```
Market Data → Indicators → Regime Detector → Strategy Selector → Risk Check → Execute
```

### System Components

1. **Market Data Feed** — Binance WebSocket for real-time price ticks + REST for OHLCV candles
2. **Indicator Engine** — RSI, MACD, Bollinger Bands, ATR, volume profiles, order book depth
3. **Regime Detector** — Local ML classifies market state. Gemini overlays sentiment every 15 min
4. **Strategy Selector** — Routes to the optimal strategy for current conditions
5. **Risk Manager** — Validates every trade against position limits, stop-loss, drawdown before execution
6. **Order Executor** — Places orders via ccxt. Handles retries, partial fills, slippage protection
7. **Portfolio Tracker** — Real-time balance, open positions, unrealized P&L
8. **SQLite Database** — Stores all trades, P&L, strategy performance
9. **Telegram Notifications** — Trade alerts, errors, daily P&L summaries

## Trading Strategies

### Strategy 1: Smart Scalping (Primary — fee-aware)

The core profit engine. Fewer but higher-quality trades that overcome fee drag.

- Monitors 5m and 15m candles (not 1m — too noisy, fee-destroyed)
- Enters on multi-confirmation: volume spike + momentum + order book imbalance
- Uses LIMIT ORDERS ONLY (maker fee, never taker)
- Targets 0.8-1.0% profit per trade, stops at 0.4% loss (2:1 reward/risk)
- Waits for high-probability setups — skips marginal signals
- Exits within 15-60 minutes
- Pairs: BTC/USDT, ETH/USDT, SOL/USDT (highest liquidity, tightest spreads)

### Strategy 2: Grid Trading (Sideways markets)

Places buy/sell orders at fixed intervals around current price.

- Grid spacing: dynamic, based on ATR (volatility-adjusted)
- Grid levels: 5-10 above and below current price
- Profits from price oscillation without predicting direction
- Best in ranging/sideways markets (60-70% of the time)

### Strategy 3: Momentum / Trend Following (Trending markets)

Rides breakouts and sustained moves.

- Entry: EMA crossover + volume confirmation + RSI not overbought/oversold
- Exit: trailing stop-loss (tightens as profit grows)
- Position size: scaled by AI confidence score
- Only activates when regime detector confirms trend

### Strategy 4: Mean Reversion (Extreme moves)

Catches bounces from oversold/overbought extremes.

- Entry: RSI < 25 (buy) or RSI > 75 (sell) on 15m+ timeframe
- Confirmation: Bollinger Band touch + volume divergence
- Take profit: return to mean (middle Bollinger Band)
- Only on high-cap pairs (BTC, ETH) — small caps don't revert reliably

### Strategy 5: DEX Arbitrage Scanner (Phase 2)

Monitors Solana DEX pools for cross-pool price differences.

- Scans Jupiter, Raydium, Orca for the same token at different prices
- Executes atomic swap: buy cheap on Pool A → sell on Pool B
- Profit = price gap - gas fees - slippage
- Requires: Solana RPC, wallet, fast execution
- Build after CEX strategies are proven

### Strategy Selection Logic

```
IF regime == SIDEWAYS:
    primary = GridTrading
    secondary = MicroScalping
ELIF regime == TRENDING:
    primary = Momentum
    secondary = MicroScalping
ELIF regime == VOLATILE:
    primary = MicroScalping (tighter stops)
    secondary = MeanReversion (on extremes only)
ELIF regime == CRASH:
    reduce_all_positions()
    MeanReversion only (small size, wait for bounce)
```

MicroScalping runs continuously as a background strategy regardless of regime.

## AI Layer

### Gemini (every 15-30 minutes)

- Analyzes crypto news headlines, Twitter/X sentiment, fear/greed index
- Classifies regime: BULLISH / BEARISH / SIDEWAYS / VOLATILE / CRASH
- Outputs confidence score (0-100)
- Adjusts strategy aggressiveness:
  - Confidence > 75: full position sizes
  - Confidence 50-75: standard sizes
  - Confidence < 50: half sizes, defensive mode
  - Confidence < 25: pause new trades, tighten stops

### Local ML Model (Phase 2 — NOT day 1)

ML activates only after collecting 30+ days of paper trading data. Until then, rule-based signals only.

**Phase 1 (rule-based):**
- RSI crossovers (oversold/overbought thresholds)
- MACD signal line crossovers
- Bollinger Band breakouts with volume confirmation
- Order book imbalance detection (bid/ask ratio)
- These are proven, fee-aware, and need no training data

**Phase 2 (ML-enhanced, after 30+ days):**
- XGBoost trained on bot's own paper trading results + historical OHLCV
- Features: RSI, MACD, Bollinger width, volume ratio, order book imbalance, recent trade flow
- Output: BUY / SELL / HOLD + probability (0-1)
- Retrained weekly using walk-forward validation (train on 60 days, validate on 7)
- Minimum probability threshold: 0.6 to act
- ML signal overrides rule-based only when probability > 0.7

### Combined Decision

```
signal_strength = ml_probability * gemini_confidence_weight
IF ml_signal == BUY AND signal_strength > 0.5:
    execute_buy(size=kelly_optimal_size)
ELIF ml_signal == SELL AND signal_strength > 0.5:
    execute_sell(size=kelly_optimal_size)
ELSE:
    hold()
```

## Autonomous Intelligence (Zero Human Intervention)

The bot must run 24/7 without any human watching it. Every scenario that could require a human decision is handled automatically.

### 1. Multi-Brain Decision Engine

No single signal decides a trade. Every trade requires consensus from multiple independent analysis layers:

```
                    ┌─────────────────┐
                    │   TRADE GATE    │ ← Final yes/no
                    │  (requires 3/5  │
                    │   brains agree) │
                    └────────┬────────┘
           ┌─────────┬──────┼──────┬──────────┐
           ▼         ▼      ▼      ▼          ▼
      ┌─────────┐ ┌──────┐ ┌────┐ ┌───────┐ ┌──────┐
      │Technical│ │Order │ │ AI │ │Multi- │ │Cross-│
      │Signals  │ │Flow  │ │News│ │Time-  │ │Asset │
      │(RSI,    │ │(book │ │Senti│ │frame  │ │Corr- │
      │MACD,BB) │ │depth)│ │ment │ │Align  │ │elation│
      └─────────┘ └──────┘ └────┘ └───────┘ └──────┘
       Brain 1    Brain 2  Brain 3  Brain 4   Brain 5
```

**Brain 1 — Technical Signals:** RSI, MACD crossover, Bollinger breakout, volume confirmation
**Brain 2 — Order Flow Analysis:** Order book imbalance, large order detection (whale watching), trade flow direction
**Brain 3 — AI Sentiment (Gemini):** News impact, social sentiment, fear/greed index, macro events
**Brain 4 — Multi-Timeframe Alignment:** Signal must align on 5m + 15m + 1h. If 5m says BUY but 1h says SELL → skip.
**Brain 5 — Cross-Asset Correlation:** BTC dumping? Don't long any altcoin. ETH/BTC diverging? Adjust pair selection.

**Trade Gate rule:** At least 3 of 5 brains must agree on direction AND none can show a strong OPPOSING signal. This prevents low-confidence trades.

### 2. Self-Learning Performance Engine

The bot tracks its own performance and automatically adjusts what works and what doesn't.

**Strategy scoring (rolling 50-trade window):**
```
strategy_score = (win_rate × avg_profit) - ((1 - win_rate) × avg_loss) - avg_fees
```

- Every strategy gets a live score updated after each trade
- Strategies with score < 0 are auto-disabled (losing money after fees)
- Strategies with highest score get more capital allocation
- Weekly performance review: Gemini analyzes trade logs and recommends parameter tweaks

**Adaptive parameter tuning:**
- RSI thresholds adjust based on recent accuracy (if RSI < 30 entries win 70%, lower threshold to 25)
- Grid spacing adjusts based on ATR (wider in volatile, tighter in calm)
- Take-profit/stop-loss ratios adjust based on recent market behavior
- All adjustments are bounded — can't drift more than 30% from defaults (prevents runaway optimization)

**Performance memory:**
- Stores which parameters worked best in each market regime
- When regime changes, loads the best-performing params for that regime
- Example: "Last time BTC was in SIDEWAYS + low volatility, grid spacing of 0.3% worked best"

### 3. Autonomous Self-Protection System

Multi-layer defense that acts faster than any human could:

**Layer 1 — Per-Trade Protection (milliseconds):**
- Hard stop-loss on every position (never removed, never widened)
- Trailing stop-loss that locks in profits as price moves in your favor
- Time-based exit: close any position open > 4 hours (scalping, not investing)
- Stale data kill switch: if last price update > 10 seconds old, halt all trading

**Layer 2 — Session Protection (minutes):**
- 3 consecutive losses → reduce position sizes by 50% for next 5 trades
- 5 consecutive losses → pause trading for 30 minutes, re-evaluate regime
- If post-pause trades also lose → pause for 2 hours
- Daily drawdown > 3% → switch to defense-only mode (only mean reversion on extremes)
- Daily drawdown > 5% → full shutdown, send Telegram alert, no trades until next day

**Layer 3 — Portfolio Protection (hours):**
- Weekly drawdown > 10% → reduce all position sizes to 50% for the following week
- Monthly drawdown > 15% → emergency mode: only paper trade for 7 days, re-run backtests
- If backtests still profitable but live isn't → likely market regime shifted, retrain parameters
- If backtests also fail → strategy is broken, alert user, stay in paper mode

**Layer 4 — Black Swan Protection (instant):**
- Price moves > 5% in 1 minute on any held asset → instant market sell everything
- Exchange API errors > 3 in a row → close all positions, pause until API stable
- Balance drops below $10 absolute floor → permanent shutdown until manually restarted
- Unusual spread widening (>1%) → pause (sign of low liquidity or manipulation)

**Layer 5 — Infrastructure Self-Healing:**
- WebSocket disconnection → auto-reconnect with exponential backoff (1s, 2s, 4s, 8s...)
- If reconnection fails 5 times → switch to REST polling mode (slower but reliable)
- Process crash → auto-restart via systemd/supervisor (configured in deployment)
- Database corruption → automatic backup restored from hourly snapshots
- Gemini API down → continue with technical signals only (degrade gracefully, don't stop)

### 4. Autonomous Market Intelligence

The bot reads the market like a professional trader, without being told what to look for:

**Whale detection:**
- Monitor order book for sudden large orders (>$50K on a single level)
- Whale buys appearing → bullish signal boost
- Whale sells appearing → bearish signal boost
- Whale order cancellation (spoofing detection) → ignore fake signals

**Liquidity analysis:**
- Before entering any trade, check bid/ask spread
- Spread > 0.15% → skip (too expensive to trade)
- Low liquidity hours (2-6 AM UTC typically) → reduce position sizes by 50%
- Volume spike detection → increase attention (something is happening)

**Correlation-aware position management:**
- Track BTC/ETH/SOL correlation in real-time
- If holding ETH long and BTC starts dumping → tighten ETH stop-loss immediately
- Never hold 3 correlated longs simultaneously (all altcoins crash together)
- Diversification enforced: max 60% exposure to any single pair

**News-reactive speed:**
- Gemini analyzes breaking news every 15 minutes
- Critical keyword detection (hack, SEC, ban, ETF, crash, regulation) → instant regime re-evaluation
- Major exchange outage detected → pause trading on affected pairs
- Scheduled events (FOMC, CPI data releases) → reduce positions 30 min before, increase after

### 5. Autonomous Reporting (Telegram)

You never need to check in. The bot tells you what matters:

**Real-time alerts (only important events):**
- Daily P&L summary at midnight UTC (trades, win rate, net profit, balance)
- Protection triggered (which layer, what happened, what action taken)
- Strategy auto-disabled (which one, why, what replaced it)
- Unusual market event detected (crash, extreme volatility, liquidity crisis)

**Weekly intelligence report:**
- Best/worst performing strategy and why
- Market regime changes detected
- Parameter adjustments made
- Projected monthly return at current performance
- Any issues requiring attention (none in normal operation)

**No trade-by-trade notifications** — that's noise, not signal. You check when you want to, the bot handles the rest.

## Risk Management

Every trade passes through the risk manager. No exceptions.

### Per-Trade Rules

| Rule | Value | Purpose |
|------|-------|---------|
| Position size | $10-15 (Binance minimum floor) | Exchange minimum is $5-10; can't go lower |
| Max position as % of portfolio | 25% ($12.50 on $50) | Binance min forces larger % at micro scale |
| Stop-loss | 0.4% of position ($0.04-0.06) | 0.4% of portfolio risk at $50 |
| Take-profit | 0.8-1.0% of position | 2:1 reward/risk ratio, overcomes fees |
| Min net profit threshold | Must exceed 2× round-trip fees (0.30%) | Never trade unless net profit > 0.30% after fees |
| Order type | LIMIT ONLY | Maker fee (0.075% with BNB) vs taker (0.1%) |
| Slippage protection | Cancel if price moves >0.1% before fill | Limit orders provide natural slippage protection |

### Portfolio Rules

| Rule | Value | Purpose |
|------|-------|---------|
| Max open positions | 3 simultaneous | Don't overexpose |
| Max daily drawdown | 5% | Bot auto-shuts down |
| Max consecutive losses | 5 → pause 30 min | Prevent cascade |
| Max daily trades | 100 | Prevent overtrading on fees |
| Min balance to trade | $10 | Emergency floor |

### Kelly Criterion Position Sizing

```
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = kelly_fraction * portfolio_balance * safety_factor(0.25)
```

Uses quarter-Kelly (0.25 safety factor) for conservative sizing. Adjusts dynamically based on rolling 50-trade win rate.

### Auto-Scaling Risk Params

As the balance grows, position sizing becomes more flexible:

| Balance | Position size | Max per-trade risk | Max positions | Daily trade limit |
|---------|--------------|-------------------|---------------|-------------------|
| $20-100 | $10-15 (Binance min) | 0.4% of portfolio | 3 | 25 |
| $100-500 | 3-5% of portfolio | 0.5% of portfolio | 4 | 40 |
| $500-2000 | 2-4% of portfolio | 0.75% of portfolio | 5 | 60 |
| $2000+ | 1-3% of portfolio | 1.0% of portfolio | 6 | 80 |

Note: At micro scale ($20-100), Binance minimum order size ($10) forces larger position % than ideal.
As balance grows past $200, position sizing normalizes to proper fractional Kelly.

## Backtesting (Mandatory Phase 0a — Before Paper Trading)

Before wasting 14 days paper trading a bad strategy, backtest on historical data first.

- Download 6-12 months of historical OHLCV data from Binance (free)
- Run every strategy against historical data with realistic fee simulation
- Only strategies that show profit AFTER FEES in backtest proceed to paper trading
- Walk-forward validation: train on 80% of data, test on remaining 20%
- Minimum backtest performance: Sharpe ratio > 1.0, max drawdown < 15%, positive net P&L after fees
- Backtest takes hours, not days — fast feedback loop

## Paper Trading (Mandatory Phase 0b)

After strategies pass backtesting, paper trade against live market data.

- Simulates all trades against live market data (real prices, simulated execution)
- Tracks virtual P&L identically to live trading
- Same risk management rules apply
- Simulates Binance fees accurately (maker/taker, BNB discount)
- Generates daily performance reports
- Go-live criteria: 14+ days, 55%+ win rate, positive cumulative P&L after fees, no daily drawdown > 3%
- If paper trading fails criteria → go back to strategy tuning, not live trading

## Data Storage (SQLite)

### Tables

**trades** — Every trade executed
- id, timestamp, pair, side (buy/sell), strategy, entry_price, exit_price, quantity, pnl, fees, status

**portfolio_snapshots** — Hourly balance snapshots
- id, timestamp, total_balance, unrealized_pnl, open_positions_count

**strategy_performance** — Rolling stats per strategy
- strategy_name, total_trades, win_rate, avg_pnl, sharpe_ratio, max_drawdown

**ai_decisions** — Every AI decision logged for analysis
- id, timestamp, regime, gemini_confidence, ml_signal, ml_probability, action_taken

## Notifications (Telegram)

- Trade executed (pair, side, price, size)
- Stop-loss triggered
- Daily P&L summary (total trades, win rate, net P&L, balance)
- System errors or exchange connectivity issues
- Auto-shutdown triggered (drawdown limit hit)

## Project Structure

```
ai-trading-bot/
├── .env                           # API keys (NEVER committed, in .gitignore)
├── .env.example                   # Template showing required vars (no real values)
├── config/
│   ├── settings.yaml          # Risk params, pairs (NO secrets)
│   └── strategies.yaml        # Per-strategy parameters
├── src/
│   ├── core/
│   │   ├── engine.py          # Main trading loop
│   │   ├── event_bus.py       # Event pipeline
│   │   └── scheduler.py      # Task scheduling
│   ├── data/
│   │   ├── feed.py            # Binance WebSocket + REST
│   │   ├── indicators.py      # Technical indicator calculations
│   │   └── order_book.py      # Order book depth analysis
│   ├── strategies/
│   │   ├── base.py            # Abstract strategy interface
│   │   ├── grid.py            # Grid trading
│   │   ├── momentum.py        # Trend following
│   │   ├── mean_reversion.py  # Mean reversion
│   │   ├── micro_scalp.py     # High-frequency micro-scalping
│   │   └── dex_arb.py         # DEX arbitrage (Phase 2)
│   ├── ai/
│   │   ├── regime_detector.py # Market regime classification
│   │   ├── gemini_analyst.py  # Gemini sentiment/news analysis
│   │   ├── ml_model.py        # XGBoost signal model (Phase 2)
│   │   └── trade_gate.py      # Multi-brain consensus engine (3/5 must agree)
│   ├── intelligence/
│   │   ├── whale_detector.py  # Large order detection in order book
│   │   ├── correlation.py     # Cross-asset correlation tracker
│   │   ├── liquidity.py       # Spread/volume/liquidity analysis
│   │   ├── news_reactor.py    # Breaking news keyword detection
│   │   └── self_learner.py    # Strategy scoring + adaptive parameter tuning
│   ├── risk/
│   │   ├── manager.py         # Risk validation layer
│   │   ├── position_sizer.py  # Kelly criterion
│   │   ├── stop_loss.py       # Stop-loss + trailing stop management
│   │   └── protection.py      # 5-layer autonomous protection system
│   ├── execution/
│   │   ├── executor.py        # Order placement + management (limit orders only)
│   │   ├── paper_trader.py    # Simulated trading with fee simulation
│   │   └── binance_client.py  # Exchange API wrapper + auto-reconnect
│   ├── storage/
│   │   ├── database.py        # SQLite operations
│   │   └── models.py          # Data models
│   └── notifications/
│       └── telegram.py        # Trade alerts + daily reports
├── backtest/
│   ├── engine.py              # Backtesting framework
│   ├── data_loader.py         # Historical data fetcher
│   └── report.py              # Performance reports
├── ml/
│   ├── train.py               # Model training pipeline
│   ├── features.py            # Feature engineering
│   └── models/                # Saved model files
├── tests/
├── main.py                    # Entry point
├── watchdog.py                # Process monitor — auto-restarts bot on crash
└── requirements.txt
```

## Key Libraries

| Library | Purpose |
|---------|---------|
| ccxt | Exchange API (Binance + extensible) |
| pandas / numpy | Data manipulation |
| ta | Technical indicators |
| xgboost / scikit-learn | ML models |
| google-generativeai | Gemini API |
| python-telegram-bot | Notifications |
| websockets | Real-time data feeds |
| aiosqlite | Async SQLite |
| pyyaml | Configuration |
| solders / solana-py | Solana integration (Phase 2) |

## Build Phases

### Phase 0: Backtesting Framework (Week 1, first half)
- Historical data downloader (Binance REST API, 6-12 months OHLCV)
- Backtest engine with realistic fee simulation (maker/taker, BNB discount)
- Performance metrics: Sharpe ratio, max drawdown, win rate, net P&L after fees
- Walk-forward validation framework
- This phase gates everything — no strategy ships without passing backtest

### Phase 1: Foundation + Smart Scalping (Week 1, second half)
- Project scaffolding, .env config, database
- Binance client (REST + WebSocket)
- Indicator engine (RSI, MACD, Bollinger, ATR, volume, order book)
- Risk manager with all rules (fee-aware)
- Smart scalping strategy (rule-based signals, limit orders only)
- Paper trading mode with accurate fee simulation
- Basic Telegram alerts
- Backtest smart scalping → must pass before proceeding

### Phase 2: Grid Trading + Gemini AI (Week 2)
- Grid trading strategy (3-5 levels for micro accounts)
- Gemini regime detector (sentiment, news, regime classification)
- Strategy selector (regime-based routing)
- Combined AI decision engine (rule-based signals + Gemini confidence)
- Performance reporting (daily P&L, per-strategy breakdown)
- Backtest grid strategy → must pass

### Phase 3: Additional Strategies + Paper Trading (Week 3)
- Momentum / trend following
- Mean reversion
- Backtest all strategies individually and combined
- Begin 14-day paper trading period
- Strategy performance comparison dashboard

### Phase 4: Live Trading + ML Layer (Week 4+)
- Paper trading must pass go-live criteria first
- Live trading mode (real money, start with minimum $20)
- Auto-scaling risk params
- XGBoost ML model (trained on 30+ days of paper trading data)
- ML model retraining pipeline (weekly walk-forward)
- Advanced order types (OCO, trailing stop)
- Daily/weekly performance reports

### Phase 5: DEX Arbitrage (Future)
- Solana wallet integration
- DEX pool monitoring
- Atomic swap execution
- Cross-chain arbitrage

## Fee Minimization Strategy (Critical for Profitability)

Fees are the #1 enemy of high-frequency trading. Every decision must be fee-aware.

| Technique | Fee reduction | Implementation |
|-----------|--------------|----------------|
| LIMIT orders only (maker) | 0.1% → 0.02-0.1% | Order executor enforces limit-only mode |
| Hold BNB for fee payment | 25% discount on all fees | Buy $5 BNB on day 1, auto-replenish |
| Referral code on signup | 10-20% kickback | Use referral link when creating account |
| Increase VIP tier (future) | Up to 0.02% maker | Automatic as 30-day volume grows |
| Skip marginal trades | Avoid fee-negative trades | Min net profit must exceed 2× fees |

**Fee budget per trade:**
- Maker fee with BNB: ~0.075% per side = 0.15% round-trip
- Target profit must be >0.30% to be worth executing (2× fee)
- Every P&L calculation includes fees — no "gross profit" illusions

## Exchange Setup Required

1. Create Binance account (binance.com) — use a referral code for fee discount
2. Complete KYC verification
3. Generate API key + secret (Spot trading permissions only, NO withdrawal)
4. Deposit $20-50 USDT
5. Buy $5 worth of BNB (for fee discount — pays for itself immediately)
6. Enable "Use BNB for fees" in Binance settings
7. Set API IP whitelist (security)
8. NEVER enable withdrawal permission on API key

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Exchange API downtime | Graceful reconnection with exponential backoff |
| Flash crash | 5% daily drawdown auto-shutdown |
| API key leak | Keys in env vars only, .gitignore enforced, no withdrawal permission |
| Model overfitting | Walk-forward validation, retrain weekly on recent data only |
| Fee erosion | Min profit threshold must exceed 2x fees |
| Liquidity issues | Trade only top-10 volume pairs |
| Regulatory changes | Binance spot trading only (lowest regulatory risk) |
| Fee erosion at scale | Fee budget enforced: skip trades where net profit < 2× fees |
| Binance min order rejection | Position sizer floors at $10 (exchange minimum) |
| ML overconfidence day 1 | ML disabled until 30+ days data collected; rule-based only at start |
| Backtest overfitting | Walk-forward validation; never optimize on test data |
| WebSocket disconnection | Heartbeat monitor, auto-reconnect with exponential backoff, close positions on prolonged outage |
| Stale data trading | Timestamp check — reject any signal based on data older than 5 seconds |
