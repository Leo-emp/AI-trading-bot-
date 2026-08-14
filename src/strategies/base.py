# src/strategies/base.py
# Abstract base class for all trading strategies.
# Every strategy must implement evaluate() which takes OHLCV data
# with indicators and returns a StrategySignal.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    """Output from a strategy's evaluation.

    direction: BUY, SELL, or HOLD
    confidence: 0.0 to 1.0
    entry_price: suggested entry price (for limit order)
    stop_loss: hard stop-loss price
    take_profit: take-profit price
    strategy_name: which strategy generated this signal
    reasons: why this signal was generated
    """
    direction: str
    confidence: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_name: str = ""
    reasons: list[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclasses implement evaluate() with their specific logic.
    The strategy selector calls evaluate() and feeds the result
    into the trade gate for multi-brain consensus.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name for logging and config lookup."""
        ...

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, config: dict,
                 pair: str = "default") -> StrategySignal:
        """Analyze data and return a trading signal.

        df: OHLCV DataFrame with indicator columns already computed.
        config: strategy-specific parameters from strategies.yaml.
        pair: trading pair for per-pair cooldown tracking.
        """
        ...

    def is_enabled(self, config: dict) -> bool:
        """Check if this strategy is enabled in config."""
        return config.get("enabled", True)
