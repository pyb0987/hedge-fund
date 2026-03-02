"""Strategy protocol and abstract base class.

All strategies must implement the Strategy protocol.
The ABC provides common utilities (logging, config access).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from hedgefund.core.enums import Exchange
from hedgefund.core.models import Signal


@runtime_checkable
class Strategy(Protocol):
    """Protocol that all strategies must satisfy."""

    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    def exchange(self) -> Exchange:
        """Primary exchange for this strategy."""
        ...

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate trading signals from market data.

        Args:
            data: mapping of symbol → OHLCV DataFrame
            timestamp: current evaluation time

        Returns:
            List of Signal objects (can be empty if no action needed)
        """
        ...

    def get_universe(self) -> list[str]:
        """Return the list of symbols this strategy trades."""
        ...


class BaseStrategy(ABC):
    """Abstract base class with shared utilities for strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def exchange(self) -> Exchange:
        ...

    @abstractmethod
    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        ...

    @abstractmethod
    def get_universe(self) -> list[str]:
        ...

    @staticmethod
    def compute_returns(prices: pd.Series) -> pd.Series:
        """Compute simple returns from price series."""
        return prices.pct_change().dropna()

    @staticmethod
    def compute_momentum(prices: pd.Series, lookback: int) -> float:
        """Compute momentum as total return over lookback period."""
        if len(prices) < lookback + 1:
            return 0.0
        return float(prices.iloc[-1] / prices.iloc[-lookback - 1] - 1)

    @staticmethod
    def compute_zscore(prices: pd.Series, lookback: int) -> float:
        """Compute z-score of latest return relative to lookback window."""
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.0
        window = returns.iloc[-lookback:]
        mean = window.mean()
        std = window.std()
        if std == 0:
            return 0.0
        latest_return = returns.iloc[-1]
        return float((latest_return - mean) / std)
