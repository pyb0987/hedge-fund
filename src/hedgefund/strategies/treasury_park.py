"""Strategy: Treasury Park — park idle cash in short-term T-bill ETF.

Always holds BIL (SPDR Bloomberg 1-3 Month T-Bill ETF) with weight 1.0.
Zero parameters, zero overfitting risk, provides risk-free yield on idle capital.
"""

from datetime import datetime

import pandas as pd

from hedgefund.config.schemas import TreasuryParkConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy


class TreasuryParkStrategy(BaseStrategy):
    """Park idle capital in short-term treasury ETF for risk-free yield."""

    def __init__(self, config: TreasuryParkConfig, **kwargs: object) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    @property
    def config(self) -> TreasuryParkConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return [self._config.symbol]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Always emit LONG signal for the treasury ETF."""
        symbol = self._config.symbol
        if symbol not in data:
            return []

        return [
            Signal(
                strategy_name=self.name,
                symbol=symbol,
                exchange=self.exchange,
                direction=SignalDirection.LONG,
                strength=1.0,
                timestamp=timestamp,
                metadata={"type": "cash_equivalent"},
            )
        ]

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Always 100% weight in treasury ETF."""
        symbol = self._config.symbol
        if symbol not in data:
            return pd.DataFrame(index=dates)

        return pd.DataFrame(1.0, index=dates, columns=[symbol])
