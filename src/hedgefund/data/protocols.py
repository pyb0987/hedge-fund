"""Data provider protocol — interface for all data sources."""

from datetime import datetime
from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Protocol for fetching OHLCV market data.

    All providers must implement this interface.
    Returns pandas DataFrames with columns: open, high, low, close, volume
    and a DatetimeIndex.
    """

    @property
    def exchange_name(self) -> str:
        """Name of the exchange/data source."""
        ...

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: ticker symbol (e.g. 'KRW-BTC', 'SPY')
            interval: candle interval ('day', 'week', 'minute60')
            count: number of candles to fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and DatetimeIndex, sorted oldest-first.
        """
        ...

    def get_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a date range.

        Args:
            symbol: ticker symbol
            start: start date (inclusive)
            end: end date (inclusive)
            interval: candle interval

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and DatetimeIndex, sorted oldest-first.
        """
        ...

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols."""
        ...
