"""Data collector — fetches OHLCV data for all strategies.

전략별 유니버스에 맞는 OHLCV 데이터를 수집하고,
PortfolioManager.run_cycle()에 필요한 형식으로 반환합니다.

반환 형식: dict[strategy_name, dict[symbol, pd.DataFrame]]
"""

import logging
from dataclasses import dataclass

import pandas as pd

from hedgefund.core.exceptions import DataError
from hedgefund.data.protocols import DataProvider
from hedgefund.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionResult:
    """Result of data collection for all strategies."""

    data: dict[str, dict[str, pd.DataFrame]]
    latest_prices: dict[str, float]  # symbol → latest close price
    errors: tuple[str, ...]  # collection error messages


class DataCollector:
    """Collects OHLCV data for all strategies from providers.

    각 전략의 유니버스를 순회하며 적절한 provider에서 데이터를 가져옵니다.
    """

    def __init__(
        self,
        providers: dict[str, DataProvider],
        strategies: dict[str, Strategy],
        lookback_buffer: int = 30,
    ) -> None:
        """Initialize data collector.

        Args:
            providers: exchange_name → DataProvider (e.g., "upbit" → UpbitProvider)
            strategies: strategy_name → Strategy instance
            lookback_buffer: extra days to fetch beyond strategy lookback
        """
        self._providers = providers
        self._strategies = strategies
        self._lookback_buffer = lookback_buffer

    def collect_all(self, count: int = 200) -> CollectionResult:
        """Collect OHLCV data for all strategies.

        Args:
            count: number of candles to fetch per symbol

        Returns:
            CollectionResult with data, latest prices, and any errors
        """
        all_data: dict[str, dict[str, pd.DataFrame]] = {}
        latest_prices: dict[str, float] = {}
        errors: list[str] = []

        for strategy_name, strategy in self._strategies.items():
            strategy_data, strategy_errors = self._collect_for_strategy(
                strategy, count,
            )
            all_data[strategy_name] = strategy_data
            errors.extend(strategy_errors)

            # Extract latest prices
            for symbol, df in strategy_data.items():
                if not df.empty:
                    latest_prices[symbol] = float(df["close"].iloc[-1])

        return CollectionResult(
            data=all_data,
            latest_prices=dict(latest_prices),
            errors=tuple(errors),
        )

    def _collect_for_strategy(
        self,
        strategy: Strategy,
        count: int,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Collect data for a single strategy's universe."""
        data: dict[str, pd.DataFrame] = {}
        errors: list[str] = []

        universe = strategy.get_universe()
        if not universe:
            errors.append(f"{strategy.name}: empty universe")
            return data, errors

        for symbol in universe:
            provider = self._resolve_provider(symbol)
            if provider is None:
                errors.append(
                    f"{strategy.name}/{symbol}: no provider found"
                )
                continue

            try:
                df = provider.get_ohlcv(symbol, interval="day", count=count)
                if df.empty:
                    errors.append(f"{strategy.name}/{symbol}: empty data")
                    continue
                data[symbol] = df
            except DataError as e:
                errors.append(f"{strategy.name}/{symbol}: {e}")
            except Exception as e:
                errors.append(f"{strategy.name}/{symbol}: unexpected error: {e}")

        return data, errors

    def _resolve_provider(self, symbol: str) -> DataProvider | None:
        """Resolve the correct data provider for a symbol.

        Convention:
        - KRW-* symbols → upbit provider
        - Otherwise → yfinance provider
        """
        if symbol.startswith("KRW-"):
            return self._providers.get("upbit")
        return self._providers.get("yfinance")
