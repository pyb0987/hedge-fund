"""Portfolio-level backtest — combines multiple strategies for walk-forward validation.

각 전략의 backtest_weights()를 배분 비율로 스케일링하여 합산한 포트폴리오 가중치를 생성.
교차 전략 효과(심볼 중복, SPY+SH 충돌 등)를 포트폴리오 레벨에서 검증 가능.
"""

from typing import Any, Protocol

import pandas as pd

from hedgefund.backtest.engine import BacktestConfig
from hedgefund.backtest.walk_forward import WalkForwardResult, run_walk_forward


class BacktestableStrategy(Protocol):
    """Strategy that supports backtest_weights."""

    @property
    def name(self) -> str: ...

    def get_universe(self) -> list[str]: ...

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame: ...


def portfolio_weight_generator(
    strategies: dict[str, Any],
    allocation: dict[str, float],
) -> callable:  # type: ignore[type-arg]
    """Create a combined weight generator for portfolio-level walk-forward.

    Each strategy's backtest_weights output is scaled by its allocation weight,
    then all are merged into a single DataFrame. This captures cross-strategy
    effects like symbol overlap and opposing positions.

    Args:
        strategies: name → strategy instance (must have backtest_weights + get_universe)
        allocation: name → target allocation fraction (0.0-1.0)

    Returns:
        Callable(data, dates) → combined weight DataFrame
    """

    def generate(
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        # Collect all symbols across all active strategies
        all_symbols: set[str] = set()
        for name, strategy in strategies.items():
            if allocation.get(name, 0.0) > 0:
                all_symbols.update(strategy.get_universe())

        if not all_symbols:
            return pd.DataFrame(0.0, index=dates, columns=["_empty"])

        sorted_symbols = sorted(all_symbols)
        combined = pd.DataFrame(0.0, index=dates, columns=sorted_symbols)

        for name, strategy in strategies.items():
            alloc = allocation.get(name, 0.0)
            if alloc <= 0:
                continue

            # Filter data to strategy's universe.
            # walk_forward passes {symbol: DataFrame[col=symbol]} but strategies
            # expect {symbol: DataFrame[col="close"]}, so rename the column.
            universe = strategy.get_universe()
            strategy_data = {}
            for sym in universe:
                if sym in data:
                    df = data[sym]
                    if sym in df.columns and "close" not in df.columns:
                        df = df.rename(columns={sym: "close"})
                    strategy_data[sym] = df

            if not strategy_data:
                continue

            weights = strategy.backtest_weights(strategy_data, dates)

            # Merge scaled weights into combined DataFrame
            for col in weights.columns:
                if col in combined.columns:
                    combined[col] = combined[col] + weights[col] * alloc

        return combined

    return generate


def run_portfolio_walk_forward(
    strategies: dict[str, Any],
    allocation: dict[str, float],
    data: dict[str, pd.DataFrame],
    is_ratio: float = 0.30,
    oos_ratio: float = 0.10,
    config: BacktestConfig | None = None,
    min_windows: int = 3,
) -> WalkForwardResult:
    """Run walk-forward validation on the combined portfolio.

    Args:
        strategies: name → strategy instance
        allocation: name → target allocation fraction
        data: symbol → OHLCV DataFrame (all symbols across all strategies)
        is_ratio: in-sample fraction
        oos_ratio: out-of-sample fraction
        config: backtest config (costs, capital)
        min_windows: minimum number of WF windows

    Returns:
        WalkForwardResult for the combined portfolio
    """
    if config is None:
        config = BacktestConfig()

    # Build unified prices DataFrame (close prices only)
    all_symbols: set[str] = set()
    for name, strategy in strategies.items():
        if allocation.get(name, 0.0) > 0:
            all_symbols.update(strategy.get_universe())

    # Find common date range across all symbols
    date_sets = []
    for sym in all_symbols:
        if sym in data and not data[sym].empty:
            date_sets.append(set(data[sym].index))

    if not date_sets:
        msg = "No data available for any strategy symbol"
        raise ValueError(msg)

    common_dates = sorted(set.intersection(*date_sets))
    prices_index = pd.DatetimeIndex(common_dates)

    prices = pd.DataFrame(index=prices_index)
    for sym in sorted(all_symbols):
        if sym in data:
            prices[sym] = data[sym].loc[prices_index, "close"]

    # Create combined weight generator
    gen = portfolio_weight_generator(strategies, allocation)

    return run_walk_forward(
        prices=prices,
        weight_generator=gen,
        is_ratio=is_ratio,
        oos_ratio=oos_ratio,
        config=config,
        min_windows=min_windows,
    )
