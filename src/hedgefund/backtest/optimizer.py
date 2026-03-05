"""Parameter optimizer — grid search with Deflated Sharpe correction.

IS/OOS split으로 모든 파라미터 조합을 평가하고,
Deflated Sharpe Ratio로 다중 테스트 보정합니다.

Usage:
    result = run_grid_search(
        strategy_factory=lambda p: EtfMeanReversionStrategy(EtfMeanReversionConfig(**p)),
        data=ohlcv_data,
        param_grid={"lookback_days": [15, 20, 25], "z_entry_threshold": [-1.0, -1.5, -2.0]},
    )
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from hedgefund.backtest.deflated_sharpe import compute_return_moments, deflated_sharpe_ratio
from hedgefund.backtest.engine import BacktestConfig, BacktestResult, run_backtest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParamResult:
    """Result of a single parameter combination."""

    params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_max_dd: float
    oos_profit_factor: float
    oos_return: float
    efficiency: float  # OOS Sharpe / IS Sharpe

    @property
    def passes_go_nogo(self) -> bool:
        return (
            self.oos_sharpe > 0.8
            and self.oos_max_dd < 0.20
            and self.oos_profit_factor > 1.3
        )


@dataclass(frozen=True)
class GridSearchResult:
    """Complete grid search result with DSR correction."""

    all_results: tuple[ParamResult, ...]  # sorted by OOS Sharpe desc
    num_trials: int
    deflated_sharpe_p: float
    best_oos_sharpe: float

    @property
    def best(self) -> ParamResult | None:
        return self.all_results[0] if self.all_results else None


def run_grid_search(
    strategy_factory: Callable[[dict[str, Any]], Any],
    data: dict[str, pd.DataFrame],
    param_grid: dict[str, list[Any]],
    is_fraction: float = 0.70,
    backtest_config: BacktestConfig = BacktestConfig(),
    base_params: dict[str, Any] | None = None,
) -> GridSearchResult:
    """Run grid search over parameter combinations.

    Args:
        strategy_factory: callable(params_dict) → strategy instance
            Must return an object with get_universe() and backtest_weights()
        data: symbol → OHLCV DataFrame
        param_grid: param_name → list of values to try
        is_fraction: fraction of data for in-sample (rest is OOS)
        backtest_config: commission/slippage config
        base_params: default params merged with each grid point

    Returns:
        GridSearchResult with all evaluated combos, sorted by OOS Sharpe
    """
    if base_params is None:
        base_params = {}

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))
    num_trials = len(combos)

    logger.info("Grid search: %d combinations", num_trials)

    results: list[ParamResult] = []

    for combo in combos:
        params = {**base_params, **dict(zip(param_names, combo))}

        try:
            result = _evaluate_params(
                strategy_factory, data, params, is_fraction, backtest_config,
            )
            results.append(result)
        except Exception:
            logger.warning("Failed for params %s", params, exc_info=True)

    # Sort by OOS Sharpe descending
    results.sort(key=lambda r: r.oos_sharpe, reverse=True)

    # Compute Deflated Sharpe for best result
    best_oos_sharpe = results[0].oos_sharpe if results else 0.0
    dsr_p = 0.0

    if results:
        # Use the best OOS result's return moments for DSR
        best_params = results[0].params
        try:
            strategy = strategy_factory(best_params)
            moments = _get_oos_moments(strategy, data, is_fraction, backtest_config)
            oos_length = int(len(next(iter(data.values()))) * (1 - is_fraction))
            dsr_p = deflated_sharpe_ratio(
                best_oos_sharpe, num_trials, oos_length, **moments,
            )
        except Exception:
            logger.warning("DSR computation failed", exc_info=True)

    return GridSearchResult(
        all_results=tuple(results),
        num_trials=num_trials,
        deflated_sharpe_p=dsr_p,
        best_oos_sharpe=best_oos_sharpe,
    )


def _evaluate_params(
    strategy_factory: Callable[[dict[str, Any]], Any],
    data: dict[str, pd.DataFrame],
    params: dict[str, Any],
    is_fraction: float,
    config: BacktestConfig,
) -> ParamResult:
    """Evaluate a single parameter set on IS and OOS data."""
    strategy = strategy_factory(params)
    universe = strategy.get_universe()

    # Filter data to strategy's universe
    strategy_data = {s: data[s] for s in universe if s in data}
    if not strategy_data:
        msg = f"No data for universe {universe}"
        raise ValueError(msg)

    # Find common date range
    date_sets = [set(df.index) for df in strategy_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    split_idx = int(len(common_dates) * is_fraction)

    is_dates = pd.DatetimeIndex(common_dates[:split_idx])
    oos_dates = pd.DatetimeIndex(common_dates[split_idx:])

    if len(is_dates) < 50 or len(oos_dates) < 20:
        msg = f"Insufficient data: IS={len(is_dates)}, OOS={len(oos_dates)}"
        raise ValueError(msg)

    # Build IS/OOS data subsets
    is_data = {s: df.loc[df.index.isin(is_dates)] for s, df in strategy_data.items()}
    oos_data = {s: df.loc[df.index.isin(set(common_dates[:]))] for s, df in strategy_data.items()}

    # Run IS backtest
    is_prices = pd.DataFrame({s: is_data[s]["close"] for s in is_data})
    is_weights = strategy.backtest_weights(is_data, is_dates)
    is_result = run_backtest(is_prices, is_weights, config)

    # Run OOS backtest — strategy sees all historical data up to each OOS date
    oos_prices = pd.DataFrame({s: strategy_data[s].loc[oos_dates, "close"] for s in strategy_data})
    oos_weights = strategy.backtest_weights(oos_data, oos_dates)
    # Filter weights to OOS dates only
    oos_weights = oos_weights.loc[oos_dates]
    oos_result = run_backtest(oos_prices, oos_weights, config)

    efficiency = oos_result.sharpe_ratio / is_result.sharpe_ratio if is_result.sharpe_ratio != 0 else 0.0

    return ParamResult(
        params=params,
        is_sharpe=is_result.sharpe_ratio,
        oos_sharpe=oos_result.sharpe_ratio,
        oos_max_dd=oos_result.max_drawdown,
        oos_profit_factor=oos_result.profit_factor,
        oos_return=oos_result.annualized_return,
        efficiency=efficiency,
    )


def _get_oos_moments(
    strategy: Any,
    data: dict[str, pd.DataFrame],
    is_fraction: float,
    config: BacktestConfig,
) -> dict[str, float]:
    """Compute return moments for DSR calculation."""
    universe = strategy.get_universe()
    strategy_data = {s: data[s] for s in universe if s in data}
    if not strategy_data:
        return {"skewness": 0.0, "excess_kurtosis": 0.0}

    date_sets = [set(df.index) for df in strategy_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    split_idx = int(len(common_dates) * is_fraction)
    oos_dates = pd.DatetimeIndex(common_dates[split_idx:])

    oos_prices = pd.DataFrame({s: strategy_data[s].loc[oos_dates, "close"] for s in strategy_data})
    oos_weights = strategy.backtest_weights(strategy_data, oos_dates)
    oos_weights = oos_weights.loc[oos_dates]
    oos_result = run_backtest(oos_prices, oos_weights, config)

    return compute_return_moments(oos_result.returns.values.astype(np.float64))
