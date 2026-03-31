"""Portfolio-level beta hedging overlay.

Computes rolling beta of portfolio returns against a benchmark (SPY)
and generates hedge weights to neutralize market exposure.

Used by both backtest (weight matrix) and paper/live (signal generation).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BetaHedgeResult:
    """Result of a single-point beta hedge computation."""

    beta: float
    hedge_weight: float  # negative = short
    data_points: int
    is_active: bool


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """Compute rolling beta of portfolio vs benchmark.

    Uses cov(port, bench) / var(bench) on a rolling window.
    Returns NaN for rows with fewer than min_periods data points.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    pr = portfolio_returns.loc[common]
    br = benchmark_returns.loc[common]

    rolling_cov = pr.rolling(window=window, min_periods=min_periods).cov(br)
    rolling_var = br.rolling(window=window, min_periods=min_periods).var()

    # Avoid division by zero
    beta = rolling_cov / rolling_var.replace(0, np.nan)
    return beta.fillna(0.0)


def compute_hedge_weights(
    combined_weights: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    window: int = 60,
    min_periods: int = 20,
    max_hedge_ratio: float = 0.30,
) -> pd.DataFrame:
    """Add beta hedge weights to the combined portfolio weight matrix.

    Computes portfolio returns from weights × price returns (same math
    as backtest engine), then calculates rolling beta and adds a negative
    benchmark weight to neutralize market exposure.

    Args:
        combined_weights: strategy-merged weight matrix (dates × symbols)
        prices: OHLCV or close prices DataFrame (dates × symbols)
        benchmark: benchmark symbol for beta computation
        window: rolling beta window
        min_periods: minimum data points for beta
        max_hedge_ratio: cap on absolute hedge weight

    Returns:
        Modified weight matrix with hedge weights added to benchmark column
    """
    if benchmark not in prices.columns:
        return combined_weights

    result = combined_weights.copy()

    # Compute asset returns from prices
    asset_returns = prices.pct_change().fillna(0.0)
    benchmark_returns = asset_returns[benchmark]

    # Simulate portfolio returns using shifted weights (same as engine.py)
    shifted = combined_weights.shift(1).fillna(0.0)
    common_cols = shifted.columns.intersection(asset_returns.columns)
    portfolio_returns = (shifted[common_cols] * asset_returns[common_cols]).sum(axis=1)

    # Compute rolling beta
    beta_series = rolling_beta(
        portfolio_returns, benchmark_returns, window=window, min_periods=min_periods
    )

    # Compute gross exposure per day for scaling
    gross_exposure = combined_weights.abs().sum(axis=1).replace(0, 1.0)

    # Hedge weight: -beta * gross_exposure, capped
    hedge_raw = -beta_series * gross_exposure
    hedge_capped = hedge_raw.clip(lower=-max_hedge_ratio, upper=0.0)

    # Add/merge hedge into benchmark column
    if benchmark not in result.columns:
        result[benchmark] = 0.0
    result[benchmark] = result[benchmark] + hedge_capped.fillna(0.0)

    return result


def compute_hedge_signal(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
    max_hedge_ratio: float = 0.30,
) -> BetaHedgeResult:
    """Compute single-point beta hedge for paper/live trading.

    Uses the same cov/var math as the rolling version on the last
    `window` data points.

    Args:
        portfolio_returns: array of recent daily portfolio returns
        benchmark_returns: array of recent daily benchmark returns
        window: lookback window for beta
        min_periods: minimum data points required
        max_hedge_ratio: cap on hedge ratio

    Returns:
        BetaHedgeResult with beta and hedge weight
    """
    n = min(len(portfolio_returns), len(benchmark_returns), window)

    if n < min_periods:
        return BetaHedgeResult(beta=0.0, hedge_weight=0.0, data_points=n, is_active=False)

    pr = portfolio_returns[-n:]
    br = benchmark_returns[-n:]

    var_b = float(np.var(br, ddof=1))
    if var_b < 1e-12:
        return BetaHedgeResult(beta=0.0, hedge_weight=0.0, data_points=n, is_active=False)

    cov_pb = float(np.cov(pr, br, ddof=1)[0, 1])
    beta = cov_pb / var_b

    hedge_weight = max(-max_hedge_ratio, -beta)
    hedge_weight = min(hedge_weight, 0.0)  # only short

    return BetaHedgeResult(
        beta=round(beta, 6),
        hedge_weight=round(hedge_weight, 6),
        data_points=n,
        is_active=True,
    )
