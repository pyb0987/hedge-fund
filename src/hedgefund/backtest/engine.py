"""Vectorized backtest engine.

Uses weight matrices for fast backtesting without event loops.
Supports transaction costs, position sizing, and multiple strategies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hedgefund.core import risk_metrics


@dataclass(frozen=True)
class BacktestResult:
    """Immutable backtest result container."""

    # Equity curve
    equity_curve: pd.Series  # daily portfolio value
    returns: pd.Series  # daily returns

    # Weights over time
    weights: pd.DataFrame  # symbol weights per day

    # Metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    num_trades: int
    turnover: float  # average daily turnover

    # Costs
    total_commission: float
    total_slippage: float

    @property
    def total_costs(self) -> float:
        return self.total_commission + self.total_slippage

    @property
    def net_return(self) -> float:
        return self.total_return - self.total_costs


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0025  # 0.25%
    slippage_rate: float = 0.0015  # 0.15%


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run vectorized backtest.

    Args:
        prices: DataFrame of close prices (symbols as columns, DatetimeIndex)
        weights: DataFrame of target weights (same shape as prices, 0.0~1.0)
        config: backtest configuration

    Returns:
        BacktestResult with all metrics computed
    """
    # Align prices and weights
    common_dates = prices.index.intersection(weights.index)
    prices = prices.loc[common_dates]
    weights = weights.loc[common_dates]

    # Compute returns
    asset_returns = prices.pct_change().fillna(0.0)

    # Compute turnover (weight changes → transaction costs)
    weight_changes = weights.diff().fillna(0.0).abs()
    daily_turnover = weight_changes.sum(axis=1)

    # Transaction costs per day
    cost_rate = config.commission_rate + config.slippage_rate
    daily_costs = daily_turnover * cost_rate

    # Portfolio returns = sum(weight * asset_return) - costs
    portfolio_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1) - daily_costs

    # Build equity curve
    equity_curve = config.initial_capital * (1 + portfolio_returns).cumprod()

    # Convert to numpy for metrics
    returns_np: NDArray[np.float64] = portfolio_returns.values.astype(np.float64)

    # Count trades (weight changes > threshold)
    trade_threshold = 0.01
    num_trades = int((weight_changes > trade_threshold).sum().sum())

    # Compute total costs
    total_commission = float((daily_turnover * config.commission_rate).sum() * config.initial_capital)
    total_slippage = float((daily_turnover * config.slippage_rate).sum() * config.initial_capital)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=portfolio_returns,
        weights=weights,
        total_return=float(equity_curve.iloc[-1] / config.initial_capital - 1),
        annualized_return=risk_metrics.annualized_return(returns_np),
        annualized_volatility=risk_metrics.annualized_volatility(returns_np),
        sharpe_ratio=risk_metrics.sharpe_ratio(returns_np),
        sortino_ratio=risk_metrics.sortino_ratio(returns_np),
        max_drawdown=risk_metrics.max_drawdown(returns_np),
        calmar_ratio=risk_metrics.calmar_ratio(returns_np),
        profit_factor=risk_metrics.profit_factor(returns_np),
        win_rate=risk_metrics.win_rate(returns_np),
        num_trades=num_trades,
        turnover=float(daily_turnover.mean()),
        total_commission=total_commission,
        total_slippage=total_slippage,
    )


def validate_result(
    result: BacktestResult,
    min_sharpe: float = 0.8,
    max_drawdown: float = 0.20,
    min_profit_factor: float = 1.3,
) -> dict[str, bool]:
    """Validate backtest result against Go/No-Go criteria.

    Returns dict of criterion name → pass/fail.
    """
    return {
        "sharpe_ratio": result.sharpe_ratio >= min_sharpe,
        "max_drawdown": result.max_drawdown <= max_drawdown,
        "profit_factor": result.profit_factor >= min_profit_factor,
    }
