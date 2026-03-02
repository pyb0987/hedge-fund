"""Pure functions for risk and performance metrics.

All functions take numpy arrays of returns and produce scalar metrics.
No side effects, no external dependencies beyond numpy.
"""

import numpy as np
from numpy.typing import NDArray

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.04  # 4% (US T-bill proxy)


def annualized_return(returns: NDArray[np.float64]) -> float:
    """Compute annualized return from daily returns."""
    if len(returns) == 0:
        return 0.0
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    if n_days == 0:
        return 0.0
    return float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


def annualized_volatility(returns: NDArray[np.float64]) -> float:
    """Compute annualized volatility from daily returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """Annualized Sharpe ratio.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    """
    vol = annualized_volatility(returns)
    if vol == 0.0:
        return 0.0
    ann_ret = annualized_return(returns)
    return (ann_ret - risk_free_rate) / vol


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """Annualized Sortino ratio — penalizes only downside volatility.

    Sortino = (annualized_return - risk_free_rate) / downside_deviation
    """
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0.0:
        return 0.0
    ann_ret = annualized_return(returns)
    return (ann_ret - risk_free_rate) / downside_std


def max_drawdown(returns: NDArray[np.float64]) -> float:
    """Maximum drawdown from peak as a positive fraction (0.0 ~ 1.0).

    Returns 0.0 if no drawdown occurred.
    """
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return float(np.max(drawdowns))


def calmar_ratio(returns: NDArray[np.float64]) -> float:
    """Calmar ratio = annualized_return / max_drawdown.

    Higher is better. Returns 0.0 if no drawdown.
    """
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    ann_ret = annualized_return(returns)
    return ann_ret / mdd


def profit_factor(returns: NDArray[np.float64]) -> float:
    """Profit Factor = sum(gains) / abs(sum(losses)).

    > 1.0 means profitable. Returns 0.0 if no losses.
    """
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    total_loss = float(np.abs(np.sum(losses)))
    if total_loss == 0.0:
        return float("inf") if len(gains) > 0 else 0.0
    return float(np.sum(gains)) / total_loss


def rolling_drawdown(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute drawdown series from returns.

    Returns array of drawdowns (positive values, same length as input).
    """
    if len(returns) == 0:
        return np.array([], dtype=np.float64)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    return (running_max - cumulative) / running_max


def value_at_risk(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Historical Value at Risk at given confidence level.

    Returns a positive number representing the loss threshold.
    """
    if len(returns) == 0:
        return 0.0
    percentile = (1 - confidence) * 100
    return float(-np.percentile(returns, percentile))


def win_rate(returns: NDArray[np.float64]) -> float:
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))
