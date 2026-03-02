"""Rolling strategy correlation monitor.

전략 간 상관관계를 모니터링하여 분산 효과가 유지되는지 확인합니다.
상관관계 > 0.3이면 경고, > 0.5이면 위험 (분산 효과 상실).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

CORRELATION_WARNING = 0.3
CORRELATION_DANGER = 0.5


@dataclass(frozen=True)
class CorrelationReport:
    """Strategy correlation analysis report."""

    correlation_matrix: NDArray[np.float64]
    strategy_names: tuple[str, ...]
    max_off_diagonal: float
    warnings: tuple[str, ...]
    is_diversified: bool  # True if all off-diagonal < WARNING threshold


def compute_correlation_matrix(
    strategy_returns: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute pairwise Pearson correlation between strategies."""
    names = list(strategy_returns.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            r_i = strategy_returns[names[i]]
            r_j = strategy_returns[names[j]]
            min_len = min(len(r_i), len(r_j))
            if min_len < 10:
                matrix[i][j] = 0.0
            else:
                matrix[i][j] = float(np.corrcoef(r_i[:min_len], r_j[:min_len])[0, 1])

    return matrix


def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling correlation between two return series."""
    return returns_a.rolling(window).corr(returns_b)


def analyze_correlations(
    strategy_returns: dict[str, NDArray[np.float64]],
) -> CorrelationReport:
    """Full correlation analysis with warnings."""
    names = list(strategy_returns.keys())
    matrix = compute_correlation_matrix(strategy_returns)

    warnings: list[str] = []
    max_off_diag = 0.0
    n = len(names)

    for i in range(n):
        for j in range(i + 1, n):
            corr = abs(matrix[i][j])
            max_off_diag = max(max_off_diag, corr)

            if corr >= CORRELATION_DANGER:
                warnings.append(
                    f"DANGER: {names[i]} ↔ {names[j]} correlation = {matrix[i][j]:.2f} "
                    f"(> {CORRELATION_DANGER})"
                )
            elif corr >= CORRELATION_WARNING:
                warnings.append(
                    f"WARNING: {names[i]} ↔ {names[j]} correlation = {matrix[i][j]:.2f} "
                    f"(> {CORRELATION_WARNING})"
                )

    return CorrelationReport(
        correlation_matrix=matrix,
        strategy_names=tuple(names),
        max_off_diagonal=max_off_diag,
        warnings=tuple(warnings),
        is_diversified=max_off_diag < CORRELATION_WARNING,
    )


def compute_covariance_matrix(
    strategy_returns: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute covariance matrix for portfolio allocation.

    Used by risk_parity and minimum_variance allocators.
    """
    names = list(strategy_returns.keys())
    min_len = min(len(r) for r in strategy_returns.values())
    data = np.column_stack([strategy_returns[n][:min_len] for n in names])
    return np.cov(data, rowvar=False)
