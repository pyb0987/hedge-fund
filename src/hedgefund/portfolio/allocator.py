"""Portfolio allocation — determines how capital is split across strategies.

지원하는 방법:
1. Equal Weight (1/N) — 단순하지만 견고함, 데이터 < 5년일 때 최적
2. Risk Parity — 각 전략이 동일한 리스크를 기여하도록 가중
3. Fixed Weight — 설정 파일의 고정 비율 사용
"""

import numpy as np
from numpy.typing import NDArray


def equal_weight(n_strategies: int) -> NDArray[np.float64]:
    """1/N equal weight allocation."""
    if n_strategies <= 0:
        return np.array([], dtype=np.float64)
    return np.full(n_strategies, 1.0 / n_strategies)


def fixed_weight(weights: list[float]) -> NDArray[np.float64]:
    """Use pre-defined weights from config.

    Normalizes to sum to 1.0 if they don't already.
    """
    arr = np.array(weights, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.zeros_like(arr)
    return arr / total


def risk_parity(
    covariance_matrix: NDArray[np.float64],
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Risk Parity allocation — equal risk contribution from each strategy.

    Each strategy contributes equally to total portfolio volatility.
    More robust than mean-variance with estimation error.

    Args:
        covariance_matrix: N x N covariance matrix of strategy returns
        max_iter: maximum iterations for convergence
        tol: convergence tolerance

    Returns:
        Weight array summing to 1.0
    """
    n = covariance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0])

    weights = np.ones(n) / n

    for _ in range(max_iter):
        port_var = weights @ covariance_matrix @ weights
        if port_var <= 0:
            return equal_weight(n)  # fallback if degenerate

        port_vol = np.sqrt(port_var)
        marginal_risk = covariance_matrix @ weights
        risk_contrib = weights * marginal_risk / port_vol
        target_risk = port_vol / n

        # Avoid division by zero
        risk_contrib_safe = np.where(risk_contrib > 0, risk_contrib, 1e-10)
        adjustment = target_risk / risk_contrib_safe
        # Clip adjustment to prevent numerical explosion when risk_contrib ≈ 0
        adjustment = np.clip(adjustment, 0.01, 100.0)
        new_weights = weights * adjustment
        new_weights = np.maximum(new_weights, 0.0)  # no negative weights
        total = new_weights.sum()
        if total > 0:
            new_weights = new_weights / total

        if np.max(np.abs(new_weights - weights)) < tol:
            break
        weights = new_weights

    return weights


def minimum_variance(
    covariance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Minimum variance portfolio — only needs covariance (no return estimates).

    Solves: min w'Σw s.t. Σw = 1, w >= 0

    Uses analytical solution for long-only:
    w = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
    """
    n = covariance_matrix.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0])

    try:
        inv_cov = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        return equal_weight(n)  # fallback if singular

    ones = np.ones(n)
    raw_weights = inv_cov @ ones
    total = ones @ raw_weights

    if total <= 0:
        return equal_weight(n)

    weights = raw_weights / total

    # Clamp negatives and renormalize (long-only constraint)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights = weights / total

    return weights
