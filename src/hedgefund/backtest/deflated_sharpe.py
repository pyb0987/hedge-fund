"""Deflated Sharpe Ratio — multiple testing correction.

Bailey & Lopez de Prado (2014): 여러 전략을 테스트할 때,
가장 좋은 Sharpe가 순전히 운(noise)일 확률을 보정합니다.

p-value > 0.95이면 관측된 Sharpe가 진짜(실력)일 가능성이 높습니다.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def expected_max_sharpe(num_trials: int) -> float:
    """Expected maximum Sharpe ratio under null hypothesis (all noise).

    E[max(SR)] ≈ Z_{1-1/N} * (1 - γ) + γ * Z_{1-1/(N*e)}
    where γ = Euler-Mascheroni constant ≈ 0.5772
    """
    if num_trials <= 1:
        return 0.0

    gamma = 0.5772156649  # Euler-Mascheroni constant
    z1 = stats.norm.ppf(1 - 1 / num_trials)
    z2 = stats.norm.ppf(1 - 1 / (num_trials * math.e))

    return float(z1 * (1 - gamma) + gamma * z2)


def sharpe_standard_error(
    observed_sharpe: float,
    sample_size: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Standard error of the Sharpe ratio.

    Accounts for non-normality via skewness and kurtosis.

    SE(SR) = sqrt((1 - γ₁*SR + (κ-1)/4 * SR²) / T)
    where γ₁ = skewness, κ = kurtosis (excess), T = sample size
    """
    if sample_size <= 1:
        return float("inf")

    sr2 = observed_sharpe ** 2
    numerator = 1 - skewness * observed_sharpe + (excess_kurtosis) / 4 * sr2

    # Ensure non-negative before sqrt
    numerator = max(numerator, 1e-10)
    return math.sqrt(numerator / sample_size)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    sample_size: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio p-value.

    Tests whether the observed Sharpe is statistically significant
    after accounting for multiple testing.

    Args:
        observed_sharpe: best observed Sharpe ratio
        num_trials: number of strategies/parameter sets tested
        sample_size: number of return observations
        skewness: return series skewness (0 = normal)
        excess_kurtosis: return series excess kurtosis (0 = normal)

    Returns:
        p-value in [0, 1]. > 0.95 means likely genuine skill.
    """
    if num_trials <= 0 or sample_size <= 1:
        return 0.0

    e_max_sr = expected_max_sharpe(num_trials)
    se = sharpe_standard_error(observed_sharpe, sample_size, skewness, excess_kurtosis)

    if se <= 0 or se == float("inf"):
        return 0.0

    z_stat = (observed_sharpe - e_max_sr) / se
    return float(stats.norm.cdf(z_stat))


def compute_return_moments(returns: NDArray[np.float64]) -> dict[str, float]:
    """Compute moments needed for Deflated Sharpe calculation."""
    if len(returns) < 4:
        return {"skewness": 0.0, "excess_kurtosis": 0.0}

    return {
        "skewness": float(stats.skew(returns)),
        "excess_kurtosis": float(stats.kurtosis(returns)),  # scipy uses excess kurtosis
    }
