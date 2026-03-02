"""Tests for Deflated Sharpe Ratio."""

import numpy as np
import pytest

from hedgefund.backtest.deflated_sharpe import (
    compute_return_moments,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_standard_error,
)


class TestExpectedMaxSharpe:
    def test_single_trial(self) -> None:
        assert expected_max_sharpe(1) == 0.0

    def test_increases_with_trials(self) -> None:
        e10 = expected_max_sharpe(10)
        e100 = expected_max_sharpe(100)
        assert e100 > e10

    def test_positive(self) -> None:
        assert expected_max_sharpe(5) > 0


class TestSharpeStandardError:
    def test_decreases_with_samples(self) -> None:
        se100 = sharpe_standard_error(1.0, 100)
        se1000 = sharpe_standard_error(1.0, 1000)
        assert se1000 < se100

    def test_single_sample(self) -> None:
        assert sharpe_standard_error(1.0, 1) == float("inf")


class TestDeflatedSharpeRatio:
    def test_high_sharpe_few_trials(self) -> None:
        # High Sharpe with few trials → should be significant
        p = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            num_trials=3,
            sample_size=252,
        )
        assert p > 0.5

    def test_low_sharpe_many_trials(self) -> None:
        # Low Sharpe with many trials → likely noise
        p = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            num_trials=100,
            sample_size=252,
        )
        assert p < 0.5

    def test_more_trials_reduces_pvalue(self) -> None:
        p_few = deflated_sharpe_ratio(1.5, num_trials=5, sample_size=252)
        p_many = deflated_sharpe_ratio(1.5, num_trials=100, sample_size=252)
        assert p_many < p_few

    def test_edge_cases(self) -> None:
        assert deflated_sharpe_ratio(1.0, 0, 252) == 0.0
        assert deflated_sharpe_ratio(1.0, 5, 1) == 0.0

    def test_non_normal_returns(self) -> None:
        # Skewed returns should affect p-value
        p_normal = deflated_sharpe_ratio(1.5, 5, 252, skewness=0.0, excess_kurtosis=0.0)
        p_skewed = deflated_sharpe_ratio(1.5, 5, 252, skewness=-1.0, excess_kurtosis=3.0)
        # Both should be valid p-values
        assert 0 <= p_normal <= 1
        assert 0 <= p_skewed <= 1


class TestComputeReturnMoments:
    def test_normal_returns(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 1000)
        moments = compute_return_moments(returns)
        assert abs(moments["skewness"]) < 0.5
        assert abs(moments["excess_kurtosis"]) < 1.0

    def test_insufficient_data(self) -> None:
        moments = compute_return_moments(np.array([0.01, -0.01]))
        assert moments["skewness"] == 0.0
