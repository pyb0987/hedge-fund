"""Tests for correlation monitor."""

import numpy as np
import pytest

from hedgefund.portfolio.correlation import (
    CORRELATION_WARNING,
    analyze_correlations,
    compute_correlation_matrix,
    compute_covariance_matrix,
)


@pytest.fixture
def uncorrelated_returns() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "strategy_a": rng.normal(0.001, 0.01, 200),
        "strategy_b": rng.normal(0.0005, 0.015, 200),
        "strategy_c": rng.normal(0.0008, 0.012, 200),
    }


@pytest.fixture
def correlated_returns() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    base = rng.normal(0.001, 0.01, 200)
    return {
        "strategy_a": base,
        "strategy_b": base + rng.normal(0, 0.001, 200),  # highly correlated with A
        "strategy_c": rng.normal(0.0008, 0.012, 200),  # independent
    }


class TestCorrelationMatrix:
    def test_diagonal_is_one(self, uncorrelated_returns: dict) -> None:
        matrix = compute_correlation_matrix(uncorrelated_returns)
        for i in range(3):
            assert matrix[i][i] == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self, uncorrelated_returns: dict) -> None:
        matrix = compute_correlation_matrix(uncorrelated_returns)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_uncorrelated_low(self, uncorrelated_returns: dict) -> None:
        matrix = compute_correlation_matrix(uncorrelated_returns)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(matrix[i][j]) < 0.3


class TestAnalyzeCorrelations:
    def test_diversified(self, uncorrelated_returns: dict) -> None:
        report = analyze_correlations(uncorrelated_returns)
        assert bool(report.is_diversified) is True
        assert len(report.warnings) == 0

    def test_correlated_warns(self, correlated_returns: dict) -> None:
        report = analyze_correlations(correlated_returns)
        assert bool(report.is_diversified) is False
        assert len(report.warnings) > 0

    def test_report_structure(self, uncorrelated_returns: dict) -> None:
        report = analyze_correlations(uncorrelated_returns)
        assert len(report.strategy_names) == 3
        assert report.correlation_matrix.shape == (3, 3)


class TestCovarianceMatrix:
    def test_shape(self, uncorrelated_returns: dict) -> None:
        cov = compute_covariance_matrix(uncorrelated_returns)
        assert cov.shape == (3, 3)

    def test_positive_diagonal(self, uncorrelated_returns: dict) -> None:
        cov = compute_covariance_matrix(uncorrelated_returns)
        for i in range(3):
            assert cov[i][i] > 0


class TestCorrelationNaN:
    def test_constant_returns_no_nan(self) -> None:
        """Constant returns (std=0) should return 0 correlation, not NaN."""
        returns = {
            "strategy_a": np.zeros(100),  # constant zero
            "strategy_b": np.random.default_rng(42).normal(0, 0.01, 100),
        }
        matrix = compute_correlation_matrix(returns)
        assert not np.any(np.isnan(matrix))
        # Constant vs anything should give 0 correlation
        assert matrix[0][1] == 0.0
        assert matrix[1][0] == 0.0

    def test_diagonal_always_one(self) -> None:
        """Diagonal should be 1.0 even for constant returns."""
        returns = {
            "strategy_a": np.zeros(100),
            "strategy_b": np.zeros(100),
        }
        matrix = compute_correlation_matrix(returns)
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
