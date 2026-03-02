"""Tests for portfolio allocator."""

import numpy as np
import pytest

from hedgefund.portfolio.allocator import (
    equal_weight,
    fixed_weight,
    minimum_variance,
    risk_parity,
)


class TestEqualWeight:
    def test_three_strategies(self) -> None:
        w = equal_weight(3)
        np.testing.assert_array_almost_equal(w, [1 / 3, 1 / 3, 1 / 3])

    def test_sums_to_one(self) -> None:
        w = equal_weight(5)
        assert w.sum() == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert len(equal_weight(0)) == 0


class TestFixedWeight:
    def test_normalized(self) -> None:
        w = fixed_weight([0.35, 0.35, 0.30])
        assert w.sum() == pytest.approx(1.0)

    def test_unnormalized(self) -> None:
        w = fixed_weight([2, 3, 5])
        np.testing.assert_array_almost_equal(w, [0.2, 0.3, 0.5])

    def test_all_zero(self) -> None:
        w = fixed_weight([0, 0, 0])
        np.testing.assert_array_almost_equal(w, [0, 0, 0])


class TestRiskParity:
    def test_equal_vol_gives_equal_weight(self) -> None:
        # If all assets have same volatility and zero correlation
        # → risk parity = equal weight
        cov = np.eye(3) * 0.04  # 20% vol each, uncorrelated
        w = risk_parity(cov)
        np.testing.assert_array_almost_equal(w, [1 / 3, 1 / 3, 1 / 3], decimal=2)

    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 3))
        cov = np.cov(data, rowvar=False)
        w = risk_parity(cov)
        assert w.sum() == pytest.approx(1.0, abs=0.01)

    def test_high_vol_gets_less(self) -> None:
        # Asset 0 has 2x vol → should get less weight
        # Add small positive correlations to break symmetry
        cov = np.array([
            [0.16, 0.01, 0.01],
            [0.01, 0.04, 0.005],
            [0.01, 0.005, 0.04],
        ])
        w = risk_parity(cov)
        assert w[0] < w[1]

    def test_single_asset(self) -> None:
        cov = np.array([[0.04]])
        w = risk_parity(cov)
        assert w[0] == pytest.approx(1.0)


class TestMinimumVariance:
    def test_sums_to_one(self) -> None:
        cov = np.diag([0.04, 0.09, 0.01])
        w = minimum_variance(cov)
        assert w.sum() == pytest.approx(1.0, abs=0.01)

    def test_low_vol_gets_more(self) -> None:
        cov = np.diag([0.16, 0.01, 0.09])  # Asset 1 lowest vol
        w = minimum_variance(cov)
        assert w[1] > w[0]

    def test_single_asset(self) -> None:
        cov = np.array([[0.04]])
        w = minimum_variance(cov)
        assert w[0] == pytest.approx(1.0)
