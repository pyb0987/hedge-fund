"""Tests for portfolio rebalancer."""

import numpy as np
import pytest

from hedgefund.portfolio.rebalancer import apply_target_weights, compute_rebalance


class TestComputeRebalance:
    def test_no_rebalance_needed(self) -> None:
        current = {"crypto_momentum": 0.35, "etf_mean_reversion": 0.35, "dual_momentum": 0.30}
        target = {"crypto_momentum": 0.35, "etf_mean_reversion": 0.35, "dual_momentum": 0.30}
        result = compute_rebalance(current, target, 1_000_000)
        assert result.needs_rebalancing is False
        assert len(result.trades) == 0

    def test_rebalance_with_drift(self) -> None:
        current = {"crypto_momentum": 0.40, "etf_mean_reversion": 0.30, "dual_momentum": 0.30}
        target = {"crypto_momentum": 0.35, "etf_mean_reversion": 0.35, "dual_momentum": 0.30}
        result = compute_rebalance(current, target, 1_000_000)
        assert result.needs_rebalancing is True
        assert len(result.trades) == 2  # crypto sells, etf buys

    def test_small_drift_ignored(self) -> None:
        current = {"a": 0.34, "b": 0.36, "c": 0.30}
        target = {"a": 0.35, "b": 0.35, "c": 0.30}
        result = compute_rebalance(current, target, 1_000_000)
        assert result.needs_rebalancing is False

    def test_cost_estimation(self) -> None:
        current = {"a": 0.50, "b": 0.50}
        target = {"a": 0.30, "b": 0.70}
        result = compute_rebalance(current, target, 1_000_000, cost_rate=0.004)
        assert result.estimated_cost > 0
        assert result.total_turnover == pytest.approx(0.40)  # 0.20 sell + 0.20 buy

    def test_min_trade_value_filter(self) -> None:
        current = {"a": 0.35}
        target = {"a": 0.30}
        # 5% of 10,000 = 500 KRW < min_trade_value of 5000
        result = compute_rebalance(current, target, 10_000, min_trade_value=5000)
        assert result.needs_rebalancing is False


class TestApplyTargetWeights:
    def test_weighted_combination(self) -> None:
        strategy_weights = {
            "crypto": np.array([0.5, 0.3, 0.2]),
            "etf": np.array([0.0, 0.6, 0.4]),
        }
        allocation = {"crypto": 0.6, "etf": 0.4}
        combined = apply_target_weights(strategy_weights, allocation)
        expected = np.array([0.3, 0.42, 0.28])
        np.testing.assert_array_almost_equal(combined, expected)

    def test_empty(self) -> None:
        result = apply_target_weights({}, {})
        assert len(result) == 0
