"""Tests for drawdown tracking and control."""

import numpy as np
import pytest

from hedgefund.risk.drawdown import (
    apply_drawdown_scaling,
    compute_drawdown,
    get_drawdown_state,
    position_multiplier,
    update_peak,
)


class TestComputeDrawdown:
    def test_no_drawdown(self) -> None:
        assert compute_drawdown(100, 100) == 0.0

    def test_ten_pct_drawdown(self) -> None:
        assert compute_drawdown(90, 100) == pytest.approx(0.10)

    def test_twenty_pct_drawdown(self) -> None:
        assert compute_drawdown(80, 100) == pytest.approx(0.20)

    def test_zero_peak(self) -> None:
        assert compute_drawdown(50, 0) == 0.0

    def test_above_peak(self) -> None:
        assert compute_drawdown(110, 100) == 0.0


class TestUpdatePeak:
    def test_new_high(self) -> None:
        assert update_peak(110, 100) == 110

    def test_below_peak(self) -> None:
        assert update_peak(90, 100) == 100


class TestPositionMultiplier:
    def test_no_drawdown(self) -> None:
        assert position_multiplier(0.0) == 1.0

    def test_small_drawdown(self) -> None:
        assert position_multiplier(0.03) == 1.0

    def test_5pct_drawdown(self) -> None:
        assert position_multiplier(0.07) == 0.75

    def test_10pct_drawdown(self) -> None:
        assert position_multiplier(0.12) == 0.50

    def test_15pct_drawdown(self) -> None:
        assert position_multiplier(0.18) == 0.25

    def test_max_breached(self) -> None:
        assert position_multiplier(0.25) == 0.0


class TestGetDrawdownState:
    def test_healthy_state(self) -> None:
        state = get_drawdown_state(100, 100)
        assert state.drawdown == 0.0
        assert state.position_multiplier == 1.0
        assert state.is_max_breached is False

    def test_moderate_drawdown(self) -> None:
        state = get_drawdown_state(88, 100)
        assert state.drawdown == pytest.approx(0.12)
        assert state.position_multiplier == 0.50
        assert state.is_max_breached is False

    def test_max_breached(self) -> None:
        state = get_drawdown_state(75, 100)
        assert state.is_max_breached is True
        assert state.position_multiplier == 0.0

    def test_peak_updates(self) -> None:
        state = get_drawdown_state(110, 100)
        assert state.peak_value == 110
        assert state.drawdown == 0.0


class TestApplyDrawdownScaling:
    def test_full_multiplier(self) -> None:
        weights = np.array([0.3, 0.3, 0.4])
        result = apply_drawdown_scaling(weights, 1.0)
        np.testing.assert_array_almost_equal(result, weights)

    def test_half_multiplier(self) -> None:
        weights = np.array([0.3, 0.3, 0.4])
        result = apply_drawdown_scaling(weights, 0.5)
        np.testing.assert_array_almost_equal(result, [0.15, 0.15, 0.2])

    def test_zero_multiplier(self) -> None:
        weights = np.array([0.3, 0.3, 0.4])
        result = apply_drawdown_scaling(weights, 0.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])
