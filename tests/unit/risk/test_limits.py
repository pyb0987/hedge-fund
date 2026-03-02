"""Tests for risk limits."""

import pytest

from hedgefund.config.schemas import RiskConfig
from hedgefund.core.exceptions import OrderRejectedError, PositionLimitError
from hedgefund.risk.limits import (
    check_daily_var_limit,
    check_single_position_limit,
    check_stop_loss,
    check_strategy_allocation_limit,
    enforce_position_limit,
    enforce_strategy_limit,
)


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig()


class TestSinglePositionLimit:
    def test_within_limit(self, config: RiskConfig) -> None:
        result = check_single_position_limit(100_000, 1_000_000, config)
        assert result.passed is True

    def test_exceeds_limit(self, config: RiskConfig) -> None:
        result = check_single_position_limit(200_000, 1_000_000, config)
        assert result.passed is False

    def test_at_limit(self, config: RiskConfig) -> None:
        result = check_single_position_limit(150_000, 1_000_000, config)
        assert result.passed is True

    def test_zero_portfolio(self, config: RiskConfig) -> None:
        result = check_single_position_limit(100, 0, config)
        assert result.passed is False


class TestStrategyAllocationLimit:
    def test_within_limit(self, config: RiskConfig) -> None:
        result = check_strategy_allocation_limit(350_000, 1_000_000, config)
        assert result.passed is True

    def test_exceeds_limit(self, config: RiskConfig) -> None:
        result = check_strategy_allocation_limit(500_000, 1_000_000, config)
        assert result.passed is False


class TestDailyVarLimit:
    def test_within_limit(self, config: RiskConfig) -> None:
        result = check_daily_var_limit(0.015, config)
        assert result.passed is True

    def test_exceeds_limit(self, config: RiskConfig) -> None:
        result = check_daily_var_limit(0.03, config)
        assert result.passed is False


class TestStopLoss:
    def test_no_loss(self, config: RiskConfig) -> None:
        result = check_stop_loss(0.05, config)
        assert result.passed is True

    def test_small_loss(self, config: RiskConfig) -> None:
        result = check_stop_loss(-0.02, config)
        assert result.passed is True

    def test_stop_loss_hit(self, config: RiskConfig) -> None:
        result = check_stop_loss(-0.04, config)
        assert result.passed is False


class TestEnforcement:
    def test_enforce_position_raises(self, config: RiskConfig) -> None:
        with pytest.raises(PositionLimitError):
            enforce_position_limit(200_000, 1_000_000, config)

    def test_enforce_strategy_raises(self, config: RiskConfig) -> None:
        with pytest.raises(OrderRejectedError):
            enforce_strategy_limit(500_000, 1_000_000, config)

    def test_enforce_position_passes(self, config: RiskConfig) -> None:
        enforce_position_limit(100_000, 1_000_000, config)  # should not raise

    def test_enforce_strategy_passes(self, config: RiskConfig) -> None:
        enforce_strategy_limit(350_000, 1_000_000, config)  # should not raise
