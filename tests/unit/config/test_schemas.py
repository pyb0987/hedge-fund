"""Tests for config schemas."""

import pytest
from pydantic import ValidationError

from hedgefund.config.schemas import (
    AllocationConfig,
    CostConfig,
    CryptoMomentumConfig,
    GlobalSettings,
    PortfolioConfig,
    RiskConfig,
    ValidationConfig,
)


class TestRiskConfig:
    def test_defaults(self) -> None:
        config = RiskConfig()
        assert config.max_portfolio_drawdown == 0.20
        assert config.max_leverage == 1.0

    def test_invalid_drawdown(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(max_portfolio_drawdown=0.0)  # below ge=0.01


class TestAllocationConfig:
    def test_defaults_sum_to_one(self) -> None:
        config = AllocationConfig()
        total = config.crypto_momentum + config.etf_mean_reversion + config.dual_momentum
        assert abs(total - 1.0) < 0.01

    def test_invalid_sum(self) -> None:
        with pytest.raises(ValidationError):
            AllocationConfig(crypto_momentum=0.5, etf_mean_reversion=0.5, dual_momentum=0.5)


class TestCostConfig:
    def test_total_cost(self) -> None:
        config = CostConfig(commission_rate=0.0025, slippage_rate=0.0015)
        assert config.total_cost_rate == pytest.approx(0.004)


class TestCryptoMomentumConfig:
    def test_defaults(self) -> None:
        config = CryptoMomentumConfig()
        assert config.lookback_days == 20
        assert config.top_n == 3

    def test_custom_params(self) -> None:
        config = CryptoMomentumConfig(lookback_days=30, top_n=5)
        assert config.lookback_days == 30
        assert config.top_n == 5

    def test_invalid_lookback(self) -> None:
        with pytest.raises(ValidationError):
            CryptoMomentumConfig(lookback_days=3)  # below ge=5


class TestGlobalSettings:
    def test_valid_config(self) -> None:
        settings = GlobalSettings(
            portfolio=PortfolioConfig(initial_capital=1_000_000)
        )
        assert settings.portfolio.initial_capital == 1_000_000
        assert settings.risk.max_portfolio_drawdown == 0.20

    def test_invalid_capital(self) -> None:
        with pytest.raises(ValidationError):
            GlobalSettings(
                portfolio=PortfolioConfig(initial_capital=100)  # below ge=10000
            )
