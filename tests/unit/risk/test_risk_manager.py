"""Tests for the risk manager orchestrator."""

from datetime import datetime

import numpy as np
import pytest

from hedgefund.config.schemas import RiskConfig
from hedgefund.risk.manager import RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(config=RiskConfig(), initial_capital=1_000_000)


@pytest.fixture
def normal_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.01, 60)


class TestRiskManager:
    def test_pre_trade_passes(
        self, risk_manager: RiskManager, normal_returns: np.ndarray
    ) -> None:
        report = risk_manager.pre_trade_check(
            order_value=100_000,
            strategy_total_value=200_000,
            portfolio_value=1_000_000,
            portfolio_returns=normal_returns,
            timestamp=datetime(2024, 6, 1),
        )
        assert report.all_passed is True
        assert report.blocked_reason is None

    def test_pre_trade_position_limit_fail(
        self, risk_manager: RiskManager, normal_returns: np.ndarray
    ) -> None:
        report = risk_manager.pre_trade_check(
            order_value=200_000,
            strategy_total_value=100_000,
            portfolio_value=1_000_000,
            portfolio_returns=normal_returns,
            timestamp=datetime(2024, 6, 1),
        )
        assert report.all_passed is False

    def test_pre_trade_strategy_limit_fail(
        self, risk_manager: RiskManager, normal_returns: np.ndarray
    ) -> None:
        report = risk_manager.pre_trade_check(
            order_value=100_000,
            strategy_total_value=350_000,
            portfolio_value=1_000_000,
            portfolio_returns=normal_returns,
            timestamp=datetime(2024, 6, 1),
        )
        # Strategy total after order = 450K / 1M = 45% > 40%
        assert report.all_passed is False

    def test_drawdown_tracking(self, risk_manager: RiskManager) -> None:
        state = risk_manager.get_drawdown_state(900_000)
        assert state.drawdown == pytest.approx(0.10)
        assert state.position_multiplier == 0.75

    def test_peak_update(self, risk_manager: RiskManager) -> None:
        risk_manager.update_peak(1_100_000)
        state = risk_manager.get_drawdown_state(1_000_000)
        # DD = (1.1M - 1M) / 1.1M ≈ 9.09%
        expected_dd = (1_100_000 - 1_000_000) / 1_100_000
        assert state.drawdown == pytest.approx(expected_dd, abs=0.01)

    def test_post_trade_check(self, risk_manager: RiskManager) -> None:
        positions = [
            {"value": 100_000, "strategy": "crypto_momentum", "pnl_pct": 0.05},
            {"value": 80_000, "strategy": "etf_mean_reversion", "pnl_pct": -0.02},
        ]
        results = risk_manager.post_trade_check(positions, 1_000_000, datetime(2024, 6, 1))
        assert len(results) == 4  # 2 positions × 2 checks each

    def test_post_trade_stop_loss_triggered(self, risk_manager: RiskManager) -> None:
        positions = [
            {"value": 100_000, "strategy": "test", "pnl_pct": -0.05},  # > 3% loss
        ]
        results = risk_manager.post_trade_check(positions, 1_000_000, datetime(2024, 6, 1))
        stop_check = [r for r in results if r.rule_name == "stop_loss"]
        assert any(not r.passed for r in stop_check)

    def test_failed_checks_property(
        self, risk_manager: RiskManager, normal_returns: np.ndarray
    ) -> None:
        report = risk_manager.pre_trade_check(
            order_value=200_000,
            strategy_total_value=100_000,
            portfolio_value=1_000_000,
            portfolio_returns=normal_returns,
            timestamp=datetime(2024, 6, 1),
        )
        assert len(report.failed_checks) > 0
