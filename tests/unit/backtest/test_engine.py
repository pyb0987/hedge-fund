"""Tests for the backtest engine."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.engine import BacktestConfig, BacktestResult, run_backtest, validate_result


@pytest.fixture
def simple_backtest_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple backtest data: 3 assets, 100 days, equal weight."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=100)

    prices = pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 100)),
        "B": 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 100)),
        "C": 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, 100)),
    }, index=dates)

    # Equal weight, constant (no turnover after first day)
    weights = pd.DataFrame(1 / 3, index=dates, columns=["A", "B", "C"])

    return prices, weights


class TestRunBacktest:
    def test_returns_backtest_result(
        self, simple_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        prices, weights = simple_backtest_data
        result = run_backtest(prices, weights)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(
        self, simple_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        prices, weights = simple_backtest_data
        result = run_backtest(prices, weights)
        assert len(result.equity_curve) == len(prices)

    def test_initial_capital(
        self, simple_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        prices, weights = simple_backtest_data
        config = BacktestConfig(initial_capital=500_000)
        result = run_backtest(prices, weights, config)
        assert result.equity_curve.iloc[0] == pytest.approx(500_000, rel=0.01)

    def test_zero_weights_no_return(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=50)
        prices = pd.DataFrame({"A": np.linspace(100, 200, 50)}, index=dates)
        weights = pd.DataFrame(0.0, index=dates, columns=["A"])

        result = run_backtest(prices, weights)
        assert result.total_return == pytest.approx(0.0, abs=0.001)

    def test_transaction_costs_reduce_returns(
        self, simple_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        prices, weights = simple_backtest_data
        # Add turnover by changing weights
        weights_dynamic = weights.copy()
        for i in range(0, len(weights), 5):
            weights_dynamic.iloc[i] = [0.5, 0.3, 0.2]

        no_cost = run_backtest(
            prices, weights_dynamic, BacktestConfig(commission_rate=0.0, slippage_rate=0.0)
        )
        with_cost = run_backtest(
            prices, weights_dynamic, BacktestConfig(commission_rate=0.005, slippage_rate=0.003)
        )
        assert with_cost.total_return < no_cost.total_return

    def test_metrics_are_finite(
        self, simple_backtest_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        prices, weights = simple_backtest_data
        result = run_backtest(prices, weights)
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.max_drawdown)
        assert 0.0 <= result.max_drawdown <= 1.0
        assert 0.0 <= result.win_rate <= 1.0


class TestValidateResult:
    def test_all_pass(self) -> None:
        # Create a mock result that passes all criteria
        dates = pd.bdate_range("2024-01-01", periods=50)
        result = BacktestResult(
            equity_curve=pd.Series(np.linspace(100, 120, 50), index=dates),
            returns=pd.Series(np.full(50, 0.004), index=dates),
            weights=pd.DataFrame(),
            total_return=0.20,
            annualized_return=0.40,
            annualized_volatility=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.10,
            calmar_ratio=4.0,
            profit_factor=1.8,
            win_rate=0.55,
            num_trades=100,
            turnover=0.05,
            total_commission=1000,
            total_slippage=500,
        )
        validation = validate_result(result)
        assert all(validation.values())

    def test_sharpe_fail(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        result = BacktestResult(
            equity_curve=pd.Series(range(10), index=dates),
            returns=pd.Series(np.zeros(10), index=dates),
            weights=pd.DataFrame(),
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.3,  # below 0.8
            sortino_ratio=0.0,
            max_drawdown=0.05,
            calmar_ratio=0.0,
            profit_factor=2.0,
            win_rate=0.5,
            num_trades=10,
            turnover=0.01,
            total_commission=0,
            total_slippage=0,
        )
        validation = validate_result(result)
        assert validation["sharpe_ratio"] is False
        assert validation["max_drawdown"] is True
        assert validation["profit_factor"] is True
