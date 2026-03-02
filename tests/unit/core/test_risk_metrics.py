"""Tests for risk metrics — pure math functions."""

import numpy as np
import pytest

from hedgefund.core import risk_metrics


class TestAnnualizedReturn:
    def test_positive_returns(self, sample_returns: np.ndarray) -> None:
        result = risk_metrics.annualized_return(sample_returns)
        # With ~15% target annual return, should be in reasonable range
        assert -0.5 < result < 1.0

    def test_empty_returns(self) -> None:
        assert risk_metrics.annualized_return(np.array([])) == 0.0

    def test_zero_returns(self, flat_returns: np.ndarray) -> None:
        assert risk_metrics.annualized_return(flat_returns) == pytest.approx(0.0)


class TestAnnualizedVolatility:
    def test_positive_volatility(self, sample_returns: np.ndarray) -> None:
        vol = risk_metrics.annualized_volatility(sample_returns)
        # With ~20% target volatility, should be in range
        assert 0.05 < vol < 0.50

    def test_insufficient_data(self) -> None:
        assert risk_metrics.annualized_volatility(np.array([0.01])) == 0.0

    def test_zero_volatility(self, flat_returns: np.ndarray) -> None:
        assert risk_metrics.annualized_volatility(flat_returns) == 0.0


class TestSharpeRatio:
    def test_positive_sharpe(self, sample_returns: np.ndarray) -> None:
        sharpe = risk_metrics.sharpe_ratio(sample_returns)
        # Should be reasonable for positive-return strategy
        assert -2.0 < sharpe < 5.0

    def test_zero_vol_returns_zero(self, flat_returns: np.ndarray) -> None:
        assert risk_metrics.sharpe_ratio(flat_returns) == 0.0

    def test_custom_risk_free(self, sample_returns: np.ndarray) -> None:
        sharpe_default = risk_metrics.sharpe_ratio(sample_returns)
        sharpe_zero_rf = risk_metrics.sharpe_ratio(sample_returns, risk_free_rate=0.0)
        # Lower risk-free rate → higher Sharpe
        assert sharpe_zero_rf >= sharpe_default


class TestSortinoRatio:
    def test_sortino_computed(self, sample_returns: np.ndarray) -> None:
        sortino = risk_metrics.sortino_ratio(sample_returns)
        # Sortino typically >= Sharpe for same returns
        assert sortino != 0.0

    def test_all_positive_returns(self) -> None:
        returns = np.array([0.01, 0.02, 0.005, 0.015])
        # Only 1 negative return (none) → not enough downside
        assert risk_metrics.sortino_ratio(returns) == 0.0


class TestMaxDrawdown:
    def test_drawdown_range(self, sample_returns: np.ndarray) -> None:
        mdd = risk_metrics.max_drawdown(sample_returns)
        assert 0.0 <= mdd <= 1.0

    def test_losing_strategy_higher_dd(
        self, sample_returns: np.ndarray, losing_returns: np.ndarray
    ) -> None:
        mdd_good = risk_metrics.max_drawdown(sample_returns)
        mdd_bad = risk_metrics.max_drawdown(losing_returns)
        assert mdd_bad > mdd_good

    def test_empty_returns(self) -> None:
        assert risk_metrics.max_drawdown(np.array([])) == 0.0

    def test_always_positive(self) -> None:
        returns = np.array([0.01] * 100)
        assert risk_metrics.max_drawdown(returns) == pytest.approx(0.0)


class TestCalmarRatio:
    def test_calmar_computed(self, sample_returns: np.ndarray) -> None:
        calmar = risk_metrics.calmar_ratio(sample_returns)
        assert calmar != 0.0

    def test_no_drawdown(self) -> None:
        returns = np.array([0.01] * 100)
        assert risk_metrics.calmar_ratio(returns) == 0.0


class TestProfitFactor:
    def test_profitable_strategy(self, sample_returns: np.ndarray) -> None:
        pf = risk_metrics.profit_factor(sample_returns)
        assert pf > 0

    def test_all_gains(self) -> None:
        returns = np.array([0.01, 0.02, 0.005])
        assert risk_metrics.profit_factor(returns) == float("inf")

    def test_no_gains_no_losses(self) -> None:
        returns = np.array([0.0, 0.0, 0.0])
        assert risk_metrics.profit_factor(returns) == 0.0


class TestRollingDrawdown:
    def test_shape_matches(self, sample_returns: np.ndarray) -> None:
        dd = risk_metrics.rolling_drawdown(sample_returns)
        assert len(dd) == len(sample_returns)

    def test_values_non_negative(self, sample_returns: np.ndarray) -> None:
        dd = risk_metrics.rolling_drawdown(sample_returns)
        assert np.all(dd >= 0)


class TestValueAtRisk:
    def test_var_positive(self, sample_returns: np.ndarray) -> None:
        var = risk_metrics.value_at_risk(sample_returns, 0.95)
        # 95% VaR should be a positive loss number
        assert var > 0

    def test_higher_confidence_higher_var(self, sample_returns: np.ndarray) -> None:
        var_95 = risk_metrics.value_at_risk(sample_returns, 0.95)
        var_99 = risk_metrics.value_at_risk(sample_returns, 0.99)
        assert var_99 >= var_95


class TestWinRate:
    def test_win_rate_range(self, sample_returns: np.ndarray) -> None:
        wr = risk_metrics.win_rate(sample_returns)
        assert 0.0 <= wr <= 1.0

    def test_all_wins(self) -> None:
        returns = np.array([0.01, 0.02, 0.005])
        assert risk_metrics.win_rate(returns) == 1.0

    def test_empty(self) -> None:
        assert risk_metrics.win_rate(np.array([])) == 0.0
