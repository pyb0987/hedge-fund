"""Tests for backtest report generation."""

import numpy as np
import pandas as pd

from hedgefund.backtest.engine import BacktestResult
from hedgefund.backtest.metrics import generate_report


def _make_result(**overrides) -> BacktestResult:
    dates = pd.bdate_range("2024-01-01", periods=10)
    defaults = dict(
        equity_curve=pd.Series(np.linspace(100, 120, 10), index=dates),
        returns=pd.Series(np.full(10, 0.002), index=dates),
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
        num_trades=50,
        turnover=0.03,
        total_commission=2500,
        total_slippage=1500,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


class TestGenerateReport:
    def test_contains_strategy_name(self) -> None:
        report = generate_report(_make_result(), "Crypto Momentum")
        assert "Crypto Momentum" in report

    def test_contains_metrics(self) -> None:
        report = generate_report(_make_result())
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report
        assert "Profit Factor" in report

    def test_go_decision(self) -> None:
        report = generate_report(_make_result(sharpe_ratio=1.5, max_drawdown=0.10, profit_factor=1.8))
        assert "GO" in report
        assert "PASS" in report

    def test_no_go_decision(self) -> None:
        report = generate_report(_make_result(sharpe_ratio=0.3))
        assert "NO-GO" in report
        assert "FAIL" in report

    def test_contains_costs(self) -> None:
        report = generate_report(_make_result(total_commission=5000, total_slippage=3000))
        assert "Commission" in report
        assert "Slippage" in report
