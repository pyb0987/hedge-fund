"""Tests for paper trading validation report."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.data.store import DataStore
from hedgefund.monitoring.paper_report import (
    CostReport,
    PaperReport,
    PerformanceReport,
    ReportThresholds,
    RiskComplianceReport,
    SignalFidelityReport,
    ValidationResult,
    analyze_costs,
    analyze_performance,
    analyze_risk_compliance,
    analyze_signal_fidelity,
    format_report,
    generate_report,
    validate,
)


# --- Signal Fidelity ---


class TestSignalFidelity:
    def test_empty_data(self) -> None:
        signals = pd.DataFrame()
        trades = pd.DataFrame()
        result = analyze_signal_fidelity(signals, trades)
        assert result.total_signals == 0
        assert result.total_trades == 0
        assert result.conversion_rate == 0.0

    def test_full_conversion(self) -> None:
        signals = pd.DataFrame({"symbol": ["A", "B", "C"]})
        trades = pd.DataFrame({"symbol": ["A", "B", "C"]})
        result = analyze_signal_fidelity(signals, trades)
        assert result.conversion_rate == 1.0

    def test_partial_conversion(self) -> None:
        signals = pd.DataFrame({"symbol": ["A", "B", "C", "D"]})
        trades = pd.DataFrame({"symbol": ["A", "B"]})
        result = analyze_signal_fidelity(signals, trades)
        assert result.conversion_rate == 0.5

    def test_more_trades_than_signals_caps_at_one(self) -> None:
        signals = pd.DataFrame({"symbol": ["A"]})
        trades = pd.DataFrame({"symbol": ["A", "B", "C"]})
        result = analyze_signal_fidelity(signals, trades)
        assert result.conversion_rate == 1.0


# --- Cost Analysis ---


class TestCostAnalysis:
    def test_empty_trades(self) -> None:
        trades = pd.DataFrame()
        result = analyze_costs(trades)
        assert result.total_commission == 0.0
        assert result.avg_cost_per_trade == 0.0

    def test_with_trades(self) -> None:
        trades = pd.DataFrame({
            "commission": [100, 200, 300],
            "slippage": [10, 20, 30],
        })
        result = analyze_costs(trades)
        assert result.total_commission == 600
        assert result.total_slippage == 60
        assert result.total_cost == 660
        assert result.avg_cost_per_trade == 220


# --- Performance ---


class TestPerformance:
    def test_empty_snapshots(self) -> None:
        result = analyze_performance(pd.DataFrame())
        assert result.insufficient_data is True
        assert result.num_cycles == 0

    def test_single_snapshot_insufficient(self) -> None:
        df = pd.DataFrame({"total_value": [1000000.0]})
        result = analyze_performance(df)
        assert result.insufficient_data is True

    def test_positive_returns(self) -> None:
        # Simulate steady growth: 1M → 1.1M over 20 snapshots
        values = np.linspace(1_000_000, 1_100_000, 20)
        df = pd.DataFrame({"total_value": values})
        result = analyze_performance(df, ReportThresholds(min_cycles=5))
        assert result.insufficient_data is False
        assert result.total_return == pytest.approx(0.1, rel=0.01)
        assert result.sharpe_ratio > 0
        assert result.max_drawdown == pytest.approx(0.0, abs=0.001)

    def test_with_drawdown(self) -> None:
        # 1M → 900K → 950K
        values = [1_000_000, 980_000, 960_000, 900_000, 920_000, 950_000]
        df = pd.DataFrame({"total_value": values})
        result = analyze_performance(df, ReportThresholds(min_cycles=3))
        assert result.max_drawdown > 0
        assert result.total_return < 0

    def test_insufficient_data_flag(self) -> None:
        values = np.linspace(1_000_000, 1_050_000, 5)
        df = pd.DataFrame({"total_value": values})
        result = analyze_performance(df, ReportThresholds(min_cycles=10))
        assert result.insufficient_data is True
        # Metrics still computed
        assert result.num_cycles == 4


# --- Risk Compliance ---


class TestRiskCompliance:
    def test_empty(self) -> None:
        result = analyze_risk_compliance(pd.DataFrame())
        assert result.max_observed_drawdown == 0.0
        assert result.drawdown_activations == 0
        assert result.risk_rejections == 0
        assert result.cycles_blocked == 0

    def test_no_activations(self) -> None:
        df = pd.DataFrame({"drawdown": [0.01, 0.02, 0.03, 0.04]})
        result = analyze_risk_compliance(df)
        assert result.max_observed_drawdown == pytest.approx(0.04)
        assert result.drawdown_activations == 0

    def test_with_activations(self) -> None:
        df = pd.DataFrame({"drawdown": [0.01, 0.08, 0.12, 0.03]})
        result = analyze_risk_compliance(df)
        assert result.max_observed_drawdown == pytest.approx(0.12)
        assert result.drawdown_activations == 2  # 0.08 and 0.12 > 0.05

    def test_with_risk_events(self) -> None:
        snapshots = pd.DataFrame({"drawdown": [0.01, 0.08]})
        risk_events = pd.DataFrame({
            "event_type": ["execution_failed", "cycle_blocked", "drawdown_check"],
            "passed": [0, 0, 1],
            "rule_name": ["order_rejected", "max_drawdown", "portfolio_drawdown"],
        })
        result = analyze_risk_compliance(snapshots, risk_events)
        assert result.risk_rejections == 1
        assert result.cycles_blocked == 1


# --- Validation ---


class TestValidation:
    def test_all_pass(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=1.0, sortino_ratio=1.5, max_drawdown=0.10,
            profit_factor=1.5, annualized_return=0.15, win_rate=0.55,
            total_return=0.10, num_cycles=20, insufficient_data=False,
        )
        result = validate(perf)
        assert result.all_pass is True

    def test_sharpe_fail(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=0.3, sortino_ratio=0.5, max_drawdown=0.10,
            profit_factor=1.5, annualized_return=0.05, win_rate=0.45,
            total_return=0.03, num_cycles=20, insufficient_data=False,
        )
        result = validate(perf)
        assert result.sharpe_pass is False
        assert result.all_pass is False

    def test_drawdown_fail(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=1.0, sortino_ratio=1.5, max_drawdown=0.20,
            profit_factor=1.5, annualized_return=0.15, win_rate=0.55,
            total_return=0.10, num_cycles=20, insufficient_data=False,
        )
        result = validate(perf)
        assert result.drawdown_pass is False
        assert result.all_pass is False

    def test_insufficient_data_fails(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=2.0, sortino_ratio=3.0, max_drawdown=0.05,
            profit_factor=2.0, annualized_return=0.20, win_rate=0.60,
            total_return=0.15, num_cycles=5, insufficient_data=True,
        )
        result = validate(perf)
        assert result.sufficient_data is False
        assert result.all_pass is False


# --- Integration ---


class TestGenerateReport:
    def test_with_store(self, tmp_path: object) -> None:
        store = DataStore(db_path=str(tmp_path) + "/test.db")  # type: ignore[arg-type]

        # Add some snapshots
        for i in range(15):
            store.save_snapshot(
                timestamp=datetime(2024, 1, 1 + i),
                total_value=1_000_000 + i * 5_000,
                cash=500_000,
                unrealized_pnl=i * 5_000,
                realized_pnl=0,
                drawdown=0.01 * (i % 4),
                peak_value=1_100_000,
            )

        # Add signals and trades
        for i in range(5):
            store.save_signal(
                timestamp=datetime(2024, 1, 1 + i),
                strategy_name="crypto_momentum",
                symbol="KRW-BTC",
                exchange="upbit",
                direction="long",
                strength=0.8,
            )

        for i in range(3):
            store.save_trade(
                symbol="KRW-BTC",
                exchange="upbit",
                side="buy",
                quantity=0.01,
                price=50_000_000,
                commission=1250,
                slippage=750,
                strategy_name="crypto_momentum",
                timestamp=datetime(2024, 1, 1 + i),
            )

        report = generate_report(store, ReportThresholds(min_cycles=10))

        assert report.signal_fidelity.total_signals == 5
        assert report.signal_fidelity.total_trades == 3
        assert report.cost.total_commission == 3750
        assert report.performance.num_cycles == 14
        assert report.data_start is not None


class TestFormatReport:
    def test_contains_sections(self) -> None:
        report = PaperReport(
            signal_fidelity=SignalFidelityReport(10, 8, 0.8),
            cost=CostReport(5000, 3000, 8000, 1000),
            performance=PerformanceReport(
                1.2, 1.5, 0.08, 1.8, 0.15, 0.55, 0.10, 50, False,
            ),
            risk_compliance=RiskComplianceReport(0.08, 3, 0, 0),
            strategy_performance=(),
            rebalancing_gates=(),
            holding_periods=(),
            validation=ValidationResult(True, True, True, True, True),
            generated_at=datetime(2024, 6, 1, 12, 0),
            data_start=datetime(2024, 3, 1),
            data_end=datetime(2024, 6, 1),
        )
        text = format_report(report)
        assert "Signal Fidelity" in text
        assert "Transaction Costs" in text
        assert "Performance" in text
        assert "Risk Compliance" in text
        assert "Go/No-Go" in text
        assert "Overall: GO" in text

    def test_nogo_report(self) -> None:
        report = PaperReport(
            signal_fidelity=SignalFidelityReport(5, 2, 0.4),
            cost=CostReport(1000, 500, 1500, 750),
            performance=PerformanceReport(
                0.3, 0.4, 0.18, 0.9, 0.05, 0.40, -0.02, 8, True,
            ),
            risk_compliance=RiskComplianceReport(0.18, 5, 2, 1),
            strategy_performance=(),
            rebalancing_gates=(),
            holding_periods=(),
            validation=ValidationResult(False, False, False, False, False),
            generated_at=datetime(2024, 6, 1),
            data_start=None,
            data_end=None,
        )
        text = format_report(report)
        assert "Overall: NO-GO" in text
        assert "FAIL" in text
        assert "Insufficient data" in text
