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
    analyze_strategy_sharpes,
    build_strategy_daily_returns,
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
        trades = pd.DataFrame(
            {
                "commission": [100, 200, 300],
                "slippage": [10, 20, 30],
            }
        )
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
        risk_events = pd.DataFrame(
            {
                "event_type": ["execution_failed", "cycle_blocked", "drawdown_check"],
                "passed": [0, 0, 1],
                "rule_name": ["order_rejected", "max_drawdown", "portfolio_drawdown"],
            }
        )
        result = analyze_risk_compliance(snapshots, risk_events)
        assert result.risk_rejections == 1
        assert result.cycles_blocked == 1


# --- Validation ---


class TestValidation:
    def test_all_pass(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=1.0,
            sortino_ratio=1.5,
            max_drawdown=0.10,
            profit_factor=1.5,
            annualized_return=0.15,
            win_rate=0.55,
            total_return=0.10,
            num_cycles=20,
            insufficient_data=False,
        )
        result = validate(perf)
        assert result.all_pass is True

    def test_sharpe_fail(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=0.3,
            sortino_ratio=0.5,
            max_drawdown=0.10,
            profit_factor=1.5,
            annualized_return=0.05,
            win_rate=0.45,
            total_return=0.03,
            num_cycles=20,
            insufficient_data=False,
        )
        result = validate(perf)
        assert result.sharpe_pass is False
        assert result.all_pass is False

    def test_drawdown_fail(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=1.0,
            sortino_ratio=1.5,
            max_drawdown=0.20,
            profit_factor=1.5,
            annualized_return=0.15,
            win_rate=0.55,
            total_return=0.10,
            num_cycles=20,
            insufficient_data=False,
        )
        result = validate(perf)
        assert result.drawdown_pass is False
        assert result.all_pass is False

    def test_insufficient_data_fails(self) -> None:
        perf = PerformanceReport(
            sharpe_ratio=2.0,
            sortino_ratio=3.0,
            max_drawdown=0.05,
            profit_factor=2.0,
            annualized_return=0.20,
            win_rate=0.60,
            total_return=0.15,
            num_cycles=5,
            insufficient_data=True,
        )
        result = validate(perf)
        assert result.sufficient_data is False
        assert result.all_pass is False


# --- Strategy Daily Returns & Sharpe ---


def _make_position_snapshots(
    strategies: dict[str, list[float]],
    start_date: datetime = datetime(2024, 1, 1),
) -> pd.DataFrame:
    """Helper: build position_snapshots DataFrame for testing."""
    rows = []
    for name, values in strategies.items():
        for i, mv in enumerate(values):
            rows.append(
                {
                    "timestamp": datetime(start_date.year, start_date.month, start_date.day + i),
                    "strategy_name": name,
                    "symbol": f"{name}_SYM",
                    "exchange": "test",
                    "quantity": 1.0,
                    "avg_entry_price": 100.0,
                    "market_price": mv,
                    "market_value": mv,
                    "unrealized_pnl": mv - 100.0,
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


class TestBuildStrategyDailyReturns:
    def test_empty(self) -> None:
        result = build_strategy_daily_returns(pd.DataFrame())
        assert result == {}

    def test_single_day_returns_empty(self) -> None:
        df = _make_position_snapshots({"A": [100.0]})
        result = build_strategy_daily_returns(df)
        assert result == {}

    def test_two_strategies(self) -> None:
        df = _make_position_snapshots(
            {
                "crypto": [100.0, 110.0, 121.0],
                "etf": [200.0, 198.0, 202.0],
            }
        )
        result = build_strategy_daily_returns(df)
        assert "crypto" in result
        assert "etf" in result
        # crypto: 10/100=0.1, 11/110=0.1 → 2 returns
        assert len(result["crypto"]) == 2
        assert result["crypto"][0] == pytest.approx(0.1)

    def test_skips_zero_values(self) -> None:
        """Strategy with gap (sold all, re-entered) computes returns only from nonzero days."""
        df = _make_position_snapshots({"A": [100.0, 0.0, 0.0, 120.0, 126.0]})
        result = build_strategy_daily_returns(df)
        # nonzero: [100, 120, 126] → returns [0.2, 0.05]
        assert len(result["A"]) == 2
        assert result["A"][0] == pytest.approx(0.2)
        assert result["A"][1] == pytest.approx(0.05)


class TestAnalyzeStrategySharpes:
    def test_empty(self) -> None:
        result = analyze_strategy_sharpes({})
        assert result == ()

    def test_insufficient_data_zeroed(self) -> None:
        """Strategies with < MIN_DAYS_FOR_ANNUALIZATION returns get zeroed metrics."""
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
        result = analyze_strategy_sharpes({"crypto": returns})
        assert len(result) == 1
        assert result[0].strategy_name == "crypto"
        assert result[0].sharpe_ratio == 0.0
        assert result[0].num_days == 5

    def test_sufficient_data(self) -> None:
        """Strategies with >= MIN_DAYS_FOR_ANNUALIZATION returns get real metrics."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=15)
        result = analyze_strategy_sharpes({"crypto": returns})
        assert len(result) == 1
        assert result[0].sharpe_ratio != 0.0
        assert result[0].num_days == 15

    def test_multiple_strategies_sorted(self) -> None:
        rng = np.random.default_rng(42)
        result = analyze_strategy_sharpes(
            {
                "z_strat": rng.normal(0.001, 0.01, size=12),
                "a_strat": rng.normal(0.001, 0.01, size=12),
            }
        )
        assert len(result) == 2
        assert result[0].strategy_name == "a_strat"
        assert result[1].strategy_name == "z_strat"


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
                1.2,
                1.5,
                0.08,
                1.8,
                0.15,
                0.55,
                0.10,
                50,
                False,
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
                0.3,
                0.4,
                0.18,
                0.9,
                0.05,
                0.40,
                -0.02,
                8,
                True,
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
