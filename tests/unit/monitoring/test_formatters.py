"""Tests for Telegram message formatters."""

from datetime import datetime

from hedgefund.monitoring.formatters import (
    format_cycle_html,
    format_daily_digest_html,
    format_risk_alert_html,
)
from hedgefund.monitoring.paper_report import (
    CostReport,
    PaperReport,
    PerformanceReport,
    RiskComplianceReport,
    SignalFidelityReport,
    ValidationResult,
)


class TestFormatCycleHtml:
    def test_basic_cycle(self) -> None:
        msg = format_cycle_html(
            timestamp=datetime(2024, 6, 1, 12, 0),
            signals=5,
            trades=3,
            commission=1500,
            portfolio_value=1_000_000,
            risk_blocked=False,
            risk_reason=None,
            errors=(),
        )
        assert "Paper Trading Cycle" in msg
        assert "Signals: 5" in msg
        assert "Trades: 3" in msg
        assert "Status: OK" in msg

    def test_risk_blocked(self) -> None:
        msg = format_cycle_html(
            timestamp=datetime(2024, 6, 1, 12, 0),
            signals=0,
            trades=0,
            commission=0,
            portfolio_value=900_000,
            risk_blocked=True,
            risk_reason="Max drawdown breached",
            errors=(),
        )
        assert "BLOCKED" in msg
        assert "Risk Alert" in msg
        assert "Max drawdown breached" in msg

    def test_with_errors(self) -> None:
        msg = format_cycle_html(
            timestamp=datetime(2024, 6, 1, 12, 0),
            signals=2,
            trades=1,
            commission=500,
            portfolio_value=1_000_000,
            risk_blocked=False,
            risk_reason=None,
            errors=("Failed to fetch KRW-ETH", "Timeout on SPY"),
        )
        assert "Warnings (2)" in msg
        assert "Failed to fetch KRW-ETH" in msg

    def test_escapes_html(self) -> None:
        msg = format_cycle_html(
            timestamp=datetime(2024, 6, 1),
            signals=0,
            trades=0,
            commission=0,
            portfolio_value=0,
            risk_blocked=True,
            risk_reason="<script>alert('xss')</script>",
            errors=(),
        )
        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg


class TestFormatRiskAlertHtml:
    def test_basic(self) -> None:
        msg = format_risk_alert_html(
            reason="Drawdown 15% breached",
            drawdown=0.15,
            timestamp=datetime(2024, 6, 1, 12, 0),
        )
        assert "RISK ALERT" in msg
        assert "15.0%" in msg
        assert "Drawdown 15% breached" in msg


class TestFormatDailyDigestHtml:
    def test_go_report_contains_all_sections(self) -> None:
        report = PaperReport(
            signal_fidelity=SignalFidelityReport(20, 15, 0.75),
            cost=CostReport(5000, 3000, 8000, 400),
            performance=PerformanceReport(
                1.2, 1.5, 0.08, 1.8, 0.15, 0.55, 0.10, 50, False,
            ),
            risk_compliance=RiskComplianceReport(0.08, 3, 0, 0),
            validation=ValidationResult(True, True, True, True, True),
            generated_at=datetime(2024, 6, 1),
            data_start=datetime(2024, 3, 1),
            data_end=datetime(2024, 6, 1),
        )
        msg = format_daily_digest_html(report)
        # All 4 validation sections present
        assert "Signal Fidelity" in msg
        assert "Cost Accuracy" in msg
        assert "Performance" in msg
        assert "Risk Compliance" in msg
        # Key metrics
        assert "Sharpe: 1.20" in msg
        assert "Overall: GO" in msg
        # Cost data
        assert "Commission:" in msg
        assert "Slippage:" in msg
        # Risk data
        assert "Max Observed DD:" in msg
        assert "DD Activations:" in msg

    def test_nogo_report(self) -> None:
        report = PaperReport(
            signal_fidelity=SignalFidelityReport(5, 2, 0.4),
            cost=CostReport(1000, 500, 1500, 750),
            performance=PerformanceReport(
                0.3, 0.4, 0.20, 0.9, 0.05, 0.40, -0.02, 8, True,
            ),
            risk_compliance=RiskComplianceReport(0.20, 5, 3, 1),
            validation=ValidationResult(False, False, False, False, False),
            generated_at=datetime(2024, 6, 1),
            data_start=None,
            data_end=None,
        )
        msg = format_daily_digest_html(report)
        assert "NO-GO" in msg
        assert "need 10+ cycles" in msg
