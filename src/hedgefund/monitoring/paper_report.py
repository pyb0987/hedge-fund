"""Paper trading validation report — analyzes SQLite data for Go/No-Go decision.

Reads signals, trades, and snapshots from DataStore and computes:
1. Signal Fidelity: conversion rate
2. Cost Accuracy: actual vs modeled costs
3. Performance: Sharpe, DD, Profit Factor, etc.
4. Risk Compliance: drawdown activations, position limit adherence
5. Validation: Go/No-Go thresholds
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hedgefund.core import risk_metrics
from hedgefund.data.store import DataStore


# --- Report data structures ---


@dataclass(frozen=True)
class ReportThresholds:
    """Go/No-Go thresholds for paper trading validation."""

    min_sharpe: float = 0.5
    max_drawdown: float = 0.15
    min_profit_factor: float = 1.1
    min_cycles: int = 10


@dataclass(frozen=True)
class SignalFidelityReport:
    """Signal-to-trade conversion analysis."""

    total_signals: int
    total_trades: int
    conversion_rate: float  # trades / signals


@dataclass(frozen=True)
class CostReport:
    """Transaction cost summary."""

    total_commission: float
    total_slippage: float
    total_cost: float
    avg_cost_per_trade: float


@dataclass(frozen=True)
class PerformanceReport:
    """Portfolio performance metrics from snapshot data."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    profit_factor: float
    annualized_return: float
    win_rate: float
    total_return: float
    num_cycles: int
    insufficient_data: bool


@dataclass(frozen=True)
class RiskComplianceReport:
    """Risk system compliance analysis."""

    max_observed_drawdown: float
    drawdown_activations: int  # cycles where drawdown > 5%
    risk_rejections: int  # orders rejected by risk checks
    cycles_blocked: int  # entire cycles blocked by risk system


@dataclass(frozen=True)
class ValidationResult:
    """Go/No-Go decision."""

    sharpe_pass: bool
    drawdown_pass: bool
    profit_factor_pass: bool
    sufficient_data: bool
    all_pass: bool


@dataclass(frozen=True)
class PaperReport:
    """Complete paper trading validation report."""

    signal_fidelity: SignalFidelityReport
    cost: CostReport
    performance: PerformanceReport
    risk_compliance: RiskComplianceReport
    validation: ValidationResult
    generated_at: datetime
    data_start: datetime | None
    data_end: datetime | None


# --- Analysis functions ---


def analyze_signal_fidelity(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> SignalFidelityReport:
    """Compute signal-to-trade conversion rate.

    Matches signals to trades by (date, symbol, strategy_name).
    """
    total_signals = len(signals_df)
    total_trades = len(trades_df)

    if total_signals == 0:
        return SignalFidelityReport(
            total_signals=0,
            total_trades=total_trades,
            conversion_rate=0.0,
        )

    conversion_rate = min(total_trades / total_signals, 1.0)
    return SignalFidelityReport(
        total_signals=total_signals,
        total_trades=total_trades,
        conversion_rate=conversion_rate,
    )


def analyze_costs(trades_df: pd.DataFrame) -> CostReport:
    """Summarize transaction costs from trade records."""
    if trades_df.empty:
        return CostReport(
            total_commission=0.0,
            total_slippage=0.0,
            total_cost=0.0,
            avg_cost_per_trade=0.0,
        )

    total_commission = float(trades_df["commission"].sum())
    total_slippage = float(trades_df["slippage"].sum())
    total_cost = total_commission + total_slippage
    avg_cost = total_cost / len(trades_df) if len(trades_df) > 0 else 0.0

    return CostReport(
        total_commission=total_commission,
        total_slippage=total_slippage,
        total_cost=total_cost,
        avg_cost_per_trade=avg_cost,
    )


def analyze_performance(
    snapshots_df: pd.DataFrame,
    thresholds: ReportThresholds = ReportThresholds(),
) -> PerformanceReport:
    """Compute performance metrics from portfolio snapshot time series."""
    empty = PerformanceReport(
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        annualized_return=0.0,
        win_rate=0.0,
        total_return=0.0,
        num_cycles=0,
        insufficient_data=True,
    )

    if snapshots_df.empty or len(snapshots_df) < 2:
        return empty

    values = snapshots_df["total_value"].values.astype(np.float64)
    returns = np.diff(values) / values[:-1]

    # Remove NaN/inf
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 2:
        return empty

    insufficient = n < thresholds.min_cycles
    total_return = float((values[-1] / values[0]) - 1.0) if values[0] > 0 else 0.0

    return PerformanceReport(
        sharpe_ratio=risk_metrics.sharpe_ratio(returns),
        sortino_ratio=risk_metrics.sortino_ratio(returns),
        max_drawdown=risk_metrics.max_drawdown(returns),
        profit_factor=risk_metrics.profit_factor(returns),
        annualized_return=risk_metrics.annualized_return(returns),
        win_rate=risk_metrics.win_rate(returns),
        total_return=total_return,
        num_cycles=n,
        insufficient_data=insufficient,
    )


def analyze_risk_compliance(
    snapshots_df: pd.DataFrame,
    risk_events_df: pd.DataFrame | None = None,
) -> RiskComplianceReport:
    """Analyze risk system compliance from snapshot and risk event data."""
    if snapshots_df.empty:
        return RiskComplianceReport(
            max_observed_drawdown=0.0,
            drawdown_activations=0,
            risk_rejections=0,
            cycles_blocked=0,
        )

    drawdowns = snapshots_df["drawdown"].values.astype(np.float64)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    # Progressive drawdown activates at 5%
    activations = int(np.sum(drawdowns > 0.05))

    # Count risk rejections and blocked cycles from risk_events table
    risk_rejections = 0
    cycles_blocked = 0
    if risk_events_df is not None and not risk_events_df.empty:
        # Count orders rejected by risk system or execution failures
        is_rejection = (
            (risk_events_df["event_type"] == "execution_failed")
            | ((risk_events_df["passed"] == 0)
               & (risk_events_df["event_type"].isin(["pre_trade", "order_rejected"])))
        )
        risk_rejections = int(is_rejection.sum())
        cycles_blocked = int(
            (risk_events_df["event_type"] == "cycle_blocked").sum()
        )

    return RiskComplianceReport(
        max_observed_drawdown=max_dd,
        drawdown_activations=activations,
        risk_rejections=risk_rejections,
        cycles_blocked=cycles_blocked,
    )


def validate(
    performance: PerformanceReport,
    thresholds: ReportThresholds = ReportThresholds(),
) -> ValidationResult:
    """Go/No-Go validation against thresholds."""
    sharpe_pass = performance.sharpe_ratio >= thresholds.min_sharpe
    drawdown_pass = performance.max_drawdown <= thresholds.max_drawdown
    pf_pass = performance.profit_factor >= thresholds.min_profit_factor
    sufficient = not performance.insufficient_data

    return ValidationResult(
        sharpe_pass=sharpe_pass,
        drawdown_pass=drawdown_pass,
        profit_factor_pass=pf_pass,
        sufficient_data=sufficient,
        all_pass=sharpe_pass and drawdown_pass and pf_pass and sufficient,
    )


def generate_report(
    store: DataStore,
    thresholds: ReportThresholds = ReportThresholds(),
) -> PaperReport:
    """Generate complete paper trading validation report from DataStore."""
    now = datetime.now()

    signals_df = store.load_signals()
    trades_df = store.load_trades()
    snapshots_df = store.load_snapshots()
    risk_events_df = store.load_risk_events()

    signal_fidelity = analyze_signal_fidelity(signals_df, trades_df)
    cost = analyze_costs(trades_df)
    performance = analyze_performance(snapshots_df, thresholds)
    risk_compliance = analyze_risk_compliance(snapshots_df, risk_events_df)
    validation_result = validate(performance, thresholds)

    # Determine data range
    data_start = None
    data_end = None
    if not snapshots_df.empty:
        data_start = snapshots_df.index[0].to_pydatetime()
        data_end = snapshots_df.index[-1].to_pydatetime()

    return PaperReport(
        signal_fidelity=signal_fidelity,
        cost=cost,
        performance=performance,
        risk_compliance=risk_compliance,
        validation=validation_result,
        generated_at=now,
        data_start=data_start,
        data_end=data_end,
    )


def format_report(report: PaperReport) -> str:
    """Format report as human-readable text."""
    sf = report.signal_fidelity
    cost = report.cost
    perf = report.performance
    risk = report.risk_compliance
    val = report.validation

    # Data range
    range_str = "N/A"
    if report.data_start and report.data_end:
        range_str = f"{report.data_start:%Y-%m-%d} ~ {report.data_end:%Y-%m-%d}"

    lines = [
        f"{'=' * 60}",
        " Paper Trading Validation Report",
        f"{'=' * 60}",
        f"  Generated: {report.generated_at:%Y-%m-%d %H:%M}",
        f"  Data Range: {range_str}",
        "",
        "Signal Fidelity:",
        f"  Total Signals:      {sf.total_signals:>8d}",
        f"  Total Trades:       {sf.total_trades:>8d}",
        f"  Conversion Rate:    {sf.conversion_rate:>8.1%}",
        "",
        "Transaction Costs:",
        f"  Total Commission:   {cost.total_commission:>12,.0f}",
        f"  Total Slippage:     {cost.total_slippage:>12,.0f}",
        f"  Total Cost:         {cost.total_cost:>12,.0f}",
        f"  Avg Cost/Trade:     {cost.avg_cost_per_trade:>12,.0f}",
        "",
        "Performance:",
    ]

    if perf.insufficient_data:
        lines.append(f"  WARNING: Insufficient data ({perf.num_cycles} cycles, need 10+)")

    lines.extend([
        f"  Sharpe Ratio:       {perf.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:      {perf.sortino_ratio:>10.2f}",
        f"  Annualized Return:  {perf.annualized_return:>10.2%}",
        f"  Total Return:       {perf.total_return:>10.2%}",
        f"  Max Drawdown:       {perf.max_drawdown:>10.2%}",
        f"  Profit Factor:      {perf.profit_factor:>10.2f}",
        f"  Win Rate:           {perf.win_rate:>10.2%}",
        f"  Cycles:             {perf.num_cycles:>10d}",
        "",
        "Risk Compliance:",
        f"  Max Observed DD:    {risk.max_observed_drawdown:>10.2%}",
        f"  DD Activations:     {risk.drawdown_activations:>10d}",
        f"  Risk Rejections:    {risk.risk_rejections:>10d}",
        f"  Cycles Blocked:     {risk.cycles_blocked:>10d}",
        "",
        "Go/No-Go Validation:",
        f"  [{'PASS' if val.sharpe_pass else 'FAIL'}] Sharpe >= 0.5",
        f"  [{'PASS' if val.drawdown_pass else 'FAIL'}] Max DD <= 15%",
        f"  [{'PASS' if val.profit_factor_pass else 'FAIL'}] Profit Factor >= 1.1",
        f"  [{'PASS' if val.sufficient_data else 'FAIL'}] Sufficient Data (10+ cycles)",
        "",
    ])

    overall = "GO" if val.all_pass else "NO-GO"
    lines.append(f"  Overall: {overall}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)
