"""Paper trading validation report — analyzes SQLite data for Go/No-Go decision.

Reads signals, trades, and snapshots from DataStore and computes:
1. Signal Fidelity: conversion rate
2. Cost Accuracy: actual vs modeled costs
3. Performance: Sharpe, DD, Profit Factor, etc.
4. Risk Compliance: drawdown activations, position limit adherence
5. Per-Strategy Attribution: performance breakdown per strategy
6. Rebalancing Gate: audit of hold vs rebalance decisions
7. Holding Period: distribution per strategy
8. Validation: Go/No-Go thresholds
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

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
class StrategyPerformanceReport:
    """Per-strategy performance from position snapshots."""

    strategy_name: str
    total_value: float  # latest market value
    total_pnl: float  # sum of unrealized PnL
    num_positions: int  # current position count
    avg_holding_days: float  # average holding period
    trade_count: int
    total_commission: float


@dataclass(frozen=True)
class RebalancingGateReport:
    """Rebalancing gate audit summary per strategy."""

    strategy_name: str
    total_decisions: int
    rebalance_count: int
    hold_count: int
    skip_count: int
    avg_dd_multiplier: float


@dataclass(frozen=True)
class HoldingPeriodReport:
    """Holding period distribution per strategy."""

    strategy_name: str
    avg_days: float
    min_days: int
    max_days: int
    num_positions_tracked: int


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
    strategy_performance: tuple[StrategyPerformanceReport, ...]
    rebalancing_gates: tuple[RebalancingGateReport, ...]
    holding_periods: tuple[HoldingPeriodReport, ...]
    validation: ValidationResult
    generated_at: datetime
    data_start: datetime | None
    data_end: datetime | None


# --- Analysis functions ---


def analyze_signal_fidelity(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> SignalFidelityReport:
    """Compute signal-to-trade conversion rate."""
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

    risk_rejections = 0
    cycles_blocked = 0
    if risk_events_df is not None and not risk_events_df.empty:
        is_rejection = (risk_events_df["event_type"] == "execution_failed") | (
            (risk_events_df["passed"] == 0)
            & (risk_events_df["event_type"].isin(["pre_trade", "order_rejected"]))
        )
        risk_rejections = int(is_rejection.sum())
        cycles_blocked = int((risk_events_df["event_type"] == "cycle_blocked").sum())

    return RiskComplianceReport(
        max_observed_drawdown=max_dd,
        drawdown_activations=activations,
        risk_rejections=risk_rejections,
        cycles_blocked=cycles_blocked,
    )


def analyze_strategy_performance(
    position_snapshots_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> tuple[StrategyPerformanceReport, ...]:
    """Compute per-strategy performance from position snapshots and trades."""
    if position_snapshots_df.empty:
        return ()

    reports: list[StrategyPerformanceReport] = []

    # Get latest snapshot per strategy
    latest_ts = position_snapshots_df.index.max()
    latest = position_snapshots_df.loc[position_snapshots_df.index == latest_ts]

    strategy_names = sorted(latest["strategy_name"].unique())

    for name in strategy_names:
        strat_positions = latest[latest["strategy_name"] == name]
        total_value = float(strat_positions["market_value"].sum())
        total_pnl = float(strat_positions["unrealized_pnl"].sum())
        num_positions = len(strat_positions)

        # Trade stats
        strat_trades = (
            trades_df[trades_df["strategy_name"] == name] if not trades_df.empty else pd.DataFrame()
        )
        trade_count = len(strat_trades)
        total_commission = (
            float(strat_trades["commission"].sum()) if not strat_trades.empty else 0.0
        )

        # Holding period from position snapshots (first → last appearance)
        strat_all = position_snapshots_df[position_snapshots_df["strategy_name"] == name]
        avg_holding = _compute_avg_holding_days(strat_all)

        reports.append(
            StrategyPerformanceReport(
                strategy_name=name,
                total_value=total_value,
                total_pnl=total_pnl,
                num_positions=num_positions,
                avg_holding_days=avg_holding,
                trade_count=trade_count,
                total_commission=total_commission,
            )
        )

    return tuple(reports)


def analyze_rebalancing_gates(
    decisions_df: pd.DataFrame,
) -> tuple[RebalancingGateReport, ...]:
    """Analyze rebalancing gate decisions per strategy."""
    if decisions_df.empty:
        return ()

    reports: list[RebalancingGateReport] = []

    for name in sorted(decisions_df["strategy_name"].unique()):
        strat = decisions_df[decisions_df["strategy_name"] == name]
        total = len(strat)
        rebalance = int((strat["action"] == "rebalance").sum())
        hold = int((strat["action"] == "hold").sum())
        skip = int((strat["action"] == "skip_no_data").sum())
        avg_dd = float(strat["dd_multiplier"].mean()) if total > 0 else 1.0

        reports.append(
            RebalancingGateReport(
                strategy_name=name,
                total_decisions=total,
                rebalance_count=rebalance,
                hold_count=hold,
                skip_count=skip,
                avg_dd_multiplier=avg_dd,
            )
        )

    return tuple(reports)


def analyze_holding_periods(
    position_snapshots_df: pd.DataFrame,
) -> tuple[HoldingPeriodReport, ...]:
    """Compute holding period distribution per strategy from position snapshots."""
    if position_snapshots_df.empty:
        return ()

    reports: list[HoldingPeriodReport] = []

    for name in sorted(position_snapshots_df["strategy_name"].unique()):
        strat = position_snapshots_df[position_snapshots_df["strategy_name"] == name]
        holding_days = _compute_holding_days_per_symbol(strat)

        if not holding_days:
            reports.append(
                HoldingPeriodReport(
                    strategy_name=name,
                    avg_days=0.0,
                    min_days=0,
                    max_days=0,
                    num_positions_tracked=0,
                )
            )
            continue

        reports.append(
            HoldingPeriodReport(
                strategy_name=name,
                avg_days=float(np.mean(holding_days)),
                min_days=int(min(holding_days)),
                max_days=int(max(holding_days)),
                num_positions_tracked=len(holding_days),
            )
        )

    return tuple(reports)


def _compute_holding_days_per_symbol(strat_df: pd.DataFrame) -> list[int]:
    """Compute holding days for each symbol from position snapshot appearances."""
    if strat_df.empty:
        return []

    holding_days: list[int] = []
    for symbol in strat_df["symbol"].unique():
        sym_data = strat_df[strat_df["symbol"] == symbol]
        dates = sorted(sym_data.index.unique())
        if len(dates) < 1:
            continue
        first = dates[0]
        last = dates[-1]
        days = (last - first).days
        # Minimum 1 day (single-snapshot positions)
        holding_days.append(max(1, days))

    return holding_days


def _compute_avg_holding_days(strat_df: pd.DataFrame) -> float:
    """Compute average holding days from position snapshots."""
    days = _compute_holding_days_per_symbol(strat_df)
    return float(np.mean(days)) if days else 0.0


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
    position_snapshots_df = store.load_position_snapshots()
    decisions_df = store.load_strategy_decisions()

    signal_fidelity = analyze_signal_fidelity(signals_df, trades_df)
    cost = analyze_costs(trades_df)
    performance = analyze_performance(snapshots_df, thresholds)
    risk_compliance = analyze_risk_compliance(snapshots_df, risk_events_df)
    strategy_perf = analyze_strategy_performance(position_snapshots_df, trades_df)
    rebalancing_gates = analyze_rebalancing_gates(decisions_df)
    holding_periods = analyze_holding_periods(position_snapshots_df)
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
        strategy_performance=strategy_perf,
        rebalancing_gates=rebalancing_gates,
        holding_periods=holding_periods,
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

    lines.extend(
        [
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
        ]
    )

    # Per-strategy performance
    if report.strategy_performance:
        lines.extend(["", "Per-Strategy Performance:"])
        for sp in report.strategy_performance:
            lines.extend(
                [
                    f"  [{sp.strategy_name}]",
                    f"    Value: {sp.total_value:>12,.0f}  "
                    f"PnL: {sp.total_pnl:>+10,.0f}  "
                    f"Positions: {sp.num_positions}  "
                    f"Trades: {sp.trade_count}",
                    f"    Avg Holding: {sp.avg_holding_days:.0f}d  "
                    f"Commission: {sp.total_commission:,.0f}",
                ]
            )

    # Rebalancing gate audit
    if report.rebalancing_gates:
        lines.extend(["", "Rebalancing Gate Audit:"])
        for rg in report.rebalancing_gates:
            lines.append(
                f"  [{rg.strategy_name}] "
                f"rebalance={rg.rebalance_count} hold={rg.hold_count} "
                f"skip={rg.skip_count} "
                f"avg_dd_mult={rg.avg_dd_multiplier:.2f}"
            )

    # Holding periods
    if report.holding_periods:
        lines.extend(["", "Holding Periods:"])
        for hp in report.holding_periods:
            if hp.num_positions_tracked > 0:
                lines.append(
                    f"  [{hp.strategy_name}] "
                    f"avg={hp.avg_days:.0f}d "
                    f"min={hp.min_days}d max={hp.max_days}d "
                    f"(n={hp.num_positions_tracked})"
                )

    lines.extend(
        [
            "",
            "Go/No-Go Validation:",
            f"  [{'PASS' if val.sharpe_pass else 'FAIL'}] Sharpe >= 0.5",
            f"  [{'PASS' if val.drawdown_pass else 'FAIL'}] Max DD <= 15%",
            f"  [{'PASS' if val.profit_factor_pass else 'FAIL'}] Profit Factor >= 1.1",
            f"  [{'PASS' if val.sufficient_data else 'FAIL'}] Sufficient Data (10+ cycles)",
            "",
        ]
    )

    overall = "GO" if val.all_pass else "NO-GO"
    lines.append(f"  Overall: {overall}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)
