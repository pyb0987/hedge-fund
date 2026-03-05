"""Pure formatting functions for Telegram notifications.

All functions take structured data and return HTML-formatted strings.
No side effects, no external dependencies.
"""

from datetime import datetime

from hedgefund.monitoring.paper_report import PaperReport


def format_cycle_html(
    timestamp: datetime,
    signals: int,
    trades: int,
    commission: float,
    portfolio_value: float,
    risk_blocked: bool,
    risk_reason: str | None,
    errors: tuple[str, ...],
) -> str:
    """Format cycle summary as Telegram HTML message."""
    status = "BLOCKED" if risk_blocked else "OK"
    lines = [
        f"<b>Paper Trading Cycle</b>",
        f"{timestamp:%Y-%m-%d %H:%M}",
        "",
        f"Status: {status}",
        f"Signals: {signals}",
        f"Trades: {trades}",
        f"Commission: {commission:,.0f}",
        f"Portfolio: {portfolio_value:,.0f}",
    ]

    if risk_blocked and risk_reason:
        lines.append(f"\n<b>Risk Alert:</b> {_escape_html(risk_reason)}")

    if errors:
        lines.append(f"\nWarnings ({len(errors)}):")
        for err in errors[:5]:
            lines.append(f"  - {_escape_html(err)}")

    return "\n".join(lines)


def format_risk_alert_html(
    reason: str,
    drawdown: float,
    timestamp: datetime,
) -> str:
    """Format risk alert as high-priority Telegram HTML message."""
    return (
        f"<b>RISK ALERT</b>\n"
        f"{timestamp:%Y-%m-%d %H:%M}\n"
        f"\n"
        f"Reason: {_escape_html(reason)}\n"
        f"Current Drawdown: {drawdown:.1%}"
    )


def format_daily_digest_html(report: PaperReport) -> str:
    """Format daily performance digest as Telegram HTML message."""
    perf = report.performance
    val = report.validation
    sf = report.signal_fidelity

    overall = "GO" if val.all_pass else "NO-GO"
    check = lambda passed: "+" if passed else "X"  # noqa: E731

    lines = [
        f"<b>Daily Digest</b>",
        f"{report.generated_at:%Y-%m-%d}",
        "",
        f"Sharpe: {perf.sharpe_ratio:.2f} [{check(val.sharpe_pass)}]",
        f"Max DD: {perf.max_drawdown:.1%} [{check(val.drawdown_pass)}]",
        f"PF: {perf.profit_factor:.2f} [{check(val.profit_factor_pass)}]",
        f"Return: {perf.total_return:.2%}",
        f"Cycles: {perf.num_cycles}",
        "",
        f"Signals: {sf.total_signals} | Trades: {sf.total_trades}",
        f"Conversion: {sf.conversion_rate:.0%}",
        "",
        f"<b>Overall: {overall}</b>",
    ]

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
