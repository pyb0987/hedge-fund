"""Backtest performance report generation."""

from hedgefund.backtest.engine import BacktestResult, validate_result


def generate_report(
    result: BacktestResult,
    strategy_name: str = "Strategy",
) -> str:
    """Generate a human-readable performance report.

    Args:
        result: backtest result to report on
        strategy_name: name for the report header

    Returns:
        Formatted string report
    """
    validation = validate_result(result)
    all_pass = all(validation.values())

    lines = [
        f"{'=' * 60}",
        f" Backtest Report: {strategy_name}",
        f"{'=' * 60}",
        "",
        "Performance Metrics:",
        f"  Total Return:         {result.total_return:>10.2%}",
        f"  Annualized Return:    {result.annualized_return:>10.2%}",
        f"  Annualized Volatility:{result.annualized_volatility:>10.2%}",
        f"  Sharpe Ratio:         {result.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:        {result.sortino_ratio:>10.2f}",
        f"  Max Drawdown:         {result.max_drawdown:>10.2%}",
        f"  Calmar Ratio:         {result.calmar_ratio:>10.2f}",
        f"  Profit Factor:        {result.profit_factor:>10.2f}",
        f"  Win Rate:             {result.win_rate:>10.2%}",
        "",
        "Trading Statistics:",
        f"  Number of Trades:     {result.num_trades:>10d}",
        f"  Avg Daily Turnover:   {result.turnover:>10.4f}",
        "",
        "Transaction Costs:",
        f"  Total Commission:     {result.total_commission:>10,.0f} KRW",
        f"  Total Slippage:       {result.total_slippage:>10,.0f} KRW",
        f"  Total Costs:          {result.total_costs:>10,.0f} KRW",
        "",
        "Go/No-Go Validation:",
    ]

    for criterion, passed in validation.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {criterion}")

    lines.append("")
    overall = "GO" if all_pass else "NO-GO"
    lines.append(f"  Overall: {overall}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)
