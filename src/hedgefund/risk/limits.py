"""Hard risk limits — non-negotiable rules that override all strategy signals.

이 모듈의 규칙을 위반하면 주문이 거부되거나 포지션이 강제 감소됩니다.
"""

from dataclasses import dataclass

from hedgefund.config.schemas import RiskConfig
from hedgefund.core.exceptions import OrderRejectedError, PositionLimitError


@dataclass(frozen=True)
class LimitCheckResult:
    """Result of a risk limit check."""

    passed: bool
    rule_name: str
    current_value: float
    limit_value: float
    message: str


def check_single_position_limit(
    position_value: float,
    portfolio_value: float,
    config: RiskConfig,
) -> LimitCheckResult:
    """Check if a single position exceeds max allocation.

    Rule: No single position > 15% of portfolio.
    """
    if portfolio_value <= 0:
        return LimitCheckResult(
            passed=False, rule_name="single_position",
            current_value=0.0, limit_value=config.max_single_position,
            message="Portfolio value is zero or negative",
        )

    ratio = position_value / portfolio_value
    passed = ratio <= config.max_single_position
    return LimitCheckResult(
        passed=passed,
        rule_name="single_position",
        current_value=ratio,
        limit_value=config.max_single_position,
        message=f"Position {ratio:.1%} {'<=' if passed else '>'} {config.max_single_position:.1%} limit",
    )


def check_strategy_allocation_limit(
    strategy_value: float,
    portfolio_value: float,
    config: RiskConfig,
) -> LimitCheckResult:
    """Check if a strategy's total allocation exceeds max.

    Rule: No single strategy > 40% of portfolio.
    """
    if portfolio_value <= 0:
        return LimitCheckResult(
            passed=False, rule_name="strategy_allocation",
            current_value=0.0, limit_value=config.max_strategy_allocation,
            message="Portfolio value is zero or negative",
        )

    ratio = strategy_value / portfolio_value
    passed = ratio <= config.max_strategy_allocation
    return LimitCheckResult(
        passed=passed,
        rule_name="strategy_allocation",
        current_value=ratio,
        limit_value=config.max_strategy_allocation,
        message=f"Strategy {ratio:.1%} {'<=' if passed else '>'} {config.max_strategy_allocation:.1%} limit",
    )


def check_daily_var_limit(
    estimated_var: float,
    config: RiskConfig,
) -> LimitCheckResult:
    """Check if portfolio's daily VaR exceeds limit.

    Rule: 95% VaR < 2% of capital.
    """
    passed = estimated_var <= config.daily_var_95
    return LimitCheckResult(
        passed=passed,
        rule_name="daily_var",
        current_value=estimated_var,
        limit_value=config.daily_var_95,
        message=f"VaR {estimated_var:.2%} {'<=' if passed else '>'} {config.daily_var_95:.2%} limit",
    )


def check_stop_loss(
    trade_pnl_pct: float,
    config: RiskConfig,
) -> LimitCheckResult:
    """Check if a trade has hit stop-loss.

    Rule: 3% per-trade loss → mechanical liquidation.
    """
    loss = abs(min(0.0, trade_pnl_pct))
    passed = loss < config.per_trade_stop_loss
    return LimitCheckResult(
        passed=passed,
        rule_name="stop_loss",
        current_value=loss,
        limit_value=config.per_trade_stop_loss,
        message=f"Trade loss {loss:.2%} {'<' if passed else '>='} {config.per_trade_stop_loss:.2%} stop",
    )


def enforce_position_limit(
    position_value: float,
    portfolio_value: float,
    config: RiskConfig,
) -> None:
    """Raise if position limit violated. Use as pre-trade gate."""
    result = check_single_position_limit(position_value, portfolio_value, config)
    if not result.passed:
        raise PositionLimitError(result.message)


def enforce_strategy_limit(
    strategy_value: float,
    portfolio_value: float,
    config: RiskConfig,
) -> None:
    """Raise if strategy allocation limit violated."""
    result = check_strategy_allocation_limit(strategy_value, portfolio_value, config)
    if not result.passed:
        raise OrderRejectedError(result.message)
