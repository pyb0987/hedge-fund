"""Risk Manager — pre-trade and post-trade risk orchestration.

모든 주문은 Risk Manager를 통과해야 합니다:
1. Pre-trade: 주문 실행 전 리밋 체크
2. Post-trade: 실행 후 포트폴리오 상태 검증
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from hedgefund.config.schemas import RiskConfig
from hedgefund.core.risk_metrics import value_at_risk
from hedgefund.risk.drawdown import DrawdownState, get_drawdown_state
from hedgefund.risk.limits import (
    LimitCheckResult,
    check_daily_var_limit,
    check_single_position_limit,
    check_stop_loss,
    check_strategy_allocation_limit,
    check_symbol_aggregate_exposure,
)


@dataclass(frozen=True)
class RiskCheckReport:
    """Aggregated risk check result."""

    timestamp: datetime
    checks: tuple[LimitCheckResult, ...]
    drawdown_state: DrawdownState
    all_passed: bool
    blocked_reason: str | None

    @property
    def failed_checks(self) -> tuple[LimitCheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)


class RiskManager:
    """Stateful risk manager that tracks portfolio risk over time."""

    def __init__(self, config: RiskConfig, initial_capital: float) -> None:
        self._config = config
        self._peak_value = initial_capital
        self._initial_capital = initial_capital

    @property
    def config(self) -> RiskConfig:
        return self._config

    def pre_trade_check(
        self,
        order_value: float,
        strategy_total_value: float,
        portfolio_value: float,
        portfolio_returns: NDArray[np.float64],
        timestamp: datetime,
        symbol: str = "",
        symbol_total_value: float = 0.0,
    ) -> RiskCheckReport:
        """Pre-trade risk check — must pass before order submission.

        Args:
            order_value: value of the proposed order
            strategy_total_value: total value of all positions in this strategy
            portfolio_value: current total portfolio value
            portfolio_returns: recent portfolio returns for VaR
            timestamp: current time
        """
        checks: list[LimitCheckResult] = []

        # 1. Single position limit
        checks.append(
            check_single_position_limit(
                order_value,
                portfolio_value,
                self._config,
            )
        )

        # 2. Strategy allocation limit (after order)
        new_strategy_value = strategy_total_value + order_value
        checks.append(
            check_strategy_allocation_limit(
                new_strategy_value,
                portfolio_value,
                self._config,
            )
        )

        # 3. Cross-strategy symbol aggregate exposure
        if symbol:
            checks.append(
                check_symbol_aggregate_exposure(
                    symbol,
                    symbol_total_value + order_value,
                    portfolio_value,
                    self._config,
                )
            )

        # 4. Daily VaR
        if len(portfolio_returns) >= 20:
            var_95 = value_at_risk(portfolio_returns, 0.95)
            checks.append(check_daily_var_limit(var_95, self._config))

        # 4. Drawdown state
        dd_state = get_drawdown_state(
            portfolio_value,
            self._peak_value,
            self._config.max_portfolio_drawdown,
        )

        # Update peak
        self._peak_value = dd_state.peak_value

        # Determine if blocked
        all_passed = all(c.passed for c in checks) and not dd_state.is_max_breached
        blocked_reason = None
        if dd_state.is_max_breached:
            blocked_reason = (
                f"Max drawdown breached: {dd_state.drawdown:.1%} "
                f"> {self._config.max_portfolio_drawdown:.1%}"
            )
        elif not all(c.passed for c in checks):
            failed = [c for c in checks if not c.passed]
            blocked_reason = "; ".join(c.message for c in failed)

        return RiskCheckReport(
            timestamp=timestamp,
            checks=tuple(checks),
            drawdown_state=dd_state,
            all_passed=all_passed,
            blocked_reason=blocked_reason,
        )

    def post_trade_check(
        self,
        positions: list[dict],
        portfolio_value: float,
        timestamp: datetime,
    ) -> list[LimitCheckResult]:
        """Post-trade verification — check all positions are within limits.

        Args:
            positions: list of {"value": float, "strategy": str, "pnl_pct": float}
            portfolio_value: current total portfolio value
            timestamp: current time

        Returns:
            List of limit check results (including stop-loss checks)
        """
        results: list[LimitCheckResult] = []

        for pos in positions:
            # Position size check
            results.append(
                check_single_position_limit(
                    pos["value"],
                    portfolio_value,
                    self._config,
                )
            )

            # Stop-loss check
            results.append(check_stop_loss(pos["pnl_pct"], self._config))

        return results

    def get_drawdown_state(self, portfolio_value: float) -> DrawdownState:
        """Get current drawdown state without modifying peak."""
        return get_drawdown_state(
            portfolio_value,
            self._peak_value,
            self._config.max_portfolio_drawdown,
        )

    def update_peak(self, portfolio_value: float) -> None:
        """Explicitly update peak value."""
        self._peak_value = max(portfolio_value, self._peak_value)
