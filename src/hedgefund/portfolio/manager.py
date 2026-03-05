"""Portfolio Manager — the system's heart.

run_cycle 오케스트레이터:
1. 데이터 수집
2. 시그널 생성
3. 리스크 체크
4. 주문 생성
5. 주문 실행
6. 포스트-트레이드 체크
7. 결과 기록
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from hedgefund.config.schemas import AllocationConfig, GlobalSettings, RiskConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.execution.order_builder import build_orders_from_signals
from hedgefund.execution.protocols import AccountInfo, ExecutionResult, Executor
from hedgefund.risk.manager import RiskManager
from hedgefund.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleResult:
    """Result of a single trading cycle."""

    timestamp: datetime
    signals: tuple[Signal, ...]
    executions: tuple[ExecutionResult, ...]
    portfolio_value: float
    cash_balance: float
    risk_blocked: bool
    risk_reason: str | None = None

    @property
    def num_trades(self) -> int:
        return sum(1 for e in self.executions if e.success)

    @property
    def total_commission(self) -> float:
        return sum(e.commission for e in self.executions if e.success)


class PortfolioManager:
    """Orchestrates the complete trading cycle.

    Coordinates strategies, executors, risk management, and position sizing.
    """

    def __init__(
        self,
        strategies: dict[str, Strategy],
        executors: dict[Exchange, Executor],
        risk_manager: RiskManager,
        allocation: AllocationConfig,
        risk_config: RiskConfig = RiskConfig(),
    ) -> None:
        self._strategies = strategies
        self._executors = executors
        self._risk_manager = risk_manager
        self._allocation = allocation
        self._risk_config = risk_config
        self._cycle_history: list[CycleResult] = []
        self._portfolio_returns: list[float] = []

    @property
    def cycle_history(self) -> list[CycleResult]:
        return list(self._cycle_history)

    def run_cycle(
        self,
        data: dict[str, dict[str, pd.DataFrame]],
        timestamp: datetime | None = None,
    ) -> CycleResult:
        """Execute one complete trading cycle.

        Args:
            data: strategy_name → {symbol → OHLCV DataFrame}
            timestamp: current time (defaults to now)

        Returns:
            CycleResult with all execution details
        """
        now = timestamp or datetime.now()
        all_signals: list[Signal] = []
        all_executions: list[ExecutionResult] = []

        # Step 1: Get current portfolio state
        total_value = self._get_total_portfolio_value()
        cash_balance = self._get_total_cash()

        # Step 2: Pre-trade risk check (portfolio level)
        portfolio_returns = np.array(self._portfolio_returns[-60:], dtype=np.float64)
        if len(portfolio_returns) < 5:
            portfolio_returns = np.zeros(20)

        dd_state = self._risk_manager.get_drawdown_state(total_value)
        if dd_state.is_max_breached:
            logger.warning("Max drawdown breached: %.1f%% — blocking all trades", dd_state.drawdown * 100)
            result = CycleResult(
                timestamp=now,
                signals=(),
                executions=(),
                portfolio_value=total_value,
                cash_balance=cash_balance,
                risk_blocked=True,
                risk_reason=f"Max drawdown breached: {dd_state.drawdown:.1%}",
            )
            self._cycle_history.append(result)
            return result

        dd_multiplier = dd_state.position_multiplier

        # Step 3: Generate signals from each strategy
        for strategy_name, strategy in self._strategies.items():
            strategy_data = data.get(strategy_name, {})
            if not strategy_data:
                logger.warning("No data for strategy %s", strategy_name)
                continue

            try:
                signals = strategy.generate_signals(strategy_data, now)
                all_signals.extend(signals)
            except Exception:
                logger.exception("Signal generation failed for %s", strategy_name)

        if not all_signals:
            result = CycleResult(
                timestamp=now,
                signals=(),
                executions=(),
                portfolio_value=total_value,
                cash_balance=cash_balance,
                risk_blocked=False,
            )
            self._cycle_history.append(result)
            return result

        # Step 4: Compute target values per symbol
        allocation_weights = self._get_allocation_weights()
        prices = self._get_current_prices(all_signals)
        current_positions = self._get_current_positions()

        target_values: dict[str, float] = {}
        for signal in all_signals:
            if signal.direction == SignalDirection.LONG:
                strategy_allocation = allocation_weights.get(signal.strategy_name, 0.0)
                # Target = portfolio_value × strategy_allocation × signal_strength × DD_multiplier
                target = total_value * strategy_allocation * signal.strength * dd_multiplier
                target_values[signal.symbol] = target
            else:
                target_values[signal.symbol] = 0.0  # FLAT = close

        # Step 5: Build orders
        orders = build_orders_from_signals(
            all_signals, prices, target_values, current_positions,
        )

        # Step 6: Pre-trade risk check per order
        for order in orders:
            strategy_total = self._get_strategy_total_value(order.strategy_name)
            report = self._risk_manager.pre_trade_check(
                order_value=order.quantity * (order.price or 0),
                strategy_total_value=strategy_total,
                portfolio_value=total_value,
                portfolio_returns=portfolio_returns,
                timestamp=now,
            )

            if not report.all_passed:
                logger.warning(
                    "Risk check failed for %s %s: %s",
                    order.symbol, order.side.value, report.blocked_reason,
                )
                # Record rejected execution for audit trail
                all_executions.append(ExecutionResult(
                    order=order,
                    success=False,
                    error_message=f"Risk rejected: {report.blocked_reason}",
                ))
                continue

            # Step 7: Execute
            executor = self._executors.get(order.exchange)
            if executor is None:
                logger.error("No executor for exchange %s", order.exchange.value)
                continue

            try:
                exec_result = executor.submit_order(order)
                all_executions.append(exec_result)

                if exec_result.success:
                    logger.info(
                        "Executed: %s %s %.4f @ %.2f (cost: %.2f)",
                        order.side.value, order.symbol,
                        exec_result.filled_quantity or 0,
                        exec_result.filled_price or 0,
                        exec_result.total_cost,
                    )
            except Exception:
                logger.exception("Execution failed for %s", order.symbol)

        # Step 8: Update state
        new_total_value = self._get_total_portfolio_value()
        self._risk_manager.update_peak(new_total_value)

        if total_value > 0:
            period_return = (new_total_value - total_value) / total_value
            self._portfolio_returns.append(period_return)

        result = CycleResult(
            timestamp=now,
            signals=tuple(all_signals),
            executions=tuple(all_executions),
            portfolio_value=new_total_value,
            cash_balance=self._get_total_cash(),
            risk_blocked=False,
        )
        self._cycle_history.append(result)
        return result

    def _get_allocation_weights(self) -> dict[str, float]:
        """Get target allocation weights from config."""
        return {
            "crypto_momentum": self._allocation.crypto_momentum,
            "etf_mean_reversion": self._allocation.etf_mean_reversion,
            "dual_momentum": self._allocation.dual_momentum,
        }

    def _get_total_portfolio_value(self) -> float:
        """Sum total value across all executors."""
        total = 0.0
        for executor in self._executors.values():
            try:
                info = executor.get_account_info()
                total += info.total_value
            except Exception:
                logger.exception("Failed to get account info for %s", executor.exchange)
        return total

    def _get_total_cash(self) -> float:
        """Sum cash across all executors."""
        total = 0.0
        for executor in self._executors.values():
            try:
                info = executor.get_account_info()
                total += info.cash_balance
            except Exception:
                pass
        return total

    def _get_current_prices(self, signals: list[Signal]) -> dict[str, float]:
        """Get current prices for all signal symbols."""
        prices: dict[str, float] = {}
        for signal in signals:
            if signal.symbol in prices:
                continue
            executor = self._executors.get(signal.exchange)
            if executor is not None:
                prices[signal.symbol] = executor.get_current_price(signal.symbol)
        return prices

    def _get_current_positions(self) -> dict[str, float]:
        """Get current positions across all executors."""
        positions: dict[str, float] = {}
        for executor in self._executors.values():
            try:
                info = executor.get_account_info()
                positions.update(info.positions)
            except Exception:
                pass
        return positions

    def _get_strategy_total_value(self, strategy_name: str) -> float:
        """Get total value for a specific strategy's positions."""
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            return 0.0

        total = 0.0
        universe = strategy.get_universe()
        executor = self._executors.get(strategy.exchange)
        if executor is None:
            return 0.0

        info = executor.get_account_info()
        for symbol in universe:
            qty = info.positions.get(symbol, 0.0)
            if qty > 0:
                price = executor.get_current_price(symbol)
                total += qty * price

        return total
