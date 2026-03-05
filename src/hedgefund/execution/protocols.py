"""Executor protocol — interface for all order execution backends.

모든 거래소 실행기(Paper, Upbit, Alpaca)는 이 프로토콜을 구현해야 합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from hedgefund.core.enums import Exchange, OrderStatus
from hedgefund.core.models import Order


@dataclass(frozen=True)
class ExecutionResult:
    """Result of an order execution attempt."""

    order: Order
    success: bool
    filled_price: float | None = None
    filled_quantity: float | None = None
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime | None = None
    error_message: str | None = None

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage

    @property
    def fill_value(self) -> float:
        if self.filled_price is None or self.filled_quantity is None:
            return 0.0
        return self.filled_price * self.filled_quantity


@dataclass(frozen=True)
class AccountInfo:
    """Snapshot of exchange account state."""

    exchange: Exchange
    total_value: float
    cash_balance: float
    positions: dict[str, float] = field(default_factory=dict)  # symbol → quantity
    timestamp: datetime | None = None


@runtime_checkable
class Executor(Protocol):
    """Protocol for order execution backends."""

    @property
    def exchange(self) -> Exchange:
        """Which exchange this executor handles."""
        ...

    def submit_order(self, order: Order) -> ExecutionResult:
        """Submit an order for execution.

        Args:
            order: the order to execute

        Returns:
            ExecutionResult with fill details or error
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Returns:
            True if successfully cancelled
        """
        ...

    def get_account_info(self) -> AccountInfo:
        """Get current account state (balance + positions)."""
        ...

    def get_current_price(self, symbol: str) -> float:
        """Get latest price for a symbol.

        Returns:
            Current market price
        """
        ...
