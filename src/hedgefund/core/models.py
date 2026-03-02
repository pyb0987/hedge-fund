"""Frozen domain models — immutable data structures for the hedge fund system.

All models are frozen dataclasses to enforce immutability.
Use replace() to create modified copies.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from hedgefund.core.enums import (
    AssetType,
    Exchange,
    OrderSide,
    OrderStatus,
    SignalDirection,
)

# Re-export replace for convenience
__all__ = ["Signal", "Position", "Order", "PortfolioSnapshot", "TradeRecord", "replace"]


@dataclass(frozen=True)
class Signal:
    """Strategy output signal.

    Represents a strategy's recommendation for a specific asset.
    """

    strategy_name: str
    symbol: str
    exchange: Exchange
    direction: SignalDirection
    strength: float  # 0.0 ~ 1.0, normalized signal strength
    timestamp: datetime
    metadata: dict[str, float] | None = None  # e.g. {"momentum_score": 0.85}


@dataclass(frozen=True)
class Position:
    """Current holding in a single asset."""

    symbol: str
    exchange: Exchange
    asset_type: AssetType
    quantity: float
    avg_entry_price: float
    current_price: float
    strategy_name: str
    opened_at: datetime

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass(frozen=True)
class Order:
    """Order to be submitted to an exchange."""

    symbol: str
    exchange: Exchange
    side: OrderSide
    quantity: float
    strategy_name: str
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = None
    filled_price: float | None = None
    filled_quantity: float | None = None
    order_id: str | None = None


@dataclass(frozen=True)
class TradeRecord:
    """Completed trade record for backtest and live tracking."""

    symbol: str
    exchange: Exchange
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    strategy_name: str
    timestamp: datetime
    pnl: float | None = None  # realized P&L (for closing trades)


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: datetime
    total_value: float
    cash: float
    positions: tuple[Position, ...]  # tuple for immutability
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float  # current drawdown from peak (0.0 ~ 1.0)
    peak_value: float

    @property
    def position_value(self) -> float:
        return self.total_value - self.cash

    @property
    def cash_ratio(self) -> float:
        if self.total_value == 0:
            return 1.0
        return self.cash / self.total_value
