"""Domain enumerations for the hedge fund system."""

from enum import Enum, unique


@unique
class OrderSide(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


@unique
class SignalDirection(str, Enum):
    """Strategy signal direction."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@unique
class AssetType(str, Enum):
    """Asset classification."""

    CRYPTO = "crypto"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"


@unique
class Exchange(str, Enum):
    """Supported exchanges."""

    UPBIT = "upbit"
    ALPACA = "alpaca"


@unique
class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@unique
class RebalanceFrequency(str, Enum):
    """Rebalancing frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
