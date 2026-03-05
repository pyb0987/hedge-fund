"""Exchange-specific transaction cost model.

거래소별 수수료, 슬리피지, 최소 주문 금액을 정의합니다.
"""

from dataclasses import dataclass

from hedgefund.core.enums import Exchange


@dataclass(frozen=True)
class ExchangeCosts:
    """Cost structure for a specific exchange."""

    commission_rate: float  # one-way commission (fraction)
    slippage_rate: float  # estimated slippage (fraction)
    min_order_value: float  # minimum order value in base currency

    @property
    def total_one_way(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def total_round_trip(self) -> float:
        return self.total_one_way * 2


# Exchange cost definitions
EXCHANGE_COSTS: dict[Exchange, ExchangeCosts] = {
    Exchange.UPBIT: ExchangeCosts(
        commission_rate=0.0025,  # 0.25% maker/taker
        slippage_rate=0.0015,  # 0.15% estimated
        min_order_value=5_000.0,  # 5,000 KRW
    ),
    Exchange.ALPACA: ExchangeCosts(
        commission_rate=0.0,  # commission-free
        slippage_rate=0.0005,  # 0.05% for liquid ETFs
        min_order_value=1.0,  # $1 (fractional shares)
    ),
}


def get_exchange_costs(exchange: Exchange) -> ExchangeCosts:
    """Get cost model for an exchange."""
    return EXCHANGE_COSTS[exchange]


def estimate_commission(trade_value: float, exchange: Exchange) -> float:
    """Estimate commission for a trade."""
    costs = get_exchange_costs(exchange)
    return abs(trade_value) * costs.commission_rate


def estimate_slippage(trade_value: float, exchange: Exchange) -> float:
    """Estimate slippage for a trade."""
    costs = get_exchange_costs(exchange)
    return abs(trade_value) * costs.slippage_rate


def estimate_total_cost(trade_value: float, exchange: Exchange) -> float:
    """Estimate total execution cost (commission + slippage)."""
    return estimate_commission(trade_value, exchange) + estimate_slippage(trade_value, exchange)


def is_above_minimum(trade_value: float, exchange: Exchange) -> bool:
    """Check if trade value meets minimum order requirement."""
    costs = get_exchange_costs(exchange)
    return abs(trade_value) >= costs.min_order_value
