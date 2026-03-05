"""Order builder — converts Signals into executable Orders.

Signal → 포지션 사이징 → 리스크 체크 → Order 생성.
"""

from datetime import datetime

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus, SignalDirection
from hedgefund.core.models import Order, Signal
from hedgefund.execution.cost_model import is_above_minimum


def signal_to_order(
    signal: Signal,
    current_price: float,
    target_value: float,
    current_position_qty: float = 0.0,
) -> Order | None:
    """Convert a Signal into an Order.

    Args:
        signal: trading signal from a strategy
        current_price: latest market price for the symbol
        target_value: target position value in base currency (from position sizing)
        current_position_qty: current quantity held (for sell/rebalance)

    Returns:
        Order if action needed, None if no trade required
    """
    if current_price <= 0:
        return None

    if signal.direction == SignalDirection.LONG:
        # Compute how much to buy
        target_qty = target_value / current_price
        delta_qty = target_qty - current_position_qty

        if delta_qty <= 0:
            return None  # already at or above target

        order_value = delta_qty * current_price
        if not is_above_minimum(order_value, signal.exchange):
            return None  # below minimum order

        return Order(
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=OrderSide.BUY,
            quantity=delta_qty,
            price=current_price,
            status=OrderStatus.PENDING,
            timestamp=signal.timestamp,
        )

    elif signal.direction == SignalDirection.FLAT:
        # Close entire position
        if current_position_qty <= 0:
            return None  # nothing to sell

        sell_value = current_position_qty * current_price
        if not is_above_minimum(sell_value, signal.exchange):
            return None  # below minimum order

        return Order(
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=OrderSide.SELL,
            quantity=current_position_qty,
            price=current_price,
            status=OrderStatus.PENDING,
            timestamp=signal.timestamp,
        )

    elif signal.direction == SignalDirection.SHORT:
        # Short selling via Alpaca margin account ($2K+ required)
        if signal.exchange != Exchange.ALPACA:
            return None  # Upbit does not support short selling

        target_qty = target_value / current_price
        if current_position_qty >= 0:
            # Open new short or add to no position
            short_qty = target_qty
        else:
            # Already short — adjust
            delta_qty = target_qty - abs(current_position_qty)
            if delta_qty <= 0:
                return None  # already at or beyond target short
            short_qty = delta_qty

        order_value = short_qty * current_price
        if not is_above_minimum(order_value, signal.exchange):
            return None

        return Order(
            strategy_name=signal.strategy_name,
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=OrderSide.SELL,
            quantity=short_qty,
            price=current_price,
            status=OrderStatus.PENDING,
            timestamp=signal.timestamp,
        )

    return None


def build_orders_from_signals(
    signals: list[Signal],
    prices: dict[str, float],
    target_values: dict[str, float],
    current_positions: dict[str, float],
) -> list[Order]:
    """Build orders for multiple signals.

    Args:
        signals: list of trading signals
        prices: symbol → current price
        target_values: symbol → target position value
        current_positions: symbol → current quantity held

    Returns:
        List of executable orders
    """
    orders: list[Order] = []

    for signal in signals:
        price = prices.get(signal.symbol, 0.0)
        target_val = target_values.get(signal.symbol, 0.0)
        current_qty = current_positions.get(signal.symbol, 0.0)

        order = signal_to_order(signal, price, target_val, current_qty)
        if order is not None:
            orders.append(order)

    return orders
