"""Tests for order builder."""

from datetime import datetime

import pytest

from hedgefund.core.enums import Exchange, OrderSide, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.execution.order_builder import build_orders_from_signals, signal_to_order


@pytest.fixture
def long_signal() -> Signal:
    return Signal(
        strategy_name="test_strategy",
        symbol="KRW-BTC",
        exchange=Exchange.UPBIT,
        direction=SignalDirection.LONG,
        strength=1.0,
        timestamp=datetime(2024, 6, 1),
    )


@pytest.fixture
def flat_signal() -> Signal:
    return Signal(
        strategy_name="test_strategy",
        symbol="KRW-BTC",
        exchange=Exchange.UPBIT,
        direction=SignalDirection.FLAT,
        strength=0.0,
        timestamp=datetime(2024, 6, 1),
    )


class TestSignalToOrder:
    def test_long_signal_creates_buy(self, long_signal: Signal) -> None:
        order = signal_to_order(long_signal, current_price=50_000_000, target_value=100_000)
        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity == pytest.approx(100_000 / 50_000_000)

    def test_long_already_at_target(self, long_signal: Signal) -> None:
        """Already holding enough — no order needed."""
        order = signal_to_order(
            long_signal,
            current_price=50_000_000,
            target_value=100_000,
            current_position_qty=0.005,  # 250K value > 100K target
        )
        assert order is None

    def test_flat_signal_creates_sell(self, flat_signal: Signal) -> None:
        order = signal_to_order(
            flat_signal, current_price=50_000_000, target_value=0, current_position_qty=0.002,
        )
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == pytest.approx(0.002)

    def test_flat_no_position(self, flat_signal: Signal) -> None:
        """No position to sell — no order."""
        order = signal_to_order(flat_signal, current_price=50_000_000, target_value=0)
        assert order is None

    def test_zero_price_returns_none(self, long_signal: Signal) -> None:
        order = signal_to_order(long_signal, current_price=0, target_value=100_000)
        assert order is None

    def test_below_minimum_order_returns_none(self) -> None:
        """Order below exchange minimum should be skipped."""
        signal = Signal(
            strategy_name="test",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            direction=SignalDirection.LONG,
            strength=1.0,
            timestamp=datetime(2024, 6, 1),
        )
        # Target value 1000 KRW < 5000 KRW Upbit minimum
        order = signal_to_order(signal, current_price=50_000_000, target_value=1_000)
        assert order is None

    def test_short_signal_returns_none(self) -> None:
        """SHORT not supported for small capital (leverage=1.0)."""
        signal = Signal(
            strategy_name="test",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            direction=SignalDirection.SHORT,
            strength=1.0,
            timestamp=datetime(2024, 6, 1),
        )
        order = signal_to_order(signal, current_price=50_000_000, target_value=100_000)
        assert order is None


class TestBuildOrders:
    def test_multiple_signals(self) -> None:
        signals = [
            Signal("strat_a", "KRW-BTC", Exchange.UPBIT, SignalDirection.LONG, 1.0, datetime(2024, 6, 1)),
            Signal("strat_a", "KRW-ETH", Exchange.UPBIT, SignalDirection.FLAT, 0.0, datetime(2024, 6, 1)),
        ]
        prices = {"KRW-BTC": 50_000_000, "KRW-ETH": 3_000_000}
        target_values = {"KRW-BTC": 100_000, "KRW-ETH": 0}
        current_positions = {"KRW-ETH": 0.05}

        orders = build_orders_from_signals(signals, prices, target_values, current_positions)
        assert len(orders) == 2

        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(buy_orders) == 1
        assert len(sell_orders) == 1
        assert buy_orders[0].symbol == "KRW-BTC"
        assert sell_orders[0].symbol == "KRW-ETH"
