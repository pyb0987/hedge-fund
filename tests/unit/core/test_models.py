"""Tests for core domain models."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime

import pytest

from hedgefund.core.enums import (
    AssetType,
    Exchange,
    OrderSide,
    OrderStatus,
    SignalDirection,
)
from hedgefund.core.models import Order, Position, PortfolioSnapshot, Signal, TradeRecord


class TestSignal:
    def test_create_signal(self, sample_signal: Signal) -> None:
        assert sample_signal.strategy_name == "crypto_momentum"
        assert sample_signal.symbol == "KRW-BTC"
        assert sample_signal.exchange == Exchange.UPBIT
        assert sample_signal.direction == SignalDirection.LONG
        assert sample_signal.strength == 0.85

    def test_signal_is_frozen(self, sample_signal: Signal) -> None:
        with pytest.raises(FrozenInstanceError):
            sample_signal.strength = 0.5  # type: ignore[misc]

    def test_signal_replace(self, sample_signal: Signal) -> None:
        new_signal = replace(sample_signal, strength=0.5)
        assert new_signal.strength == 0.5
        assert sample_signal.strength == 0.85  # original unchanged

    def test_signal_metadata(self) -> None:
        signal = Signal(
            strategy_name="test",
            symbol="TEST",
            exchange=Exchange.ALPACA,
            direction=SignalDirection.FLAT,
            strength=0.0,
            timestamp=datetime(2024, 1, 1),
        )
        assert signal.metadata is None


class TestPosition:
    def test_market_value(self, sample_position: Position) -> None:
        expected = 0.01 * 52_000_000
        assert sample_position.market_value == expected

    def test_unrealized_pnl(self, sample_position: Position) -> None:
        expected = 0.01 * (52_000_000 - 50_000_000)
        assert sample_position.unrealized_pnl == expected

    def test_unrealized_pnl_pct(self, sample_position: Position) -> None:
        expected = (52_000_000 - 50_000_000) / 50_000_000
        assert sample_position.unrealized_pnl_pct == pytest.approx(expected)

    def test_zero_entry_price_pnl_pct(self) -> None:
        pos = Position(
            symbol="TEST",
            exchange=Exchange.UPBIT,
            asset_type=AssetType.CRYPTO,
            quantity=1.0,
            avg_entry_price=0,
            current_price=100,
            strategy_name="test",
            opened_at=datetime(2024, 1, 1),
        )
        assert pos.unrealized_pnl_pct == 0.0

    def test_position_is_frozen(self, sample_position: Position) -> None:
        with pytest.raises(FrozenInstanceError):
            sample_position.quantity = 1.0  # type: ignore[misc]


class TestOrder:
    def test_default_status(self, sample_order: Order) -> None:
        assert sample_order.status == OrderStatus.PENDING

    def test_order_fields(self, sample_order: Order) -> None:
        assert sample_order.symbol == "KRW-BTC"
        assert sample_order.side == OrderSide.BUY
        assert sample_order.quantity == 0.01
        assert sample_order.limit_price is None
        assert sample_order.order_id is None


class TestTradeRecord:
    def test_create_trade(self) -> None:
        trade = TradeRecord(
            symbol="SPY",
            exchange=Exchange.ALPACA,
            side=OrderSide.BUY,
            quantity=10,
            price=450.0,
            commission=0.0,
            slippage=0.225,
            strategy_name="etf_mean_reversion",
            timestamp=datetime(2024, 6, 1),
        )
        assert trade.pnl is None
        assert trade.price == 450.0


class TestPortfolioSnapshot:
    def test_position_value(self) -> None:
        snapshot = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_value=1_000_000,
            cash=500_000,
            positions=(),
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            peak_value=1_000_000,
        )
        assert snapshot.position_value == 500_000

    def test_cash_ratio(self) -> None:
        snapshot = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_value=1_000_000,
            cash=300_000,
            positions=(),
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            peak_value=1_000_000,
        )
        assert snapshot.cash_ratio == pytest.approx(0.3)

    def test_cash_ratio_zero_value(self) -> None:
        snapshot = PortfolioSnapshot(
            timestamp=datetime(2024, 6, 1),
            total_value=0,
            cash=0,
            positions=(),
            unrealized_pnl=0,
            realized_pnl=0,
            drawdown=0.0,
            peak_value=0,
        )
        assert snapshot.cash_ratio == 1.0
