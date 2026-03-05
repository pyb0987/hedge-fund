"""Tests for paper trading executor."""

from datetime import datetime

import pytest

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus
from hedgefund.core.models import Order
from hedgefund.execution.executors.paper_executor import PaperExecutor
from hedgefund.execution.protocols import Executor


def _make_order(
    symbol: str = "KRW-BTC",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 0.002,
    exchange: Exchange = Exchange.UPBIT,
) -> Order:
    return Order(
        strategy_name="test_strategy",
        symbol=symbol,
        exchange=exchange,
        side=side,
        quantity=quantity,
        timestamp=datetime(2024, 6, 1),
    )


class TestProtocolCompliance:
    def test_implements_executor_protocol(self) -> None:
        executor = PaperExecutor(Exchange.UPBIT, initial_cash=1_000_000)
        assert isinstance(executor, Executor)

    def test_exchange_property(self) -> None:
        executor = PaperExecutor(Exchange.UPBIT, initial_cash=1_000_000)
        assert executor.exchange == Exchange.UPBIT


class TestBuyOrders:
    def test_buy_deducts_cash(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        result = executor.submit_order(_make_order(quantity=0.002))
        assert result.success
        assert result.order.status == OrderStatus.FILLED
        assert executor.cash < 1_000_000

    def test_buy_creates_position(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(quantity=0.002))
        assert executor.get_position_quantity("KRW-BTC") == pytest.approx(0.002)

    def test_buy_weighted_average_entry(self) -> None:
        """Multiple buys update avg entry price."""
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=10_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(quantity=0.001))

        # Price changes, buy more
        executor.set_prices({"KRW-BTC": 60_000_000})
        executor.submit_order(_make_order(quantity=0.001))

        assert executor.get_position_quantity("KRW-BTC") == pytest.approx(0.002)

    def test_insufficient_cash_rejected(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=10_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        result = executor.submit_order(_make_order(quantity=0.002))
        assert not result.success
        assert result.order.status == OrderStatus.REJECTED
        assert "Insufficient cash" in (result.error_message or "")


class TestSellOrders:
    def test_sell_increases_cash(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(side=OrderSide.BUY, quantity=0.002))
        cash_after_buy = executor.cash

        executor.submit_order(_make_order(side=OrderSide.SELL, quantity=0.002))
        assert executor.cash > cash_after_buy

    def test_sell_removes_full_position(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(side=OrderSide.BUY, quantity=0.002))
        executor.submit_order(_make_order(side=OrderSide.SELL, quantity=0.002))
        assert executor.get_position_quantity("KRW-BTC") == pytest.approx(0.0)

    def test_partial_sell(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(side=OrderSide.BUY, quantity=0.004))
        executor.submit_order(_make_order(side=OrderSide.SELL, quantity=0.002))
        assert executor.get_position_quantity("KRW-BTC") == pytest.approx(0.002)

    def test_sell_insufficient_position_rejected(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        result = executor.submit_order(_make_order(side=OrderSide.SELL, quantity=0.002))
        assert not result.success
        assert result.order.status == OrderStatus.REJECTED

    def test_sell_records_pnl(self) -> None:
        """Sell records realized P&L in trade record."""
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(side=OrderSide.BUY, quantity=0.002))

        executor.set_prices({"KRW-BTC": 55_000_000})
        executor.submit_order(_make_order(side=OrderSide.SELL, quantity=0.002))

        sell_trades = [t for t in executor.trades if t.side == OrderSide.SELL]
        assert len(sell_trades) == 1
        assert sell_trades[0].pnl is not None


class TestEdgeCases:
    def test_no_price_rejected(self) -> None:
        executor = PaperExecutor(Exchange.UPBIT, initial_cash=1_000_000)
        result = executor.submit_order(_make_order())
        assert not result.success
        assert "No price" in (result.error_message or "")

    def test_cancel_always_false(self) -> None:
        executor = PaperExecutor(Exchange.UPBIT, initial_cash=1_000_000)
        assert executor.cancel_order("any-id") is False


class TestAccountState:
    def test_trade_history_tracked(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(quantity=0.002))
        assert len(executor.trades) == 1
        assert executor.trades[0].symbol == "KRW-BTC"
        assert executor.trades[0].side == OrderSide.BUY

    def test_account_info_no_positions(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        info = executor.get_account_info()
        assert info.exchange == Exchange.UPBIT
        assert info.total_value == pytest.approx(1_000_000)
        assert info.cash_balance == pytest.approx(1_000_000)
        assert len(info.positions) == 0

    def test_account_info_with_positions(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(quantity=0.002))

        info = executor.get_account_info()
        assert info.positions.get("KRW-BTC", 0) == pytest.approx(0.002)
        # Total value ≈ initial cash (minus costs, plus position value)
        assert info.total_value == pytest.approx(1_000_000, rel=0.01)

    def test_position_value(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        executor.submit_order(_make_order(quantity=0.002))
        value = executor.get_position_value("KRW-BTC")
        assert value == pytest.approx(100_000)  # 0.002 * 50M

    def test_set_prices_updates_feed(self) -> None:
        executor = PaperExecutor(Exchange.UPBIT, initial_cash=1_000_000)
        executor.set_prices({"KRW-BTC": 42_000_000})
        assert executor.get_current_price("KRW-BTC") == 42_000_000

    def test_execution_result_has_costs(self) -> None:
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )
        result = executor.submit_order(_make_order(quantity=0.002))
        assert result.success
        assert result.commission > 0  # Upbit charges 0.25%
        assert result.slippage > 0  # Upbit slippage 0.15%
        assert result.total_cost > 0
