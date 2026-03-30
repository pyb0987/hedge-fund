"""Paper trading executor — simulates trades in memory.

실제 거래 없이 전략을 검증합니다. 비용 모델을 적용하고
포지션/현금/거래 이력을 추적합니다.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus
from hedgefund.core.models import Order, TradeRecord
from hedgefund.execution.cost_model import estimate_commission, estimate_slippage
from hedgefund.execution.protocols import AccountInfo, ExecutionResult


@dataclass(frozen=True)
class PaperPosition:
    """Immutable paper trading position."""

    symbol: str
    quantity: float
    avg_entry_price: float
    exchange: Exchange
    strategy_name: str


class PaperExecutor:
    """Simulated executor for paper trading.

    Tracks positions, cash balance, and trade history in memory.
    Applies realistic cost model (commissions + slippage).
    """

    def __init__(
        self,
        exchange: Exchange,
        initial_cash: float,
        price_feed: dict[str, float] | None = None,
    ) -> None:
        self._exchange = exchange
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[tuple[str, str], PaperPosition] = {}
        self._trades: list[TradeRecord] = []
        self._price_feed: dict[str, float] = price_feed or {}

    @property
    def exchange(self) -> Exchange:
        return self._exchange

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[tuple[str, str], "PaperPosition"]:
        """Return defensive copy of current positions.

        Keys are (strategy_name, symbol) tuples.
        """
        return dict(self._positions)

    @property
    def trades(self) -> list[TradeRecord]:
        return list(self._trades)

    def set_prices(self, prices: dict[str, float]) -> None:
        """Update simulated price feed (merge, not replace).

        Preserves stale prices for symbols not in the update so that
        positions still have a non-zero valuation when their exchange
        is skipped (e.g. Alpaca on weekends).
        """
        self._price_feed.update(prices)

    def get_current_price(self, symbol: str) -> float:
        """Get simulated price."""
        price = self._price_feed.get(symbol)
        if price is None:
            return 0.0
        return price

    def submit_order(self, order: Order) -> ExecutionResult:
        """Simulate order execution with realistic costs."""
        now = datetime.now()
        price = self.get_current_price(order.symbol)

        if price <= 0:
            return ExecutionResult(
                order=replace(order, status=OrderStatus.REJECTED),
                success=False,
                timestamp=now,
                error_message=f"No price available for {order.symbol}",
            )

        trade_value = price * order.quantity
        commission = estimate_commission(trade_value, self._exchange)
        slippage = estimate_slippage(trade_value, self._exchange)

        # Apply slippage to fill price
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + slippage / trade_value) if trade_value > 0 else price
        else:
            fill_price = price * (1 - slippage / trade_value) if trade_value > 0 else price

        fill_value = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            total_cost = fill_value + commission
            if total_cost > self._cash:
                return ExecutionResult(
                    order=replace(order, status=OrderStatus.REJECTED),
                    success=False,
                    timestamp=now,
                    error_message=f"Insufficient cash: need {total_cost:.0f}, have {self._cash:.0f}",
                )

            self._cash -= total_cost
            self._update_position_buy(order.strategy_name, order.symbol, order.quantity, fill_price)

        elif order.side == OrderSide.SELL:
            pos_key = (order.strategy_name, order.symbol)
            current_pos = self._positions.get(pos_key)
            if current_pos is None or current_pos.quantity < order.quantity:
                return ExecutionResult(
                    order=replace(order, status=OrderStatus.REJECTED),
                    success=False,
                    timestamp=now,
                    error_message=f"Insufficient position for {order.symbol}",
                )

            # Capture entry price BEFORE sell removes the position
            sell_entry_price = current_pos.avg_entry_price

            proceeds = fill_value - commission
            self._cash += proceeds
            self._update_position_sell(order.strategy_name, order.symbol, order.quantity)

        # Record trade
        filled_order = replace(
            order,
            status=OrderStatus.FILLED,
            filled_price=fill_price,
            filled_quantity=order.quantity,
        )

        entry_price = 0.0
        if order.side == OrderSide.SELL:
            entry_price = sell_entry_price  # type: ignore[possibly-undefined]
        else:
            pos = self._positions.get((order.strategy_name, order.symbol))
            if pos is not None:
                entry_price = pos.avg_entry_price

        pnl = None
        if order.side == OrderSide.SELL and entry_price > 0:
            pnl = (fill_price - entry_price) * order.quantity - commission - slippage

        trade = TradeRecord(
            strategy_name=order.strategy_name,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=now,
            pnl=pnl,
        )
        self._trades.append(trade)

        return ExecutionResult(
            order=filled_order,
            success=True,
            filled_price=fill_price,
            filled_quantity=order.quantity,
            commission=commission,
            slippage=slippage,
            timestamp=now,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Paper executor fills immediately — nothing to cancel."""
        return False

    def get_account_info(self) -> AccountInfo:
        """Get current paper account state."""
        # Aggregate quantities by symbol across all strategies
        positions_qty: dict[str, float] = {}
        for (_strat, sym), pos in self._positions.items():
            positions_qty[sym] = positions_qty.get(sym, 0.0) + pos.quantity

        # Compute total value
        position_value = sum(
            pos.quantity * self.get_current_price(pos.symbol) for pos in self._positions.values()
        )

        return AccountInfo(
            exchange=self._exchange,
            total_value=self._cash + position_value,
            cash_balance=self._cash,
            positions=positions_qty,
            timestamp=datetime.now(),
        )

    def get_position_quantity(self, symbol: str) -> float:
        """Get aggregate quantity for a symbol across all strategies."""
        total = 0.0
        for (_strat, sym), pos in self._positions.items():
            if sym == symbol:
                total += pos.quantity
        return total

    def get_position_value(self, symbol: str) -> float:
        """Get current market value of a position."""
        qty = self.get_position_quantity(symbol)
        price = self.get_current_price(symbol)
        return qty * price

    def get_strategy_position_quantity(self, strategy_name: str, symbol: str) -> float:
        """Get quantity for a specific strategy's position in a symbol."""
        pos = self._positions.get((strategy_name, symbol))
        return pos.quantity if pos is not None else 0.0

    def _update_position_buy(
        self, strategy_name: str, symbol: str, quantity: float, price: float
    ) -> None:
        """Update position after a buy (weighted average entry price)."""
        key = (strategy_name, symbol)
        existing = self._positions.get(key)
        if existing is not None:
            total_qty = existing.quantity + quantity
            avg_price = (
                existing.avg_entry_price * existing.quantity + price * quantity
            ) / total_qty
            self._positions[key] = PaperPosition(
                symbol=symbol,
                quantity=total_qty,
                avg_entry_price=avg_price,
                exchange=self._exchange,
                strategy_name=strategy_name,
            )
        else:
            self._positions[key] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                exchange=self._exchange,
                strategy_name=strategy_name,
            )

    def _update_position_sell(self, strategy_name: str, symbol: str, quantity: float) -> None:
        """Update position after a sell."""
        key = (strategy_name, symbol)
        existing = self._positions.get(key)
        if existing is None:
            return

        remaining = existing.quantity - quantity
        if remaining <= 1e-10:
            del self._positions[key]
        else:
            self._positions[key] = PaperPosition(
                symbol=symbol,
                quantity=remaining,
                avg_entry_price=existing.avg_entry_price,
                exchange=self._exchange,
                strategy_name=strategy_name,
            )
