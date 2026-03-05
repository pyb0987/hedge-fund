"""Upbit exchange executor — real trading via pyupbit API.

Upbit KRW 마켓에서 암호화폐 현물 거래를 실행합니다.
API 키가 필요하며, 실거래 전 반드시 페이퍼 트레이딩으로 검증해야 합니다.
"""

import logging
from dataclasses import replace
from datetime import datetime

import pyupbit

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus
from hedgefund.core.exceptions import ExecutionError
from hedgefund.core.models import Order
from hedgefund.execution.cost_model import estimate_commission, estimate_slippage
from hedgefund.execution.protocols import AccountInfo, ExecutionResult

logger = logging.getLogger(__name__)


class UpbitExecutor:
    """Upbit exchange executor for KRW crypto trading."""

    def __init__(self, access_key: str, secret_key: str) -> None:
        if not access_key or not secret_key:
            raise ExecutionError("Upbit API keys are required")
        self._upbit = pyupbit.Upbit(access_key, secret_key)
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Verify API keys are valid."""
        try:
            balances = self._upbit.get_balances()
            if balances is None:
                raise ExecutionError("Failed to connect to Upbit API")
        except Exception as e:
            raise ExecutionError(f"Upbit connection failed: {e}") from e

    @property
    def exchange(self) -> Exchange:
        return Exchange.UPBIT

    def submit_order(self, order: Order) -> ExecutionResult:
        """Submit a market order to Upbit."""
        now = datetime.now()

        try:
            if order.side == OrderSide.BUY:
                # Upbit buy: specify total KRW to spend
                trade_value = order.quantity * order.price
                result = self._upbit.buy_market_order(order.symbol, trade_value)
            elif order.side == OrderSide.SELL:
                result = self._upbit.sell_market_order(order.symbol, order.quantity)
            else:
                return ExecutionResult(
                    order=replace(order, status=OrderStatus.REJECTED),
                    success=False,
                    timestamp=now,
                    error_message=f"Unsupported order side: {order.side}",
                )

            if result is None or "error" in result:
                error_msg = result.get("error", {}).get("message", "Unknown error") if result else "No response"
                logger.error("Upbit order failed: %s", error_msg)
                return ExecutionResult(
                    order=replace(order, status=OrderStatus.REJECTED),
                    success=False,
                    timestamp=now,
                    error_message=str(error_msg),
                )

            # Parse fill details from response
            fill_price = order.price  # Upbit market orders fill near current price
            fill_qty = order.quantity
            trade_value = fill_price * fill_qty
            commission = estimate_commission(trade_value, Exchange.UPBIT)
            slippage = estimate_slippage(trade_value, Exchange.UPBIT)

            logger.info(
                "Upbit order filled: %s %s %.4f @ %.2f",
                order.side.value, order.symbol, fill_qty, fill_price,
            )

            return ExecutionResult(
                order=replace(
                    order,
                    status=OrderStatus.FILLED,
                    filled_price=fill_price,
                    filled_quantity=fill_qty,
                ),
                success=True,
                filled_price=fill_price,
                filled_quantity=fill_qty,
                commission=commission,
                slippage=slippage,
                timestamp=now,
            )

        except Exception as e:
            logger.exception("Upbit order error")
            return ExecutionResult(
                order=replace(order, status=OrderStatus.REJECTED),
                success=False,
                timestamp=now,
                error_message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending Upbit order."""
        try:
            result = self._upbit.cancel_order(order_id)
            return result is not None and "error" not in result
        except Exception:
            logger.exception("Failed to cancel Upbit order %s", order_id)
            return False

    def get_account_info(self) -> AccountInfo:
        """Get Upbit account balance and positions."""
        balances = self._upbit.get_balances()
        if balances is None:
            raise ExecutionError("Failed to get Upbit balances")

        cash = 0.0
        positions: dict[str, float] = {}

        for item in balances:
            currency = item.get("currency", "")
            balance = float(item.get("balance", 0))

            if currency == "KRW":
                cash = balance
            elif balance > 0:
                symbol = f"KRW-{currency}"
                positions[symbol] = balance

        # Compute total value
        position_value = 0.0
        for symbol, qty in positions.items():
            price = self.get_current_price(symbol)
            position_value += qty * price

        return AccountInfo(
            exchange=Exchange.UPBIT,
            total_value=cash + position_value,
            cash_balance=cash,
            positions=positions,
            timestamp=datetime.now(),
        )

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Upbit."""
        try:
            price = pyupbit.get_current_price(symbol)
            return float(price) if price is not None else 0.0
        except Exception:
            logger.warning("Failed to get price for %s", symbol)
            return 0.0
