"""Alpaca exchange executor — US ETF/stock trading via Alpaca API.

Alpaca Markets API를 통해 미국 ETF 현물 거래를 실행합니다.
무료 계정으로 소수점 거래 가능 ($1 최소).
"""

import logging
from dataclasses import replace
from datetime import datetime

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus
from hedgefund.core.exceptions import ExecutionError
from hedgefund.core.models import Order
from hedgefund.execution.cost_model import estimate_commission, estimate_slippage
from hedgefund.execution.protocols import AccountInfo, ExecutionResult

logger = logging.getLogger(__name__)


class AlpacaExecutor:
    """Alpaca executor for US ETF trading."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        if not api_key or not secret_key:
            raise ExecutionError("Alpaca API keys are required")

        try:
            import alpaca_trade_api as tradeapi
        except ImportError as e:
            raise ExecutionError(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            ) from e

        base_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        self._api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        self._paper = paper
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Verify API keys are valid."""
        try:
            account = self._api.get_account()
            if account.status != "ACTIVE":
                raise ExecutionError(f"Alpaca account not active: {account.status}")
        except Exception as e:
            raise ExecutionError(f"Alpaca connection failed: {e}") from e

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    def submit_order(self, order: Order) -> ExecutionResult:
        """Submit a market order to Alpaca."""
        now = datetime.now()

        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"

            # Use notional (dollar amount) for buys, qty for sells
            alpaca_order = self._api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                type="market",
                time_in_force="day",
            )

            # Market orders fill quickly — get details
            fill_price = float(alpaca_order.filled_avg_price or order.price)
            fill_qty = float(alpaca_order.filled_qty or order.quantity)
            trade_value = fill_price * fill_qty
            commission = estimate_commission(trade_value, Exchange.ALPACA)
            slippage = estimate_slippage(trade_value, Exchange.ALPACA)

            logger.info(
                "Alpaca order filled: %s %s %.4f @ $%.2f",
                side, order.symbol, fill_qty, fill_price,
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
            logger.exception("Alpaca order error")
            return ExecutionResult(
                order=replace(order, status=OrderStatus.REJECTED),
                success=False,
                timestamp=now,
                error_message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending Alpaca order."""
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel Alpaca order %s", order_id)
            return False

    def get_account_info(self) -> AccountInfo:
        """Get Alpaca account balance and positions."""
        try:
            account = self._api.get_account()
            positions = self._api.list_positions()

            pos_dict: dict[str, float] = {}
            for pos in positions:
                pos_dict[pos.symbol] = float(pos.qty)

            return AccountInfo(
                exchange=Exchange.ALPACA,
                total_value=float(account.equity),
                cash_balance=float(account.cash),
                positions=pos_dict,
                timestamp=datetime.now(),
            )
        except Exception as e:
            raise ExecutionError(f"Failed to get Alpaca account info: {e}") from e

    def get_current_price(self, symbol: str) -> float:
        """Get latest price from Alpaca."""
        try:
            quote = self._api.get_latest_quote(symbol)
            return float(quote.ap) if quote else 0.0  # ask price
        except Exception:
            logger.warning("Failed to get Alpaca price for %s", symbol)
            return 0.0
