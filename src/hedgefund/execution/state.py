"""Paper executor and strategy state persistence — save/load to JSON.

프로세스 재시작 시 PaperExecutor의 상태(현금, 포지션)와
전략의 리밸런싱 상태(last_rebalance_date)를 복원합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from hedgefund.core.enums import Exchange
from hedgefund.execution.executors.paper_executor import PaperExecutor, PaperPosition

logger = logging.getLogger(__name__)

_STATE_VERSION = 1


def save_state(executor: PaperExecutor, path: Path) -> None:
    """Save PaperExecutor state to JSON file.

    Args:
        executor: the paper executor to save
        path: file path for the state JSON
    """
    positions = {}
    for sym, pos in executor._positions.items():
        positions[sym] = {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_entry_price": pos.avg_entry_price,
            "exchange": pos.exchange.value,
        }

    trades = []
    for trade in executor._trades:
        trades.append({
            "strategy_name": trade.strategy_name,
            "symbol": trade.symbol,
            "exchange": trade.exchange.value,
            "side": trade.side.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "commission": trade.commission,
            "slippage": trade.slippage,
            "timestamp": trade.timestamp.isoformat(),
            "pnl": trade.pnl,
        })

    state = {
        "version": _STATE_VERSION,
        "exchange": executor.exchange.value,
        "cash": executor.cash,
        "initial_cash": executor._initial_cash,
        "positions": positions,
        "trades": trades,
        "saved_at": datetime.now().isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info("Saved paper executor state to %s (cash=%.0f, positions=%d)",
                path, executor.cash, len(positions))


def load_state(path: Path) -> PaperExecutor | None:
    """Load PaperExecutor state from JSON file.

    Args:
        path: file path for the state JSON

    Returns:
        Restored PaperExecutor, or None if file doesn't exist
    """
    if not path.exists():
        logger.info("No state file at %s — starting fresh", path)
        return None

    with open(path) as f:
        state = json.load(f)

    version = state.get("version", 0)
    if version != _STATE_VERSION:
        logger.warning("State version mismatch: expected %d, got %d — starting fresh",
                       _STATE_VERSION, version)
        return None

    exchange = Exchange(state["exchange"])
    executor = PaperExecutor(
        exchange=exchange,
        initial_cash=state["initial_cash"],
    )

    # Restore cash (may differ from initial due to trades)
    executor._cash = state["cash"]

    # Restore positions
    for _sym, pos_data in state["positions"].items():
        executor._positions[pos_data["symbol"]] = PaperPosition(
            symbol=pos_data["symbol"],
            quantity=pos_data["quantity"],
            avg_entry_price=pos_data["avg_entry_price"],
            exchange=Exchange(pos_data["exchange"]),
        )

    logger.info("Restored paper executor: exchange=%s, cash=%.0f, positions=%d",
                exchange.value, executor.cash, len(executor._positions))

    return executor


def save_strategy_state(
    strategies: dict[str, Any],
    path: Path,
) -> None:
    """Save strategy rebalancing state to JSON.

    Persists last_rebalance_date for strategies that gate on rebalance timing.
    """
    state: dict[str, str | None] = {}

    for name, strategy in strategies.items():
        last_date = getattr(strategy, "last_rebalance_date", None)
        state[name] = last_date.isoformat() if last_date is not None else None

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": _STATE_VERSION, "last_rebalance": state}, f, indent=2)

    logger.info("Saved strategy state: %s", {k: v is not None for k, v in state.items()})


def load_strategy_state(path: Path) -> dict[str, datetime | None]:
    """Load strategy rebalancing state from JSON.

    Returns:
        strategy_name → last_rebalance_date (None if never rebalanced)
    """
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    if data.get("version") != _STATE_VERSION:
        logger.warning("Strategy state version mismatch — starting fresh")
        return {}

    result: dict[str, datetime | None] = {}
    for name, iso_str in data.get("last_rebalance", {}).items():
        if iso_str is not None:
            result[name] = datetime.fromisoformat(iso_str)
        else:
            result[name] = None

    logger.info("Restored strategy state: %s", {k: v is not None for k, v in result.items()})
    return result
