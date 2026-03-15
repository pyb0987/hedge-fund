"""Tests for paper executor and strategy state persistence."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from hedgefund.core.enums import Exchange, OrderSide, OrderStatus
from hedgefund.core.models import Order
from hedgefund.execution.executors.paper_executor import PaperExecutor, PaperPosition
from hedgefund.execution.state import (
    load_state,
    load_strategy_state,
    save_state,
    save_strategy_state,
)


class TestStatePersistence:
    """Tests for save/load of PaperExecutor state."""

    def test_save_and_load_empty(self, tmp_path: Path):
        """Save and restore executor with no positions."""
        executor = PaperExecutor(Exchange.UPBIT, 1_000_000.0)
        state_path = tmp_path / "state.json"

        save_state(executor, state_path)
        restored = load_state(state_path)

        assert restored is not None
        assert restored.exchange == Exchange.UPBIT
        assert restored.cash == 1_000_000.0
        assert restored.get_account_info().positions == {}

    def test_save_and_load_with_positions(self, tmp_path: Path):
        """Save and restore executor with open positions."""
        executor = PaperExecutor(Exchange.UPBIT, 1_000_000.0)

        # Simulate a buy
        executor.set_prices({"KRW-BTC": 50_000_000.0})
        order = Order(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            side=OrderSide.BUY,
            quantity=0.01,
            price=50_000_000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )
        result = executor.submit_order(order)
        assert result.success

        cash_after = executor.cash
        state_path = tmp_path / "state.json"

        save_state(executor, state_path)
        restored = load_state(state_path)

        assert restored is not None
        assert restored.cash == pytest.approx(cash_after, rel=1e-6)
        assert restored.get_position_quantity("KRW-BTC") == pytest.approx(0.01)

    def test_load_nonexistent(self, tmp_path: Path):
        """Loading from nonexistent path returns None."""
        result = load_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_wrong_version(self, tmp_path: Path):
        """Loading state with wrong version returns None."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 999,
            "exchange": "upbit",
            "cash": 1000,
        }))

        result = load_state(state_path)
        assert result is None

    def test_roundtrip_preserves_exchange(self, tmp_path: Path):
        """Exchange type is preserved through save/load."""
        for exchange in [Exchange.UPBIT, Exchange.ALPACA]:
            executor = PaperExecutor(exchange, 500.0)
            path = tmp_path / f"{exchange.value}.json"

            save_state(executor, path)
            restored = load_state(path)

            assert restored is not None
            assert restored.exchange == exchange

    def test_state_file_is_valid_json(self, tmp_path: Path):
        """Saved state file is valid, human-readable JSON."""
        executor = PaperExecutor(Exchange.UPBIT, 1_000_000.0)
        state_path = tmp_path / "state.json"

        save_state(executor, state_path)

        with open(state_path) as f:
            data = json.load(f)

        assert data["version"] == 2
        assert data["exchange"] == "upbit"
        assert data["cash"] == 1_000_000.0
        assert isinstance(data["positions"], list)
        assert isinstance(data["trades"], list)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """save_state creates parent directories if needed."""
        state_path = tmp_path / "deep" / "nested" / "state.json"

        executor = PaperExecutor(Exchange.ALPACA, 500.0)
        save_state(executor, state_path)

        assert state_path.exists()

    def test_multiple_positions_roundtrip(self, tmp_path: Path):
        """Multiple positions survive save/load."""
        executor = PaperExecutor(Exchange.UPBIT, 10_000_000.0)
        executor.set_prices({
            "KRW-BTC": 50_000_000.0,
            "KRW-ETH": 3_000_000.0,
        })

        for sym, qty in [("KRW-BTC", 0.01), ("KRW-ETH", 0.5)]:
            order = Order(
                strategy_name="test",
                symbol=sym,
                exchange=Exchange.UPBIT,
                side=OrderSide.BUY,
                quantity=qty,
                price=executor.get_current_price(sym),
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
            )
            executor.submit_order(order)

        state_path = tmp_path / "state.json"
        save_state(executor, state_path)
        restored = load_state(state_path)

        assert restored is not None
        assert restored.get_position_quantity("KRW-BTC") == pytest.approx(0.01)
        assert restored.get_position_quantity("KRW-ETH") == pytest.approx(0.5)


    def test_roundtrip_preserves_strategy_name(self, tmp_path: Path):
        """Strategy name on positions survives save/load roundtrip."""
        executor = PaperExecutor(Exchange.UPBIT, 10_000_000.0)
        executor.set_prices({"KRW-BTC": 50_000_000.0, "KRW-ETH": 3_000_000.0})

        order_a = Order(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            side=OrderSide.BUY,
            quantity=0.01,
            price=50_000_000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )
        order_b = Order(
            strategy_name="dual_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            side=OrderSide.BUY,
            quantity=0.02,
            price=50_000_000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )
        executor.submit_order(order_a)
        executor.submit_order(order_b)

        state_path = tmp_path / "state.json"
        save_state(executor, state_path)
        restored = load_state(state_path)

        assert restored is not None
        assert restored.get_strategy_position_quantity("crypto_momentum", "KRW-BTC") == pytest.approx(0.01)
        assert restored.get_strategy_position_quantity("dual_momentum", "KRW-BTC") == pytest.approx(0.02)
        assert restored.get_position_quantity("KRW-BTC") == pytest.approx(0.03)

    def test_v1_state_rejected(self, tmp_path: Path):
        """Old v1 state files are rejected (version mismatch -> fresh start)."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "exchange": "upbit",
            "cash": 1_000_000,
            "initial_cash": 1_000_000,
            "positions": {"KRW-BTC": {
                "symbol": "KRW-BTC",
                "quantity": 0.01,
                "avg_entry_price": 50_000_000,
                "exchange": "upbit",
            }},
            "trades": [],
        }))

        result = load_state(state_path)
        assert result is None


class TestStrategyState:
    """Tests for strategy rebalancing state persistence."""

    def test_save_and_load_strategy_state(self, tmp_path: Path):
        """Save and restore strategy rebalance dates."""
        from hedgefund.config.schemas import CryptoMomentumConfig, DualMomentumConfig
        from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy
        from hedgefund.strategies.dual_momentum import DualMomentumStrategy

        crypto = CryptoMomentumStrategy(CryptoMomentumConfig(), universe=["KRW-BTC"])
        crypto.last_rebalance_date = datetime(2024, 3, 10)

        dual = DualMomentumStrategy(DualMomentumConfig())
        dual.last_rebalance_date = datetime(2024, 3, 1)

        strategies = {"crypto_momentum": crypto, "dual_momentum": dual}
        state_path = tmp_path / "strategies.json"

        save_strategy_state(strategies, state_path)
        restored = load_strategy_state(state_path)

        assert restored["crypto_momentum"] == datetime(2024, 3, 10)
        assert restored["dual_momentum"] == datetime(2024, 3, 1)

    def test_load_nonexistent_strategy_state(self, tmp_path: Path):
        """Loading from nonexistent path returns empty dict."""
        result = load_strategy_state(tmp_path / "nonexistent.json")
        assert result == {}

    def test_save_with_none_dates(self, tmp_path: Path):
        """Strategies with no rebalance date save as null."""
        from hedgefund.config.schemas import CryptoMomentumConfig
        from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy

        crypto = CryptoMomentumStrategy(CryptoMomentumConfig(), universe=["KRW-BTC"])
        strategies = {"crypto_momentum": crypto}
        state_path = tmp_path / "strategies.json"

        save_strategy_state(strategies, state_path)
        restored = load_strategy_state(state_path)

        assert restored["crypto_momentum"] is None

    def test_strategy_without_rebalance_date_attribute(self, tmp_path: Path):
        """Strategies without last_rebalance_date are saved as None."""

        class SimpleStrategy:
            name = "simple"

        strategies = {"simple": SimpleStrategy()}
        state_path = tmp_path / "strategies.json"

        save_strategy_state(strategies, state_path)
        restored = load_strategy_state(state_path)
        assert restored["simple"] is None
