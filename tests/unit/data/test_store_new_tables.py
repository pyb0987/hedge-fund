"""Tests for position_snapshots and strategy_decisions tables in DataStore."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from hedgefund.data.store import DataStore


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    db_path = str(tmp_path / "test.db")
    return DataStore(db_path)


class TestPositionSnapshots:
    def test_save_and_load(self, store: DataStore) -> None:
        ts = datetime(2026, 3, 1, 21, 0)
        store.save_position_snapshot(
            timestamp=ts,
            symbol="KRW-BTC",
            exchange="upbit",
            strategy_name="crypto_momentum",
            quantity=0.01,
            avg_entry_price=50_000_000,
            market_price=52_000_000,
            market_value=520_000,
            unrealized_pnl=20_000,
        )
        df = store.load_position_snapshots()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["symbol"] == "KRW-BTC"
        assert row["strategy_name"] == "crypto_momentum"
        assert row["quantity"] == 0.01
        assert row["avg_entry_price"] == 50_000_000
        assert row["market_price"] == 52_000_000
        assert row["unrealized_pnl"] == 20_000

    def test_multiple_positions(self, store: DataStore) -> None:
        ts = datetime(2026, 3, 1, 21, 0)
        store.save_position_snapshot(
            timestamp=ts, symbol="KRW-BTC", exchange="upbit",
            strategy_name="crypto_momentum", quantity=0.01,
            avg_entry_price=50_000_000, market_price=52_000_000,
            market_value=520_000, unrealized_pnl=20_000,
        )
        store.save_position_snapshot(
            timestamp=ts, symbol="BIL", exchange="alpaca",
            strategy_name="treasury_park", quantity=100,
            avg_entry_price=91.5, market_price=91.6,
            market_value=9160, unrealized_pnl=10,
        )
        df = store.load_position_snapshots()
        assert len(df) == 2
        strategies = set(df["strategy_name"].values)
        assert strategies == {"crypto_momentum", "treasury_park"}

    def test_empty_load(self, store: DataStore) -> None:
        df = store.load_position_snapshots()
        assert df.empty


class TestStrategyDecisions:
    def test_save_and_load(self, store: DataStore) -> None:
        ts = datetime(2026, 3, 1, 21, 0)
        store.save_strategy_decision(
            timestamp=ts,
            strategy_name="crypto_momentum",
            action="hold",
            reason="too_early",
            signals_generated=0,
            target_allocation=0.10,
            actual_allocation=0.0,
            dd_multiplier=1.0,
        )
        df = store.load_strategy_decisions()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["strategy_name"] == "crypto_momentum"
        assert row["action"] == "hold"
        assert row["reason"] == "too_early"
        assert row["signals_generated"] == 0
        assert row["target_allocation"] == 0.10
        assert row["dd_multiplier"] == 1.0

    def test_multiple_strategies(self, store: DataStore) -> None:
        ts = datetime(2026, 3, 1, 21, 0)
        store.save_strategy_decision(
            timestamp=ts, strategy_name="crypto_momentum",
            action="rebalance", reason="holding_period_met",
            signals_generated=5, target_allocation=0.10,
            actual_allocation=0.0, dd_multiplier=1.0,
        )
        store.save_strategy_decision(
            timestamp=ts, strategy_name="dual_momentum",
            action="hold", reason="same_month",
            signals_generated=0, target_allocation=0.15,
            actual_allocation=0.0, dd_multiplier=1.0,
        )
        store.save_strategy_decision(
            timestamp=ts, strategy_name="treasury_park",
            action="rebalance", reason="no_gate",
            signals_generated=1, target_allocation=0.45,
            actual_allocation=0.0, dd_multiplier=1.0,
        )
        df = store.load_strategy_decisions()
        assert len(df) == 3
        rebalances = df[df["action"] == "rebalance"]
        assert len(rebalances) == 2

    def test_empty_load(self, store: DataStore) -> None:
        df = store.load_strategy_decisions()
        assert df.empty
