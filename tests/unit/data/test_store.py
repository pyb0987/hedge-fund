"""Tests for SQLite data store."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.data.store import DataStore


@pytest.fixture
def store(tmp_path) -> DataStore:
    """Create a temporary DataStore."""
    db_path = str(tmp_path / "test.db")
    return DataStore(db_path=db_path)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
    }, index=dates)


class TestDataStore:
    def test_save_and_load(self, store: DataStore, sample_ohlcv_df: pd.DataFrame) -> None:
        inserted = store.save_ohlcv("KRW-BTC", "upbit", sample_ohlcv_df)
        assert inserted == 5

        loaded = store.load_ohlcv("KRW-BTC", "upbit")
        assert len(loaded) == 5
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]

    def test_has_data(self, store: DataStore, sample_ohlcv_df: pd.DataFrame) -> None:
        assert store.has_data("KRW-BTC", "upbit") is False
        store.save_ohlcv("KRW-BTC", "upbit", sample_ohlcv_df)
        assert store.has_data("KRW-BTC", "upbit") is True

    def test_no_duplicates(self, store: DataStore, sample_ohlcv_df: pd.DataFrame) -> None:
        store.save_ohlcv("KRW-BTC", "upbit", sample_ohlcv_df)
        store.save_ohlcv("KRW-BTC", "upbit", sample_ohlcv_df)
        loaded = store.load_ohlcv("KRW-BTC", "upbit")
        assert len(loaded) == 5

    def test_load_empty(self, store: DataStore) -> None:
        loaded = store.load_ohlcv("NONEXISTENT", "upbit")
        assert loaded.empty

    def test_date_range_filter(
        self, store: DataStore, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store.save_ohlcv("SPY", "yfinance", sample_ohlcv_df)
        loaded = store.load_ohlcv(
            "SPY", "yfinance",
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 4),
        )
        assert len(loaded) <= 3

    def test_save_snapshot(self, store: DataStore) -> None:
        store.save_snapshot(
            timestamp=datetime(2024, 6, 1),
            total_value=1_000_000,
            cash=500_000,
            unrealized_pnl=50_000,
            realized_pnl=10_000,
            drawdown=0.05,
            peak_value=1_050_000,
        )
        snapshots = store.load_snapshots()
        assert len(snapshots) == 1

    def test_save_empty_df(self, store: DataStore) -> None:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        inserted = store.save_ohlcv("TEST", "test", empty)
        assert inserted == 0

    def test_save_and_load_trade(self, store: DataStore) -> None:
        store.save_trade(
            symbol="KRW-BTC",
            exchange="upbit",
            side="buy",
            quantity=0.01,
            price=50_000_000,
            commission=1250.0,
            slippage=750.0,
            strategy_name="crypto_momentum",
            timestamp=datetime(2024, 3, 1, 21, 0),
            pnl=None,
        )
        trades = store.load_trades()
        assert len(trades) == 1
        assert trades.iloc[0]["symbol"] == "KRW-BTC"
        assert trades.iloc[0]["commission"] == 1250.0

    def test_save_and_load_signal(self, store: DataStore) -> None:
        store.save_signal(
            timestamp=datetime(2024, 3, 1, 21, 0),
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange="upbit",
            direction="long",
            strength=0.85,
            metadata={"momentum_score": 0.15, "rank": 1.0},
        )
        signals = store.load_signals()
        assert len(signals) == 1
        assert signals.iloc[0]["strategy_name"] == "crypto_momentum"
        assert signals.iloc[0]["direction"] == "long"
        assert signals.iloc[0]["strength"] == 0.85

    def test_save_signal_no_metadata(self, store: DataStore) -> None:
        store.save_signal(
            timestamp=datetime(2024, 3, 1),
            strategy_name="test",
            symbol="SPY",
            exchange="alpaca",
            direction="flat",
            strength=0.0,
            metadata=None,
        )
        signals = store.load_signals()
        assert len(signals) == 1
        assert signals.iloc[0]["metadata"] is None

    def test_multiple_trades_ordering(self, store: DataStore) -> None:
        for i, ts in enumerate([
            datetime(2024, 3, 1, 21, 0),
            datetime(2024, 3, 8, 21, 0),
            datetime(2024, 3, 15, 21, 0),
        ]):
            store.save_trade(
                symbol="KRW-BTC", exchange="upbit", side="buy",
                quantity=0.01, price=50_000_000 + i * 100_000,
                commission=1250.0, slippage=750.0,
                strategy_name="crypto_momentum", timestamp=ts,
            )
        trades = store.load_trades()
        assert len(trades) == 3
        # Should be ordered by timestamp
        assert trades.iloc[0]["timestamp"] < trades.iloc[2]["timestamp"]
