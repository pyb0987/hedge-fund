"""Tests for DataCollector — data collection for paper trading."""

from datetime import datetime

import pandas as pd
import pytest

from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.data.collector import DataCollector


class FakeProvider:
    """Fake data provider for testing."""

    def __init__(self, exchange_name: str, data: dict[str, pd.DataFrame] | None = None):
        self._exchange_name = exchange_name
        self._data = data or {}

    @property
    def exchange_name(self) -> str:
        return self._exchange_name

    def get_ohlcv(self, symbol: str, interval: str = "day", count: int = 200) -> pd.DataFrame:
        if symbol in self._data:
            return self._data[symbol]
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_ohlcv_range(self, symbol: str, start: datetime, end: datetime, interval: str = "day") -> pd.DataFrame:
        return self.get_ohlcv(symbol, interval)

    def get_available_symbols(self) -> list[str]:
        return list(self._data.keys())


class FakeStrategy:
    """Fake strategy for testing."""

    def __init__(self, name: str, exchange: Exchange, universe: list[str]):
        self._name = name
        self._exchange = exchange
        self._universe = universe

    @property
    def name(self) -> str:
        return self._name

    @property
    def exchange(self) -> Exchange:
        return self._exchange

    def get_universe(self) -> list[str]:
        return list(self._universe)

    def generate_signals(self, data: dict[str, pd.DataFrame], timestamp: datetime) -> list[Signal]:
        return []


def _make_ohlcv(n: int = 30, base_price: float = 100.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": [base_price] * n,
            "high": [base_price * 1.01] * n,
            "low": [base_price * 0.99] * n,
            "close": [base_price] * n,
            "volume": [1000.0] * n,
        },
        index=dates,
    )


class TestDataCollector:
    """Tests for DataCollector."""

    def test_collect_all_basic(self):
        """Basic collection with one strategy, one symbol."""
        df = _make_ohlcv(30, 50000.0)
        providers = {
            "upbit": FakeProvider("upbit", {"KRW-BTC": df}),
        }
        strategies = {
            "crypto_momentum": FakeStrategy("crypto_momentum", Exchange.UPBIT, ["KRW-BTC"]),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert "crypto_momentum" in result.data
        assert "KRW-BTC" in result.data["crypto_momentum"]
        assert len(result.data["crypto_momentum"]["KRW-BTC"]) == 30
        assert result.latest_prices["KRW-BTC"] == 50000.0
        assert len(result.errors) == 0

    def test_collect_all_multiple_strategies(self):
        """Collection across multiple strategies and providers."""
        btc_df = _make_ohlcv(30, 50000.0)
        spy_df = _make_ohlcv(30, 450.0)

        providers = {
            "upbit": FakeProvider("upbit", {"KRW-BTC": btc_df}),
            "yfinance": FakeProvider("yfinance", {"SPY": spy_df}),
        }
        strategies = {
            "crypto_momentum": FakeStrategy("crypto_momentum", Exchange.UPBIT, ["KRW-BTC"]),
            "etf_mean_reversion": FakeStrategy("etf_mean_reversion", Exchange.ALPACA, ["SPY"]),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert len(result.data) == 2
        assert result.latest_prices["KRW-BTC"] == 50000.0
        assert result.latest_prices["SPY"] == 450.0
        assert len(result.errors) == 0

    def test_collect_missing_provider(self):
        """Missing provider records error but doesn't crash."""
        strategies = {
            "etf_mean_reversion": FakeStrategy("etf_mean_reversion", Exchange.ALPACA, ["SPY"]),
        }
        providers: dict[str, FakeProvider] = {}

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert len(result.errors) == 1
        assert "no provider" in result.errors[0]

    def test_collect_empty_universe(self):
        """Strategy with empty universe records error."""
        providers = {"upbit": FakeProvider("upbit")}
        strategies = {
            "crypto_momentum": FakeStrategy("crypto_momentum", Exchange.UPBIT, []),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert len(result.errors) == 1
        assert "empty universe" in result.errors[0]

    def test_collect_empty_data(self):
        """Symbol with no data records error."""
        providers = {"upbit": FakeProvider("upbit", {})}
        strategies = {
            "crypto_momentum": FakeStrategy("crypto_momentum", Exchange.UPBIT, ["KRW-ETH"]),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert len(result.errors) == 1
        assert "empty data" in result.errors[0]

    def test_resolve_provider_krw_to_upbit(self):
        """KRW-* symbols resolve to upbit provider."""
        providers = {
            "upbit": FakeProvider("upbit", {"KRW-BTC": _make_ohlcv()}),
            "yfinance": FakeProvider("yfinance"),
        }
        strategies = {
            "test": FakeStrategy("test", Exchange.UPBIT, ["KRW-BTC"]),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert "KRW-BTC" in result.data["test"]

    def test_resolve_provider_etf_to_yfinance(self):
        """Non-KRW symbols resolve to yfinance provider."""
        providers = {
            "upbit": FakeProvider("upbit"),
            "yfinance": FakeProvider("yfinance", {"SPY": _make_ohlcv(30, 450.0)}),
        }
        strategies = {
            "test": FakeStrategy("test", Exchange.ALPACA, ["SPY"]),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        assert "SPY" in result.data["test"]

    def test_dual_momentum_cross_exchange(self):
        """Dual momentum collects from both exchanges."""
        btc_df = _make_ohlcv(60, 50000.0)
        spy_df = _make_ohlcv(60, 450.0)
        tlt_df = _make_ohlcv(60, 100.0)

        providers = {
            "upbit": FakeProvider("upbit", {"KRW-BTC": btc_df}),
            "yfinance": FakeProvider("yfinance", {"SPY": spy_df, "TLT": tlt_df}),
        }
        strategies = {
            "dual_momentum": FakeStrategy(
                "dual_momentum", Exchange.UPBIT, ["KRW-BTC", "SPY", "TLT"]
            ),
        }

        collector = DataCollector(providers, strategies)
        result = collector.collect_all()

        dm_data = result.data["dual_momentum"]
        assert "KRW-BTC" in dm_data
        assert "SPY" in dm_data
        assert "TLT" in dm_data
        assert len(result.errors) == 0
