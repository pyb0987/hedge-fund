"""Tests for Treasury Park strategy (cash → BIL)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import TreasuryParkConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.strategies.treasury_park import TreasuryParkStrategy


@pytest.fixture
def strategy() -> TreasuryParkStrategy:
    config = TreasuryParkConfig()
    return TreasuryParkStrategy(config=config)


@pytest.fixture
def mock_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    # BIL: very low volatility, slight uptrend (T-bill yield)
    close = 91.0 * np.cumprod(1 + rng.normal(0.0002, 0.0005, 60))
    return {
        "BIL": pd.DataFrame(
            {
                "open": close * 0.9999,
                "high": close * 1.0002,
                "low": close * 0.9998,
                "close": close,
                "volume": rng.uniform(1e6, 5e6, 60),
            },
            index=dates,
        )
    }


class TestTreasuryParkStrategy:
    def test_name(self, strategy: TreasuryParkStrategy) -> None:
        assert strategy.name == "treasury_park"

    def test_exchange(self, strategy: TreasuryParkStrategy) -> None:
        assert strategy.exchange == Exchange.ALPACA

    def test_universe(self, strategy: TreasuryParkStrategy) -> None:
        assert strategy.get_universe() == ["BIL"]

    def test_custom_symbol(self) -> None:
        config = TreasuryParkConfig(symbol="SHV")
        strat = TreasuryParkStrategy(config=config)
        assert strat.get_universe() == ["SHV"]

    def test_generate_signals_always_long(
        self, strategy: TreasuryParkStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(mock_data, datetime(2024, 3, 20))
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == SignalDirection.LONG
        assert sig.strength == 1.0
        assert sig.symbol == "BIL"
        assert sig.strategy_name == "treasury_park"

    def test_generate_signals_no_data(
        self, strategy: TreasuryParkStrategy
    ) -> None:
        signals = strategy.generate_signals({}, datetime(2024, 3, 20))
        assert len(signals) == 0

    def test_backtest_weights_shape(
        self, strategy: TreasuryParkStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = mock_data["BIL"].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert weights.shape == (60, 1)
        assert "BIL" in weights.columns

    def test_backtest_weights_always_one(
        self, strategy: TreasuryParkStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = mock_data["BIL"].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert (weights["BIL"] == 1.0).all()

    def test_backtest_weights_missing_symbol(
        self, strategy: TreasuryParkStrategy
    ) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        weights = strategy.backtest_weights({}, dates)
        assert weights.empty or (weights == 0.0).all().all()


class TestTreasuryParkConfig:
    def test_defaults(self) -> None:
        config = TreasuryParkConfig()
        assert config.name == "treasury_park"
        assert config.symbol == "BIL"
        assert config.exchange == "alpaca"

    def test_custom_symbol(self) -> None:
        config = TreasuryParkConfig(symbol="SHV")
        assert config.symbol == "SHV"
