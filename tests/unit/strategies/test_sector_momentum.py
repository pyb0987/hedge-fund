"""Tests for Sector ETF Momentum strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import SectorMomentumConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.strategies.sector_momentum import SectorMomentumStrategy


@pytest.fixture
def strategy() -> SectorMomentumStrategy:
    config = SectorMomentumConfig(
        lookback_days=20, top_n=3, holding_days=14,
        universe=["XLK", "XLF", "XLE", "XLV", "XLI"],
    )
    return SectorMomentumStrategy(config=config)


@pytest.fixture
def mock_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    data = {}
    # XLK: strong uptrend (tech rally)
    # XLF: moderate uptrend
    # XLE: flat
    # XLV: slight decline
    # XLI: strong decline
    for sym, drift in [("XLK", 0.003), ("XLF", 0.001), ("XLE", 0.0), ("XLV", -0.001), ("XLI", -0.002)]:
        close = 100.0 * np.cumprod(1 + rng.normal(drift, 0.01, 60))
        data[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.uniform(1e7, 1e8, 60),
            },
            index=dates,
        )
    return data


class TestSectorMomentumStrategy:
    def test_name(self, strategy: SectorMomentumStrategy) -> None:
        assert strategy.name == "sector_momentum"

    def test_exchange(self, strategy: SectorMomentumStrategy) -> None:
        assert strategy.exchange == Exchange.ALPACA

    def test_universe(self, strategy: SectorMomentumStrategy) -> None:
        assert len(strategy.get_universe()) == 5

    def test_generate_signals_selects_top_n(
        self, strategy: SectorMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(mock_data, datetime(2024, 3, 20))
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        flat_signals = [s for s in signals if s.direction == SignalDirection.FLAT]
        assert len(long_signals) <= 3  # top_n = 3
        assert len(flat_signals) >= 2  # at least bottom 2

    def test_rebalancing_gate(
        self, strategy: SectorMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        # First call: should generate signals
        signals1 = strategy.generate_signals(mock_data, datetime(2024, 3, 1))
        assert len(signals1) > 0

        # Same day + 5: should NOT generate (holding_days=14)
        signals2 = strategy.generate_signals(mock_data, datetime(2024, 3, 6))
        assert len(signals2) == 0

        # After 14 days: should generate again
        signals3 = strategy.generate_signals(mock_data, datetime(2024, 3, 15))
        assert len(signals3) > 0

    def test_insufficient_data(self, strategy: SectorMomentumStrategy) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = {"XLK": pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)}
        signals = strategy.generate_signals(data, datetime(2024, 1, 8))
        assert len(signals) == 0

    def test_backtest_weights_shape(
        self, strategy: SectorMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert weights.shape == (60, 5)

    def test_backtest_weights_bounded(
        self, strategy: SectorMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert (weights >= 0).all().all()
        assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_backtest_weights_respects_holding_days(
        self, strategy: SectorMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)
        # Weights should change every holding_days, not every day
        weight_changes = weights.diff().abs().sum(axis=1)
        # After warmup, most days should have 0 change (carry forward)
        nonzero_changes = (weight_changes > 1e-9).sum()
        total_active_days = len(dates) - strategy._config.lookback_days
        # Should rebalance roughly every 14 days
        assert nonzero_changes < total_active_days * 0.5


class TestSectorMomentumConfig:
    def test_defaults(self) -> None:
        config = SectorMomentumConfig()
        assert config.name == "sector_momentum"
        assert config.exchange == "alpaca"
        assert len(config.universe) == 11  # all 11 SPDR sector ETFs
        assert config.top_n == 3
        assert config.holding_days == 14

    def test_custom_universe(self) -> None:
        config = SectorMomentumConfig(universe=["XLK", "XLF", "XLE"])
        assert len(config.universe) == 3
