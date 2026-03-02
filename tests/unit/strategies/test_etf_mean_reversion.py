"""Tests for ETF mean reversion strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import EtfMeanReversionConfig
from hedgefund.core.enums import SignalDirection
from hedgefund.strategies.etf_mean_reversion import EtfMeanReversionStrategy


@pytest.fixture
def strategy() -> EtfMeanReversionStrategy:
    config = EtfMeanReversionConfig(lookback_days=20, z_entry_threshold=-1.5, z_exit_threshold=1.5)
    return EtfMeanReversionStrategy(config=config, universe=["SPY", "TLT", "GLD"])


@pytest.fixture
def mock_data() -> dict[str, pd.DataFrame]:
    """Create mock data with varying z-score conditions."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=40)

    # SPY: sharp recent drop (oversold → should trigger buy)
    spy_prices = np.concatenate([
        np.linspace(450, 460, 30),  # gradual rise
        np.linspace(460, 430, 10),  # sharp drop
    ])
    # TLT: steady (neutral)
    tlt_prices = np.linspace(100, 102, 40)
    # GLD: sharp rise (overbought)
    gld_prices = np.concatenate([
        np.linspace(180, 182, 30),  # steady
        np.linspace(182, 200, 10),  # sharp rise
    ])

    data = {}
    for sym, close in [("SPY", spy_prices), ("TLT", tlt_prices), ("GLD", gld_prices)]:
        data[sym] = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.uniform(1e7, 1e8, 40),
        }, index=dates)

    return data


class TestEtfMeanReversionStrategy:
    def test_name(self, strategy: EtfMeanReversionStrategy) -> None:
        assert strategy.name == "etf_mean_reversion"

    def test_universe(self, strategy: EtfMeanReversionStrategy) -> None:
        assert len(strategy.get_universe()) == 3

    def test_generate_signals(
        self, strategy: EtfMeanReversionStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(mock_data, datetime(2024, 2, 20))
        assert len(signals) == 3

        # Each signal should have z_score metadata
        for sig in signals:
            assert "z_score" in sig.metadata

    def test_insufficient_data(self, strategy: EtfMeanReversionStrategy) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = {"SPY": pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)}
        signals = strategy.generate_signals(data, datetime(2024, 1, 8))
        assert len(signals) == 0

    def test_backtest_weights_shape(
        self, strategy: EtfMeanReversionStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert weights.shape == (40, 3)

    def test_backtest_weights_bounded(
        self, strategy: EtfMeanReversionStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)
        assert (weights >= 0).all().all()
        assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_zscore_to_strength(self, strategy: EtfMeanReversionStrategy) -> None:
        # At threshold → strength 0
        assert strategy._zscore_to_strength(-1.5, is_entry=True) == pytest.approx(0.0)
        # At 2x threshold → strength 1
        assert strategy._zscore_to_strength(-3.0, is_entry=True) == pytest.approx(1.0)
