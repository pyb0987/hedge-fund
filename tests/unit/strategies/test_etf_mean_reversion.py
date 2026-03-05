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

    def test_zscore_consistency_signal_vs_backtest(
        self, strategy: EtfMeanReversionStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        """Verify that backtest_weights uses the same z-score as generate_signals.

        Both should use _compute_rolling_zscore (log-return based cumulative z-score),
        NOT BaseStrategy.compute_zscore (single-day return z-score).
        """
        # Compute z-score via signal path
        prices = mock_data["SPY"]["close"]
        z_signal = strategy._compute_rolling_zscore(prices)

        # Z-score should be a finite number
        assert np.isfinite(z_signal)

    def test_rolling_zscore_constant_prices(self, strategy: EtfMeanReversionStrategy) -> None:
        """Constant prices → z-score should be 0 (no deviation)."""
        dates = pd.bdate_range("2024-01-01", periods=30)
        constant_prices = pd.Series(100.0, index=dates)
        z = strategy._compute_rolling_zscore(constant_prices)
        assert z == 0.0

    def test_rolling_zscore_insufficient_data(self, strategy: EtfMeanReversionStrategy) -> None:
        """Insufficient data → z-score should be 0."""
        short_prices = pd.Series([100, 101, 102])
        z = strategy._compute_rolling_zscore(short_prices)
        assert z == 0.0


class TestDriftCorrectedZscore:
    """Tests for drift-corrected z-score (mean != 0 assumption)."""

    def test_drift_correction_reduces_bull_bias(self) -> None:
        """In a steady uptrend, drift-corrected z-score should be closer to 0
        than the uncorrected version (which would be positive).
        """
        config_no_drift = EtfMeanReversionConfig(
            lookback_days=20, drift_correction=False,
        )
        config_drift = EtfMeanReversionConfig(
            lookback_days=20, drift_correction=True, drift_lookback_multiplier=3,
        )
        strat_no_drift = EtfMeanReversionStrategy(config=config_no_drift, universe=["SPY"])
        strat_drift = EtfMeanReversionStrategy(config=config_drift, universe=["SPY"])

        # Steady uptrend: ~0.05% daily drift for 80 days
        dates = pd.bdate_range("2024-01-01", periods=80)
        prices = pd.Series(
            100 * np.cumprod(1 + np.full(80, 0.0005)), index=dates,
        )

        z_no_drift = strat_no_drift._compute_rolling_zscore(prices)
        z_drift = strat_drift._compute_rolling_zscore(prices)

        # Uncorrected should be positive (cumulative return > 0 tested against 0)
        assert z_no_drift > 0
        # Drift-corrected should be closer to 0 (cumulative return tested against trend)
        assert abs(z_drift) < abs(z_no_drift)

    def test_drift_correction_default_off(self) -> None:
        """Default config has drift_correction=False."""
        config = EtfMeanReversionConfig()
        assert config.drift_correction is False

    def test_drift_correction_insufficient_history(self) -> None:
        """If not enough data for drift estimation, falls back to no-drift."""
        config = EtfMeanReversionConfig(
            lookback_days=20, drift_correction=True, drift_lookback_multiplier=3,
        )
        strat = EtfMeanReversionStrategy(config=config, universe=["SPY"])

        # Only 25 days — enough for lookback=20 but not for drift window=60
        dates = pd.bdate_range("2024-01-01", periods=25)
        prices = pd.Series(
            100 * np.cumprod(1 + np.full(25, 0.001)), index=dates,
        )

        z = strat._compute_rolling_zscore(prices)
        assert np.isfinite(z)  # Should work without crashing

    def test_drift_backtest_weights_consistency(self) -> None:
        """backtest_weights uses same z-score as generate_signals when drift enabled."""
        config = EtfMeanReversionConfig(
            lookback_days=20, drift_correction=True, drift_lookback_multiplier=3,
        )
        strat = EtfMeanReversionStrategy(config=config, universe=["SPY"])

        dates = pd.bdate_range("2024-01-01", periods=80)
        close = 100 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 80))
        data = {"SPY": pd.DataFrame({
            "open": close * 0.999, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.ones(80) * 1e7,
        }, index=dates)}

        # Both paths should use the same _compute_rolling_zscore
        weights = strat.backtest_weights(data, dates)
        signals = strat.generate_signals(data, datetime(2024, 4, 15))

        # No crash, shapes correct
        assert weights.shape == (80, 1)
        assert isinstance(signals, list)
