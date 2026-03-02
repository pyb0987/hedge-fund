"""Tests for crypto momentum strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import CryptoMomentumConfig
from hedgefund.core.enums import SignalDirection
from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy


@pytest.fixture
def strategy() -> CryptoMomentumStrategy:
    config = CryptoMomentumConfig(lookback_days=20, top_n=2)
    universe = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"]
    return CryptoMomentumStrategy(config=config, universe=universe)


@pytest.fixture
def mock_data() -> dict[str, pd.DataFrame]:
    """Create mock OHLCV data with clear momentum ranking."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=30)

    data = {}
    # BTC: strong uptrend
    btc_close = 50_000_000 * np.cumprod(1 + np.full(30, 0.02))
    # ETH: moderate uptrend
    eth_close = 3_000_000 * np.cumprod(1 + np.full(30, 0.01))
    # XRP: flat
    xrp_close = 700 * np.cumprod(1 + np.full(30, 0.0))
    # SOL: downtrend
    sol_close = 100_000 * np.cumprod(1 + np.full(30, -0.01))

    for sym, close in [
        ("KRW-BTC", btc_close),
        ("KRW-ETH", eth_close),
        ("KRW-XRP", xrp_close),
        ("KRW-SOL", sol_close),
    ]:
        data[sym] = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1e9, 5e9, 30),
        }, index=dates)

    return data


class TestCryptoMomentumStrategy:
    def test_name(self, strategy: CryptoMomentumStrategy) -> None:
        assert strategy.name == "crypto_momentum"

    def test_universe(self, strategy: CryptoMomentumStrategy) -> None:
        assert len(strategy.get_universe()) == 4

    def test_set_universe(self, strategy: CryptoMomentumStrategy) -> None:
        strategy.set_universe(["KRW-BTC", "KRW-ETH"])
        assert len(strategy.get_universe()) == 2

    def test_generate_signals(
        self, strategy: CryptoMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(mock_data, datetime(2024, 2, 10))
        assert len(signals) == 4  # one per symbol

        # BTC and ETH should be LONG (top 2 momentum)
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        flat_signals = [s for s in signals if s.direction == SignalDirection.FLAT]

        assert len(long_signals) == 2
        assert len(flat_signals) == 2

        long_symbols = {s.symbol for s in long_signals}
        assert "KRW-BTC" in long_symbols
        assert "KRW-ETH" in long_symbols

    def test_no_signals_insufficient_data(
        self, strategy: CryptoMomentumStrategy
    ) -> None:
        # Only 5 days of data, but need 20 day lookback
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = {
            "KRW-BTC": pd.DataFrame(
                {"close": [100, 101, 102, 103, 104]},
                index=dates,
            )
        }
        signals = strategy.generate_signals(data, datetime(2024, 1, 8))
        assert len(signals) == 0

    def test_no_negative_momentum_long(self) -> None:
        """Negative momentum coins should not get LONG signal."""
        config = CryptoMomentumConfig(lookback_days=5, top_n=3)
        strategy = CryptoMomentumStrategy(config=config, universe=["A", "B"])

        dates = pd.bdate_range("2024-01-01", periods=10)
        data = {
            "A": pd.DataFrame(
                {"close": np.linspace(100, 80, 10)}, index=dates  # downtrend
            ),
            "B": pd.DataFrame(
                {"close": np.linspace(100, 90, 10)}, index=dates  # downtrend
            ),
        }
        signals = strategy.generate_signals(data, datetime(2024, 1, 15))
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        assert len(long_signals) == 0  # no longs for negative momentum

    def test_backtest_weights(
        self, strategy: CryptoMomentumStrategy, mock_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(mock_data.values())[0].index
        weights = strategy.backtest_weights(mock_data, dates)

        assert weights.shape[1] == 4  # 4 symbols
        assert len(weights) == len(dates)

        # After lookback period, weights should be non-zero
        post_lookback = weights.iloc[20:]
        assert post_lookback.sum(axis=1).max() > 0

        # Weights should sum to <= 1
        assert (post_lookback.sum(axis=1) <= 1.0 + 1e-9).all()
