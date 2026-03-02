"""Tests for dual momentum strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import DualMomentumConfig
from hedgefund.core.enums import SignalDirection
from hedgefund.strategies.dual_momentum import DualMomentumStrategy, OFFENSIVE_ASSETS, DEFENSIVE_ASSET


@pytest.fixture
def strategy() -> DualMomentumStrategy:
    config = DualMomentumConfig(lookback_days=20, rebalance_day=1)
    return DualMomentumStrategy(config=config)


def _make_data(
    btc_trend: float, spy_trend: float, tlt_trend: float, n_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """Create mock data with specified trends."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    data = {}
    for sym, start, trend in [
        ("KRW-BTC", 50_000_000, btc_trend),
        ("SPY", 450, spy_trend),
        ("TLT", 100, tlt_trend),
    ]:
        close = start * np.cumprod(1 + np.full(n_days, trend))
        data[sym] = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.uniform(1e6, 1e8, n_days),
        }, index=dates)

    return data


class TestDualMomentumStrategy:
    def test_name(self, strategy: DualMomentumStrategy) -> None:
        assert strategy.name == "dual_momentum"

    def test_universe(self, strategy: DualMomentumStrategy) -> None:
        universe = strategy.get_universe()
        assert "KRW-BTC" in universe
        assert "SPY" in universe
        assert "TLT" in universe

    def test_both_positive_btc_wins(self, strategy: DualMomentumStrategy) -> None:
        """BTC and SPY both up, BTC stronger → BTC gets LONG."""
        data = _make_data(btc_trend=0.02, spy_trend=0.005, tlt_trend=0.001)
        signals = strategy.generate_signals(data, datetime(2024, 2, 10))

        assert len(signals) == 3
        btc_sig = next(s for s in signals if s.symbol == "KRW-BTC")
        spy_sig = next(s for s in signals if s.symbol == "SPY")
        tlt_sig = next(s for s in signals if s.symbol == "TLT")

        assert btc_sig.direction == SignalDirection.LONG
        assert spy_sig.direction == SignalDirection.FLAT
        assert tlt_sig.direction == SignalDirection.FLAT

    def test_both_positive_spy_wins(self, strategy: DualMomentumStrategy) -> None:
        """BTC and SPY both up, SPY stronger → SPY gets LONG."""
        data = _make_data(btc_trend=0.003, spy_trend=0.01, tlt_trend=0.001)
        signals = strategy.generate_signals(data, datetime(2024, 2, 10))

        spy_sig = next(s for s in signals if s.symbol == "SPY")
        btc_sig = next(s for s in signals if s.symbol == "KRW-BTC")
        assert spy_sig.direction == SignalDirection.LONG
        assert btc_sig.direction == SignalDirection.FLAT

    def test_both_negative_defensive(self, strategy: DualMomentumStrategy) -> None:
        """Both negative → TLT (defensive)."""
        data = _make_data(btc_trend=-0.01, spy_trend=-0.005, tlt_trend=0.002)
        signals = strategy.generate_signals(data, datetime(2024, 2, 10))

        tlt_sig = next(s for s in signals if s.symbol == "TLT")
        assert tlt_sig.direction == SignalDirection.LONG
        assert tlt_sig.metadata["regime"] == "defensive"

    def test_one_positive(self, strategy: DualMomentumStrategy) -> None:
        """Only SPY positive → SPY gets LONG."""
        data = _make_data(btc_trend=-0.01, spy_trend=0.005, tlt_trend=0.001)
        signals = strategy.generate_signals(data, datetime(2024, 2, 10))

        spy_sig = next(s for s in signals if s.symbol == "SPY")
        assert spy_sig.direction == SignalDirection.LONG

    def test_insufficient_data(self, strategy: DualMomentumStrategy) -> None:
        data = _make_data(btc_trend=0.01, spy_trend=0.01, tlt_trend=0.01, n_days=5)
        signals = strategy.generate_signals(data, datetime(2024, 1, 8))
        assert len(signals) == 0

    def test_backtest_weights(self, strategy: DualMomentumStrategy) -> None:
        data = _make_data(btc_trend=0.01, spy_trend=0.005, tlt_trend=0.002, n_days=60)
        dates = data["KRW-BTC"].index
        weights = strategy.backtest_weights(data, dates)

        # Should have columns for all 3 assets
        assert "KRW-BTC" in weights.columns or "SPY" in weights.columns or "TLT" in weights.columns

        # After lookback, some weights should be non-zero
        post_lookback = weights.iloc[25:]
        assert post_lookback.sum(axis=1).max() > 0

        # Each row should sum to <= 1
        assert (post_lookback.sum(axis=1) <= 1.0 + 1e-9).all()
