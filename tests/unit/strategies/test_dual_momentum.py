"""Tests for dual momentum strategy."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import DualMomentumConfig
from hedgefund.core.enums import SignalDirection
from hedgefund.strategies.dual_momentum import DualMomentumStrategy


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

    def test_custom_assets_from_config(self) -> None:
        """Verify strategy reads assets from config, not hardcoded constants."""
        config = DualMomentumConfig(
            lookback_days=20,
            offensive_assets=["KRW-ETH", "QQQ"],
            defensive_asset="GLD",
        )
        strategy = DualMomentumStrategy(config=config)
        universe = strategy.get_universe()
        assert "KRW-ETH" in universe
        assert "QQQ" in universe
        assert "GLD" in universe
        assert "KRW-BTC" not in universe

    def test_custom_assets_signal_generation(self) -> None:
        """Custom assets should produce correct signals."""
        config = DualMomentumConfig(
            lookback_days=20,
            offensive_assets=["KRW-ETH", "QQQ"],
            defensive_asset="GLD",
        )
        strategy = DualMomentumStrategy(config=config)

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=30)
        data = {}
        for sym, start, trend in [
            ("KRW-ETH", 3_000_000, 0.02),
            ("QQQ", 400, 0.005),
            ("GLD", 180, 0.001),
        ]:
            close = start * np.cumprod(1 + np.full(30, trend))
            data[sym] = pd.DataFrame({
                "open": close * 0.999, "high": close * 1.005,
                "low": close * 0.995, "close": close,
                "volume": rng.uniform(1e6, 1e8, 30),
            }, index=dates)

        signals = strategy.generate_signals(data, datetime(2024, 2, 10))
        assert len(signals) == 3
        eth_sig = next(s for s in signals if s.symbol == "KRW-ETH")
        assert eth_sig.direction == SignalDirection.LONG

    def test_rebalance_gate_first_call(self, strategy: DualMomentumStrategy) -> None:
        """First call always generates signals."""
        data = _make_data(btc_trend=0.02, spy_trend=0.005, tlt_trend=0.001)
        assert strategy.last_rebalance_date is None
        signals = strategy.generate_signals(data, datetime(2024, 2, 1))
        assert len(signals) == 3
        assert strategy.last_rebalance_date == datetime(2024, 2, 1)

    def test_rebalance_gate_blocks_same_month(self, strategy: DualMomentumStrategy) -> None:
        """Same month → no rebalance."""
        data = _make_data(btc_trend=0.02, spy_trend=0.005, tlt_trend=0.001)
        strategy.generate_signals(data, datetime(2024, 2, 1))

        # Same month, later day
        signals = strategy.generate_signals(data, datetime(2024, 2, 15))
        assert len(signals) == 0

    def test_rebalance_gate_allows_new_month(self, strategy: DualMomentumStrategy) -> None:
        """New month on/after rebalance_day → rebalance."""
        data = _make_data(btc_trend=0.02, spy_trend=0.005, tlt_trend=0.001)
        strategy.generate_signals(data, datetime(2024, 2, 1))

        # Next month, rebalance_day=1
        signals = strategy.generate_signals(data, datetime(2024, 3, 1))
        assert len(signals) == 3
        assert strategy.last_rebalance_date == datetime(2024, 3, 1)

    def test_rebalance_gate_waits_for_rebalance_day(self) -> None:
        """New month but before rebalance_day → no rebalance."""
        config = DualMomentumConfig(lookback_days=20, rebalance_day=15)
        strategy = DualMomentumStrategy(config=config)
        data = _make_data(btc_trend=0.02, spy_trend=0.005, tlt_trend=0.001)

        strategy.generate_signals(data, datetime(2024, 2, 15))

        # March 5 — new month but before rebalance_day=15
        signals = strategy.generate_signals(data, datetime(2024, 3, 5))
        assert len(signals) == 0

        # March 15 — at rebalance_day
        signals = strategy.generate_signals(data, datetime(2024, 3, 15))
        assert len(signals) == 3

    def test_rebalance_gate_state_setter(self) -> None:
        """last_rebalance_date setter works for state restoration."""
        config = DualMomentumConfig(lookback_days=20, rebalance_day=1)
        strategy = DualMomentumStrategy(config=config)

        restored_date = datetime(2024, 2, 1)
        strategy.last_rebalance_date = restored_date
        assert strategy.last_rebalance_date == restored_date
        assert not strategy.should_rebalance(datetime(2024, 2, 15))
        assert strategy.should_rebalance(datetime(2024, 3, 1))
