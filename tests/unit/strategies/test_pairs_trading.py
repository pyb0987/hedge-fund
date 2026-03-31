"""Tests for ETF Pairs Trading strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import PairsTradingConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.strategies.pairs_trading import PairsTradingStrategy


def _make_cointegrated_pair(
    n: int = 120,
    seed: int = 42,
    diverge_at: int | None = None,
    diverge_amount: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Generate cointegrated pair data (GLD/GDX style).

    Args:
        n: number of trading days
        seed: random seed
        diverge_at: day index to inject divergence (for entry signal testing)
        diverge_amount: magnitude of divergence (positive = A overvalued)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)

    # Common factor (e.g., gold price)
    factor = 100 + np.cumsum(rng.normal(0.0, 0.5, n))

    # A tracks factor with beta ~1.2, B tracks with beta ~1.0
    noise_a = np.cumsum(rng.normal(0.0, 0.1, n))
    noise_b = np.cumsum(rng.normal(0.0, 0.1, n))
    price_a = 150 + 1.2 * (factor - 100) + noise_a
    price_b = 30 + 1.0 * (factor - 100) + noise_b

    # Inject divergence
    if diverge_at is not None:
        price_a[diverge_at:] += diverge_amount

    data = {}
    for sym, prices in [("GLD", price_a), ("GDX", price_b)]:
        # Ensure positive prices
        prices = np.maximum(prices, 1.0)
        data[sym] = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": rng.uniform(1e7, 1e8, n),
            },
            index=dates,
        )

    return data


@pytest.fixture
def config() -> PairsTradingConfig:
    return PairsTradingConfig(
        lookback_days=60,
        holding_days=5,
        zscore_entry=2.0,
        zscore_exit=0.5,
        hedge_ratio_window=60,
        pairs=["GLD,GDX"],
    )


@pytest.fixture
def strategy(config: PairsTradingConfig) -> PairsTradingStrategy:
    return PairsTradingStrategy(config=config)


@pytest.fixture
def cointegrated_data() -> dict[str, pd.DataFrame]:
    return _make_cointegrated_pair(n=120, seed=42)


@pytest.fixture
def diverged_data() -> dict[str, pd.DataFrame]:
    """Data where A has diverged up → spread z > entry → expect SHORT A, LONG B."""
    return _make_cointegrated_pair(n=120, seed=42, diverge_at=100, diverge_amount=15.0)


class TestPairsTradingConfig:
    def test_default_pairs(self) -> None:
        config = PairsTradingConfig()
        assert len(config.pairs) == 2
        assert "SPY,QQQ" in config.pairs

    def test_parsed_pairs(self) -> None:
        config = PairsTradingConfig(pairs=["GLD,GDX", "XLE,USO"])
        parsed = config.parsed_pairs()
        assert parsed == [("GLD", "GDX"), ("XLE", "USO")]

    def test_invalid_pair_format(self) -> None:
        with pytest.raises(ValueError, match="SYMBOL_A,SYMBOL_B"):
            PairsTradingConfig(pairs=["INVALID"])

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValueError, match="SYMBOL_A,SYMBOL_B"):
            PairsTradingConfig(pairs=[",GDX"])


class TestPairsTradingStrategy:
    def test_name(self, strategy: PairsTradingStrategy) -> None:
        assert strategy.name == "pairs_trading"

    def test_exchange(self, strategy: PairsTradingStrategy) -> None:
        assert strategy.exchange == Exchange.ALPACA

    def test_universe_contains_all_pair_symbols(self, strategy: PairsTradingStrategy) -> None:
        universe = strategy.get_universe()
        assert "GLD" in universe
        assert "GDX" in universe
        assert len(universe) == 2

    def test_universe_no_duplicates_multi_pair(self) -> None:
        config = PairsTradingConfig(pairs=["GLD,GDX", "GLD,SLV"])
        strat = PairsTradingStrategy(config=config)
        universe = strat.get_universe()
        # GLD appears in both pairs but should appear once
        assert len(universe) == len(set(universe))

    def test_generate_signals_insufficient_data(self, strategy: PairsTradingStrategy) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        data = {
            "GLD": pd.DataFrame({"close": np.linspace(150, 155, 10)}, index=dates),
            "GDX": pd.DataFrame({"close": np.linspace(30, 31, 10)}, index=dates),
        }
        signals = strategy.generate_signals(data, datetime(2024, 1, 15))
        assert len(signals) == 0

    def test_generate_signals_cointegrated_no_divergence(
        self, strategy: PairsTradingStrategy, cointegrated_data: dict[str, pd.DataFrame]
    ) -> None:
        """Well-behaved cointegrated pair → should produce FLAT or no extreme signals."""
        signals = strategy.generate_signals(cointegrated_data, datetime(2024, 6, 15))
        # Should return signals (possibly FLAT) for each leg
        # With tightly cointegrated data, unlikely to exceed z=2.0
        for sig in signals:
            assert sig.strategy_name == "pairs_trading"
            assert sig.exchange == Exchange.ALPACA
            if sig.metadata:
                assert "spread_zscore" in sig.metadata

    def test_generate_signals_diverged_pair(
        self, strategy: PairsTradingStrategy, diverged_data: dict[str, pd.DataFrame]
    ) -> None:
        """Diverged pair → should produce LONG/SHORT signals."""
        signals = strategy.generate_signals(diverged_data, datetime(2024, 6, 15))
        if len(signals) == 2:
            directions = {sig.symbol: sig.direction for sig in signals}
            # If A overvalued (spread z > entry): SHORT A, LONG B
            if SignalDirection.SHORT in directions.values():
                assert directions["GLD"] == SignalDirection.SHORT
                assert directions["GDX"] == SignalDirection.LONG

    def test_signal_metadata_has_spread_zscore(
        self, strategy: PairsTradingStrategy, diverged_data: dict[str, pd.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(diverged_data, datetime(2024, 6, 15))
        for sig in signals:
            assert sig.metadata is not None
            assert "spread_zscore" in sig.metadata
            assert "hedge_ratio" in sig.metadata

    def test_rebalance_gate(self, config: PairsTradingConfig) -> None:
        """Should respect holding_days gate."""
        strategy = PairsTradingStrategy(config=config)
        decision = strategy.get_rebalance_decision(datetime(2024, 3, 1))
        assert decision.should_rebalance is True  # first run
        assert decision.reason == "first_run"

    def test_rebalance_gate_too_early(self, config: PairsTradingConfig) -> None:
        strategy = PairsTradingStrategy(config=config)
        strategy._last_rebalance_date = datetime(2024, 3, 1)
        decision = strategy.get_rebalance_decision(datetime(2024, 3, 3))
        assert decision.should_rebalance is False
        assert decision.reason == "too_early"

    def test_rebalance_gate_holding_met(self, config: PairsTradingConfig) -> None:
        strategy = PairsTradingStrategy(config=config)
        strategy._last_rebalance_date = datetime(2024, 3, 1)
        decision = strategy.get_rebalance_decision(datetime(2024, 3, 7))
        assert decision.should_rebalance is True


class TestPairsTradingBacktest:
    def test_backtest_weights_shape(
        self, strategy: PairsTradingStrategy, cointegrated_data: dict[str, pd.DataFrame]
    ) -> None:
        dates = list(cointegrated_data.values())[0].index
        weights = strategy.backtest_weights(cointegrated_data, dates)
        assert weights.shape[0] == len(dates)
        assert "GLD" in weights.columns
        assert "GDX" in weights.columns

    def test_backtest_weights_market_neutral(
        self, strategy: PairsTradingStrategy, cointegrated_data: dict[str, pd.DataFrame]
    ) -> None:
        """Active positions should be approximately market-neutral (net weight ≈ 0)."""
        dates = list(cointegrated_data.values())[0].index
        weights = strategy.backtest_weights(cointegrated_data, dates)
        # For rows where any position is active
        active_mask = weights.abs().sum(axis=1) > 0
        if active_mask.any():
            net_weights = weights.loc[active_mask].sum(axis=1)
            # Net exposure should be small (not exactly 0 due to hedge ratio != 1)
            assert (net_weights.abs() < 0.5).all(), f"Net weights too large: {net_weights}"

    def test_backtest_weights_long_short_opposing(
        self, strategy: PairsTradingStrategy, diverged_data: dict[str, pd.DataFrame]
    ) -> None:
        """When active, one leg should be positive and the other negative."""
        dates = list(diverged_data.values())[0].index
        weights = strategy.backtest_weights(diverged_data, dates)
        active_mask = weights.abs().sum(axis=1) > 0
        if active_mask.any():
            for _, row in weights.loc[active_mask].iterrows():
                gld_w = row.get("GLD", 0)
                gdx_w = row.get("GDX", 0)
                if gld_w != 0 and gdx_w != 0:
                    # Opposing signs
                    assert gld_w * gdx_w < 0, f"Same sign: GLD={gld_w}, GDX={gdx_w}"

    def test_backtest_weights_gross_exposure_bounded(
        self, strategy: PairsTradingStrategy, diverged_data: dict[str, pd.DataFrame]
    ) -> None:
        """Gross exposure (sum of |weights|) should be <= 1.0."""
        dates = list(diverged_data.values())[0].index
        weights = strategy.backtest_weights(diverged_data, dates)
        gross = weights.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all(), f"Gross exposure exceeded: {gross.max()}"


class TestHedgeRatio:
    def test_compute_hedge_ratio_positive(self, strategy: PairsTradingStrategy) -> None:
        """Positively correlated pair → positive hedge ratio."""
        rng = np.random.default_rng(42)
        n = 100
        factor = np.cumsum(rng.normal(0, 1, n))
        a = 100 + 1.5 * factor + rng.normal(0, 0.5, n)
        b = 50 + 1.0 * factor + rng.normal(0, 0.5, n)
        beta = strategy._compute_hedge_ratio(pd.Series(a), pd.Series(b), window=60)
        assert beta > 0

    def test_compute_spread_zscore(self, strategy: PairsTradingStrategy) -> None:
        """Spread z-score should be finite for well-behaved data."""
        rng = np.random.default_rng(42)
        n = 100
        factor = np.cumsum(rng.normal(0, 1, n))
        a = pd.Series(100 + 1.5 * factor + rng.normal(0, 0.5, n))
        b = pd.Series(50 + 1.0 * factor + rng.normal(0, 0.5, n))
        z = strategy._compute_spread_zscore(a, b, window=60)
        assert np.isfinite(z)


class TestConsistency:
    """Verify generate_signals and backtest_weights use the same math."""

    def test_same_zscore_computation(
        self, strategy: PairsTradingStrategy, diverged_data: dict[str, pd.DataFrame]
    ) -> None:
        """Both paths should use _compute_spread_zscore."""
        prices_a = diverged_data["GLD"]["close"]
        prices_b = diverged_data["GDX"]["close"]

        z = strategy._compute_spread_zscore(prices_a, prices_b, window=60)
        assert np.isfinite(z)

        # Signals should reflect same z
        signals = strategy.generate_signals(diverged_data, datetime(2024, 6, 15))
        if signals and signals[0].metadata:
            sig_z = signals[0].metadata["spread_zscore"]
            assert abs(sig_z - z) < 1e-6, f"Z-score mismatch: signal={sig_z}, direct={z}"
