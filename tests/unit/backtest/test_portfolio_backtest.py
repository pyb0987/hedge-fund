"""Tests for portfolio-level backtest (combined weight generator)."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.engine import BacktestConfig
from hedgefund.backtest.portfolio_backtest import (
    portfolio_weight_generator,
    run_portfolio_walk_forward,
)
from hedgefund.backtest.walk_forward import WalkForwardResult


def _make_prices(n_days: int = 500) -> dict[str, pd.DataFrame]:
    """Create mock OHLCV data for multiple symbols."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    data: dict[str, pd.DataFrame] = {}
    for sym, base, drift in [
        ("KRW-BTC", 50_000_000, 0.0005),
        ("KRW-ETH", 3_000_000, 0.0003),
        ("SPY", 400, 0.0004),
        ("QQQ", 350, 0.0003),
        ("TLT", 100, 0.0001),
        ("GLD", 180, 0.0002),
        ("BIL", 91, 0.00001),
    ]:
        close = base * np.cumprod(1 + rng.normal(drift, 0.015, n_days))
        data[sym] = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.uniform(1e6, 1e8, n_days),
        }, index=dates)

    return data


class TestPortfolioWeightGenerator:
    def test_returns_callable(self) -> None:
        """portfolio_weight_generator returns a callable."""
        strategies = _make_mock_strategies()
        allocation = {"strat_a": 0.3, "strat_b": 0.2}
        gen = portfolio_weight_generator(strategies, allocation)
        assert callable(gen)

    def test_combined_weights_shape(self) -> None:
        """Output has correct shape: all symbols x all dates."""
        strategies = _make_mock_strategies()
        allocation = {"strat_a": 0.3, "strat_b": 0.2}
        gen = portfolio_weight_generator(strategies, allocation)

        data = _make_prices(100)
        dates = pd.bdate_range("2020-01-01", periods=100)
        weights = gen(data, dates)

        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(dates)
        # Should have columns for symbols from both strategies
        assert len(weights.columns) > 0

    def test_allocation_scaling(self) -> None:
        """Weights are scaled by allocation."""
        strategies = _make_mock_strategies()
        # strat_a=100%, strat_b=0%
        gen_a = portfolio_weight_generator(strategies, {"strat_a": 1.0, "strat_b": 0.0})
        gen_b = portfolio_weight_generator(strategies, {"strat_a": 0.0, "strat_b": 1.0})

        data = _make_prices(100)
        dates = pd.bdate_range("2020-01-01", periods=100)

        weights_a = gen_a(data, dates)
        weights_b = gen_b(data, dates)

        # strat_a symbols should have non-zero weights in weights_a
        # strat_b symbols should have zero weights in weights_a
        strat_a_syms = strategies["strat_a"].get_universe()
        strat_b_syms = strategies["strat_b"].get_universe()

        for sym in strat_a_syms:
            if sym in weights_a.columns:
                assert weights_a[sym].abs().sum() > 0

        # strat_b has at least one symbol with non-zero weight
        strat_b_total = sum(
            weights_b[sym].abs().sum()
            for sym in strat_b_syms if sym in weights_b.columns
        )
        assert strat_b_total > 0

    def test_zero_allocation_excluded(self) -> None:
        """Strategy with 0 allocation produces zero weights."""
        strategies = _make_mock_strategies()
        gen = portfolio_weight_generator(strategies, {"strat_a": 0.0, "strat_b": 0.0})

        data = _make_prices(100)
        dates = pd.bdate_range("2020-01-01", periods=100)
        weights = gen(data, dates)

        assert (weights == 0.0).all().all()

    def test_weights_sum_bounded(self) -> None:
        """Combined weights per row should not exceed 1.0."""
        strategies = _make_mock_strategies()
        allocation = {"strat_a": 0.10, "strat_b": 0.20}
        gen = portfolio_weight_generator(strategies, allocation)

        data = _make_prices(200)
        dates = pd.bdate_range("2020-01-01", periods=200)
        weights = gen(data, dates)

        row_sums = weights.sum(axis=1)
        assert (row_sums <= 1.0 + 1e-9).all()


class TestRunPortfolioWalkForward:
    def test_runs_successfully(self) -> None:
        """Portfolio WF completes and returns WalkForwardResult."""
        strategies = _make_mock_strategies()
        allocation = {"strat_a": 0.20, "strat_b": 0.20}
        data = _make_prices(1500)

        result = run_portfolio_walk_forward(
            strategies=strategies,
            allocation=allocation,
            data=data,
            is_ratio=0.30,
            oos_ratio=0.10,
            min_windows=3,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.num_windows >= 3

    def test_efficiency_ratio_finite(self) -> None:
        strategies = _make_mock_strategies()
        allocation = {"strat_a": 0.20, "strat_b": 0.20}
        data = _make_prices(1500)

        result = run_portfolio_walk_forward(
            strategies=strategies,
            allocation=allocation,
            data=data,
            is_ratio=0.30,
            oos_ratio=0.10,
            min_windows=3,
        )

        assert np.isfinite(result.efficiency_ratio)


# --- Mock strategies for testing ---


class _MockStrategyA:
    """Mock strategy that equal-weights KRW-BTC and KRW-ETH."""

    @property
    def name(self) -> str:
        return "strat_a"

    def get_universe(self) -> list[str]:
        return ["KRW-BTC", "KRW-ETH"]

    def backtest_weights(
        self, data: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        symbols = [s for s in self.get_universe() if s in data]
        if not symbols:
            return pd.DataFrame(index=dates)
        w = 1.0 / len(symbols)
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        # Start after warmup
        weights.iloc[20:] = w
        return weights


class _MockStrategyB:
    """Mock strategy that weights SPY."""

    @property
    def name(self) -> str:
        return "strat_b"

    def get_universe(self) -> list[str]:
        return ["SPY", "TLT"]

    def backtest_weights(
        self, data: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        symbols = [s for s in self.get_universe() if s in data]
        if not symbols:
            return pd.DataFrame(index=dates)
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        weights.iloc[20:, 0] = 1.0  # SPY only
        return weights


def _make_mock_strategies() -> dict[str, object]:
    return {"strat_a": _MockStrategyA(), "strat_b": _MockStrategyB()}
