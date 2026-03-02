"""Tests for walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.engine import BacktestConfig
from hedgefund.backtest.walk_forward import run_walk_forward, WalkForwardResult


def _simple_weight_generator(data: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Equal-weight generator for testing."""
    symbols = list(data.keys())
    return pd.DataFrame(1.0 / len(symbols), index=dates, columns=symbols)


@pytest.fixture
def long_prices() -> pd.DataFrame:
    """1500 trading days — enough for multiple walk-forward windows."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=1500)
    return pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 1500)),
        "B": 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, 1500)),
    }, index=dates)


class TestRunWalkForward:
    # Use smaller IS/OOS ratios so multiple windows fit within 1500 days
    IS_RATIO = 0.30
    OOS_RATIO = 0.10

    def test_returns_result(self, long_prices: pd.DataFrame) -> None:
        result = run_walk_forward(
            long_prices, _simple_weight_generator,
            is_ratio=self.IS_RATIO, oos_ratio=self.OOS_RATIO, min_windows=3,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.num_windows >= 3

    def test_efficiency_ratio_computed(self, long_prices: pd.DataFrame) -> None:
        result = run_walk_forward(
            long_prices, _simple_weight_generator,
            is_ratio=self.IS_RATIO, oos_ratio=self.OOS_RATIO, min_windows=3,
        )
        assert isinstance(result.efficiency_ratio, float)

    def test_oos_sharpe_is_finite(self, long_prices: pd.DataFrame) -> None:
        result = run_walk_forward(
            long_prices, _simple_weight_generator,
            is_ratio=self.IS_RATIO, oos_ratio=self.OOS_RATIO, min_windows=3,
        )
        assert np.isfinite(result.oos_combined_sharpe)

    def test_too_few_windows_raises(self) -> None:
        """120 days with is=0.6, oos=0.2 → window=96, oos=32, step=32.
        Fits at most 1 window (need 120 for 2nd). min_windows=3 → error.
        """
        dates = pd.bdate_range("2024-01-01", periods=120)
        prices = pd.DataFrame({
            "A": np.linspace(100, 130, 120),
        }, index=dates)
        with pytest.raises(ValueError, match="windows generated"):
            run_walk_forward(prices, _simple_weight_generator, min_windows=3)

    def test_oos_too_small_raises(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = pd.DataFrame({"A": np.linspace(100, 110, 30)}, index=dates)
        with pytest.raises(ValueError, match="OOS window too small"):
            run_walk_forward(prices, _simple_weight_generator, is_ratio=0.95, oos_ratio=0.01)

    def test_window_dates_non_overlapping(self, long_prices: pd.DataFrame) -> None:
        result = run_walk_forward(
            long_prices, _simple_weight_generator,
            is_ratio=self.IS_RATIO, oos_ratio=self.OOS_RATIO, min_windows=3,
        )
        for i in range(1, len(result.windows)):
            prev = result.windows[i - 1]
            curr = result.windows[i]
            assert curr.oos_start >= prev.oos_start

    def test_is_valid_property(self, long_prices: pd.DataFrame) -> None:
        result = run_walk_forward(
            long_prices, _simple_weight_generator,
            is_ratio=self.IS_RATIO, oos_ratio=self.OOS_RATIO, min_windows=3,
        )
        assert isinstance(result.is_valid, bool)
