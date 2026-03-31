"""Tests for beta hedge overlay."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.risk.beta_hedge import (
    BetaHedgeResult,
    compute_hedge_signal,
    compute_hedge_weights,
    rolling_beta,
)


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=120)


@pytest.fixture
def spy_returns(dates: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)


class TestRollingBeta:
    def test_known_beta(self, dates: pd.DatetimeIndex, spy_returns: pd.Series) -> None:
        """Portfolio = 0.5 * SPY + noise → beta ≈ 0.5."""
        rng = np.random.default_rng(99)
        noise = pd.Series(rng.normal(0, 0.002, len(dates)), index=dates)
        port = 0.5 * spy_returns + noise

        beta = rolling_beta(port, spy_returns, window=60, min_periods=20)
        # Last values should be approximately 0.5
        assert 0.3 < beta.iloc[-1] < 0.7

    def test_zero_beta_uncorrelated(self, dates: pd.DatetimeIndex) -> None:
        """Uncorrelated returns → beta ≈ 0."""
        rng = np.random.default_rng(42)
        port = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
        bench = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates + pd.Timedelta(days=1))
        # Different dates → no common index → beta = 0
        # Use same dates but independent random
        bench2 = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
        beta = rolling_beta(port, bench2, window=60, min_periods=20)
        assert abs(beta.iloc[-1]) < 0.3

    def test_insufficient_data_returns_zero(self) -> None:
        """Fewer than min_periods → beta = 0."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        port = pd.Series(np.random.randn(10), index=dates)
        bench = pd.Series(np.random.randn(10), index=dates)
        beta = rolling_beta(port, bench, window=60, min_periods=20)
        assert (beta == 0.0).all()


class TestComputeHedgeWeights:
    def test_adds_spy_hedge(self, dates: pd.DatetimeIndex) -> None:
        """Hedge weights add negative SPY when portfolio has positive beta."""
        rng = np.random.default_rng(42)
        spy_returns = rng.normal(0.001, 0.01, len(dates))
        spy_prices = pd.Series(100 * np.cumprod(1 + spy_returns), index=dates)
        # QQQ = 1.2 * SPY + noise → guaranteed positive beta
        qqq_returns = 1.2 * spy_returns + rng.normal(0, 0.003, len(dates))
        qqq_prices = pd.Series(200 * np.cumprod(1 + qqq_returns), index=dates)

        prices = pd.DataFrame({"SPY": spy_prices, "QQQ": qqq_prices})

        # Portfolio 100% in QQQ (positively correlated with SPY)
        weights = pd.DataFrame({"QQQ": np.ones(len(dates))}, index=dates)

        result = compute_hedge_weights(weights, prices, benchmark="SPY", window=60, min_periods=20)

        assert "SPY" in result.columns
        # Last rows should have negative SPY weight (hedge)
        last_spy = result["SPY"].iloc[-1]
        assert last_spy < 0

    def test_hedge_capped_at_max(self, dates: pd.DatetimeIndex) -> None:
        """Hedge weight should not exceed max_hedge_ratio."""
        rng = np.random.default_rng(42)
        spy_prices = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates))), index=dates
        )

        prices = pd.DataFrame({"SPY": spy_prices})

        # Portfolio = SPY itself (beta=1.0) — hedge should be capped
        weights = pd.DataFrame({"SPY": np.ones(len(dates))}, index=dates)

        result = compute_hedge_weights(weights, prices, benchmark="SPY", max_hedge_ratio=0.30)

        # SPY hedge should be >= -0.30 (capped) + existing 1.0 = >= 0.70
        assert result["SPY"].min() >= 1.0 - 0.30 - 0.01  # small tolerance

    def test_no_spy_in_prices_returns_unchanged(self, dates: pd.DatetimeIndex) -> None:
        """If benchmark not in prices, return weights unchanged."""
        prices = pd.DataFrame({"QQQ": np.ones(len(dates))}, index=dates)
        weights = pd.DataFrame({"QQQ": np.ones(len(dates))}, index=dates)

        result = compute_hedge_weights(weights, prices, benchmark="SPY")
        pd.testing.assert_frame_equal(result, weights)

    def test_hedge_preserves_existing_weights(self, dates: pd.DatetimeIndex) -> None:
        """Non-benchmark columns should be unchanged."""
        rng = np.random.default_rng(42)
        spy_p = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates))), index=dates)
        btc_p = pd.Series(50000 * np.cumprod(1 + rng.normal(0.002, 0.03, len(dates))), index=dates)

        prices = pd.DataFrame({"SPY": spy_p, "KRW-BTC": btc_p})
        weights = pd.DataFrame({"KRW-BTC": [0.5] * len(dates)}, index=dates)

        result = compute_hedge_weights(weights, prices)
        pd.testing.assert_series_equal(result["KRW-BTC"], weights["KRW-BTC"])


class TestComputeHedgeSignal:
    def test_known_beta(self) -> None:
        """Portfolio = SPY → beta ≈ 1.0."""
        rng = np.random.default_rng(42)
        spy = rng.normal(0.0004, 0.01, 100)
        result = compute_hedge_signal(spy, spy, window=60, min_periods=20)

        assert result.is_active
        assert 0.8 < result.beta < 1.2
        assert result.hedge_weight < 0

    def test_insufficient_data(self) -> None:
        """Too few data points → inactive."""
        result = compute_hedge_signal(
            np.array([0.01, -0.01]),
            np.array([0.01, -0.01]),
            window=60,
            min_periods=20,
        )
        assert not result.is_active
        assert result.hedge_weight == 0.0

    def test_cap_applied(self) -> None:
        """Hedge weight capped at max_hedge_ratio."""
        rng = np.random.default_rng(42)
        spy = rng.normal(0.001, 0.01, 100)
        result = compute_hedge_signal(spy, spy, max_hedge_ratio=0.10)

        assert result.hedge_weight >= -0.10

    def test_result_frozen(self) -> None:
        """BetaHedgeResult is immutable."""
        result = BetaHedgeResult(beta=0.5, hedge_weight=-0.3, data_points=60, is_active=True)
        with pytest.raises(AttributeError):
            result.beta = 0.0  # type: ignore[misc]
