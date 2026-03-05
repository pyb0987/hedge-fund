"""Tests for parameter optimizer (grid search + Deflated Sharpe)."""

import numpy as np
import pandas as pd
import pytest

from hedgefund.backtest.engine import BacktestConfig
from hedgefund.backtest.optimizer import (
    GridSearchResult,
    ParamResult,
    run_grid_search,
)
from hedgefund.config.schemas import EtfMeanReversionConfig
from hedgefund.strategies.etf_mean_reversion import EtfMeanReversionStrategy


def _make_etf_data(n_days: int = 500) -> dict[str, pd.DataFrame]:
    """Create mock OHLCV data for ETF universe."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = {}
    for sym, base, drift in [
        ("SPY", 400, 0.0003),
        ("QQQ", 350, 0.0004),
        ("TLT", 100, 0.0001),
        ("GLD", 180, 0.0002),
        ("IEF", 110, 0.00005),
    ]:
        close = base * np.cumprod(1 + rng.normal(drift, 0.012, n_days))
        data[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.uniform(1e6, 1e8, n_days),
            },
            index=dates,
        )
    return data


def _etf_factory(params: dict) -> EtfMeanReversionStrategy:
    config = EtfMeanReversionConfig(**params)
    return EtfMeanReversionStrategy(config=config)


class TestGridSearch:
    def test_returns_grid_search_result(self) -> None:
        data = _make_etf_data(500)
        param_grid = {
            "lookback_days": [15, 20],
            "z_entry_threshold": [-1.5, -2.0],
        }
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        assert isinstance(result, GridSearchResult)
        assert result.num_trials == 4  # 2 × 2

    def test_all_combos_evaluated(self) -> None:
        data = _make_etf_data(500)
        param_grid = {
            "lookback_days": [15, 20, 25],
            "z_entry_threshold": [-1.5],
        }
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        assert len(result.all_results) == 3

    def test_results_have_metrics(self) -> None:
        data = _make_etf_data(500)
        param_grid = {"lookback_days": [20]}
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        assert len(result.all_results) == 1
        r = result.all_results[0]
        assert isinstance(r, ParamResult)
        assert np.isfinite(r.is_sharpe)
        assert np.isfinite(r.oos_sharpe)
        assert np.isfinite(r.oos_max_dd)

    def test_efficiency_ratio(self) -> None:
        data = _make_etf_data(500)
        param_grid = {"lookback_days": [20]}
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        r = result.all_results[0]
        if r.is_sharpe != 0:
            assert np.isfinite(r.efficiency)

    def test_deflated_sharpe_computed(self) -> None:
        data = _make_etf_data(500)
        param_grid = {
            "lookback_days": [15, 20, 25],
            "z_entry_threshold": [-1.0, -1.5, -2.0],
        }
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        assert 0.0 <= result.deflated_sharpe_p <= 1.0
        assert result.num_trials == 9

    def test_best_result_is_highest_oos_sharpe(self) -> None:
        data = _make_etf_data(500)
        param_grid = {
            "lookback_days": [15, 20, 25],
            "z_entry_threshold": [-1.5, -2.0],
        }
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        if result.best is not None and len(result.all_results) > 1:
            best_oos = result.best.oos_sharpe
            for r in result.all_results:
                assert r.oos_sharpe <= best_oos + 1e-9

    def test_sorted_descending_by_oos_sharpe(self) -> None:
        data = _make_etf_data(500)
        param_grid = {
            "lookback_days": [15, 20, 25],
            "z_entry_threshold": [-1.0, -1.5],
        }
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
        )
        oos_sharpes = [r.oos_sharpe for r in result.all_results]
        assert oos_sharpes == sorted(oos_sharpes, reverse=True)

    def test_custom_backtest_config(self) -> None:
        data = _make_etf_data(300)
        param_grid = {"lookback_days": [20]}
        config = BacktestConfig(commission_rate=0.001, slippage_rate=0.0005)
        result = run_grid_search(
            strategy_factory=_etf_factory,
            data=data,
            param_grid=param_grid,
            is_fraction=0.70,
            backtest_config=config,
        )
        assert len(result.all_results) == 1
