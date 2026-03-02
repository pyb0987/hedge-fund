"""Walk-Forward validation.

Splits data into rolling windows of In-Sample (IS) and Out-of-Sample (OOS).
Prevents overfitting by validating strategy on unseen data.

Window structure:
  |--- IS (60%) ---|--- OOS (20%) ---|--- Walk-Forward (20%) ---|
  Each window slides forward by OOS size.
"""

from dataclasses import dataclass

import pandas as pd

from hedgefund.backtest.engine import BacktestConfig, BacktestResult, run_backtest


@dataclass(frozen=True)
class WalkForwardWindow:
    """Single walk-forward window result."""

    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_result: BacktestResult
    oos_result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """Complete walk-forward analysis result."""

    windows: tuple[WalkForwardWindow, ...]
    oos_combined_sharpe: float
    oos_combined_return: float
    oos_combined_max_dd: float
    efficiency_ratio: float  # OOS Sharpe / IS Sharpe (< 1 means overfitting)

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def is_valid(self) -> bool:
        """Basic validity check: efficiency ratio > 0.5 and positive OOS Sharpe."""
        return self.efficiency_ratio > 0.5 and self.oos_combined_sharpe > 0


def run_walk_forward(
    prices: pd.DataFrame,
    weight_generator: callable,  # type: ignore[type-arg]
    is_ratio: float = 0.60,
    oos_ratio: float = 0.20,
    config: BacktestConfig = BacktestConfig(),
    min_windows: int = 3,
) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        prices: full period price data
        weight_generator: callable(prices_subset, dates) → weights DataFrame
            This is typically strategy.backtest_weights()
        is_ratio: fraction of total data for in-sample
        oos_ratio: fraction for out-of-sample
        config: backtest configuration
        min_windows: minimum number of windows to generate

    Returns:
        WalkForwardResult with all windows and combined metrics
    """
    n_total = len(prices)
    window_size = int(n_total * (is_ratio + oos_ratio))
    is_size = int(window_size * (is_ratio / (is_ratio + oos_ratio)))
    oos_size = window_size - is_size
    step_size = oos_size  # non-overlapping OOS

    if oos_size < 20:
        msg = f"OOS window too small ({oos_size} days). Need at least 20."
        raise ValueError(msg)

    windows: list[WalkForwardWindow] = []
    start = 0
    window_id = 0

    while start + window_size <= n_total:
        is_start_idx = start
        is_end_idx = start + is_size
        oos_start_idx = is_end_idx
        oos_end_idx = start + window_size

        is_prices = prices.iloc[is_start_idx:is_end_idx]
        oos_prices = prices.iloc[oos_start_idx:oos_end_idx]

        # Generate weights on IS data, evaluate on both IS and OOS
        is_weights = weight_generator(
            {col: is_prices[[col]] for col in is_prices.columns},
            is_prices.index,
        )
        oos_weights = weight_generator(
            # Use IS data for parameter fitting, but OOS prices for evaluation
            {col: prices.iloc[is_start_idx:oos_end_idx][[col]] for col in prices.columns},
            oos_prices.index,
        )

        is_result = run_backtest(is_prices, is_weights, config)
        oos_result = run_backtest(oos_prices, oos_weights, config)

        windows.append(WalkForwardWindow(
            window_id=window_id,
            is_start=pd.Timestamp(is_prices.index[0]),
            is_end=pd.Timestamp(is_prices.index[-1]),
            oos_start=pd.Timestamp(oos_prices.index[0]),
            oos_end=pd.Timestamp(oos_prices.index[-1]),
            is_result=is_result,
            oos_result=oos_result,
        ))

        start += step_size
        window_id += 1

    if len(windows) < min_windows:
        msg = (
            f"Only {len(windows)} windows generated, need at least {min_windows}. "
            f"Provide more data or reduce window ratios."
        )
        raise ValueError(msg)

    # Combine OOS results
    avg_is_sharpe = sum(w.is_result.sharpe_ratio for w in windows) / len(windows)
    avg_oos_sharpe = sum(w.oos_result.sharpe_ratio for w in windows) / len(windows)
    avg_oos_return = sum(w.oos_result.annualized_return for w in windows) / len(windows)
    max_oos_dd = max(w.oos_result.max_drawdown for w in windows)

    efficiency = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0.0

    return WalkForwardResult(
        windows=tuple(windows),
        oos_combined_sharpe=avg_oos_sharpe,
        oos_combined_return=avg_oos_return,
        oos_combined_max_dd=max_oos_dd,
        efficiency_ratio=efficiency,
    )
