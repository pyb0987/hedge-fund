"""Backtest package — provides beta hedge overlay builder for portfolio backtests."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def build_beta_hedge_overlay(
    prices: pd.DataFrame,
) -> tuple[Callable[[pd.DataFrame], pd.DataFrame] | None, set[str]]:
    """Build a beta hedge weight overlay from settings.yaml config.

    Loads BetaHedgeConfig from settings.yaml. If enabled and the benchmark
    exists in prices, returns a closure that applies compute_hedge_weights.

    Args:
        prices: unified prices DataFrame (dates × symbols)

    Returns:
        (overlay_fn, extra_symbols) or (None, empty set) if disabled/unavailable
    """
    try:
        from hedgefund.config.loader import load_global_settings
        from hedgefund.risk.beta_hedge import compute_hedge_weights

        config_candidates = [
            Path("config"),
            Path(__file__).resolve().parents[2] / "config",
        ]
        settings = None
        for config_dir in config_candidates:
            if (config_dir / "settings.yaml").exists():
                settings = load_global_settings(config_dir)
                break

        if settings is None or not settings.beta_hedge.enabled:
            return None, set()

        bh = settings.beta_hedge
        extra = {bh.benchmark}

        if bh.benchmark not in prices.columns:
            logger.debug("Benchmark %s not in prices, skipping hedge", bh.benchmark)
            return None, extra

        def _overlay(combined_weights: pd.DataFrame) -> pd.DataFrame:
            aligned = prices.reindex(combined_weights.index).ffill()
            return compute_hedge_weights(
                combined_weights=combined_weights,
                prices=aligned,
                benchmark=bh.benchmark,
                window=bh.window,
                min_periods=bh.min_periods,
                max_hedge_ratio=bh.max_hedge_ratio,
            )

        return _overlay, extra

    except Exception:
        logger.debug("Beta hedge overlay build failed", exc_info=True)
        return None, set()
