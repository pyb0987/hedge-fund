"""Shared test fixtures."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hedgefund.core.enums import AssetType, Exchange, OrderSide, OrderStatus, SignalDirection
from hedgefund.core.models import Order, Position, PortfolioSnapshot, Signal


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Sample daily returns for testing (252 trading days)."""
    rng = np.random.default_rng(42)
    # Simulate ~15% annual return, ~20% volatility
    daily_mu = 0.15 / 252
    daily_sigma = 0.20 / np.sqrt(252)
    return rng.normal(daily_mu, daily_sigma, 252)


@pytest.fixture
def losing_returns() -> np.ndarray:
    """Sample losing strategy returns."""
    rng = np.random.default_rng(99)
    daily_mu = -0.10 / 252
    daily_sigma = 0.30 / np.sqrt(252)
    return rng.normal(daily_mu, daily_sigma, 252)


@pytest.fixture
def flat_returns() -> np.ndarray:
    """Zero returns."""
    return np.zeros(100)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Sample multi-asset price data for backtest testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=252)

    data = {}
    for symbol in ["ASSET_A", "ASSET_B", "ASSET_C"]:
        daily_returns = rng.normal(0.0003, 0.015, 252)
        prices = 100 * np.cumprod(1 + daily_returns)
        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 60))
    high = close * (1 + rng.uniform(0, 0.02, 60))
    low = close * (1 - rng.uniform(0, 0.02, 60))
    open_ = close * (1 + rng.normal(0, 0.01, 60))
    volume = rng.uniform(1e6, 1e8, 60)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def sample_signal() -> Signal:
    """Sample Signal object."""
    return Signal(
        strategy_name="crypto_momentum",
        symbol="KRW-BTC",
        exchange=Exchange.UPBIT,
        direction=SignalDirection.LONG,
        strength=0.85,
        timestamp=datetime(2024, 6, 1),
        metadata={"momentum_score": 0.15},
    )


@pytest.fixture
def sample_position() -> Position:
    """Sample Position object."""
    return Position(
        symbol="KRW-BTC",
        exchange=Exchange.UPBIT,
        asset_type=AssetType.CRYPTO,
        quantity=0.01,
        avg_entry_price=50_000_000,
        current_price=52_000_000,
        strategy_name="crypto_momentum",
        opened_at=datetime(2024, 6, 1),
    )


@pytest.fixture
def sample_order() -> Order:
    """Sample Order object."""
    return Order(
        symbol="KRW-BTC",
        exchange=Exchange.UPBIT,
        side=OrderSide.BUY,
        quantity=0.01,
        strategy_name="crypto_momentum",
        timestamp=datetime(2024, 6, 1),
    )
