"""Strategy registry — load strategies by name."""

from typing import Any

from hedgefund.config.schemas import (
    CryptoMomentumConfig,
    DualMomentumConfig,
    EtfMeanReversionConfig,
    MarketHedgeConfig,
    SectorMomentumConfig,
    TreasuryParkConfig,
)
from hedgefund.core.exceptions import StrategyNotFoundError
from hedgefund.strategies.base import Strategy
from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy
from hedgefund.strategies.dual_momentum import DualMomentumStrategy
from hedgefund.strategies.etf_mean_reversion import EtfMeanReversionStrategy
from hedgefund.strategies.market_hedge import MarketHedgeStrategy
from hedgefund.strategies.sector_momentum import SectorMomentumStrategy
from hedgefund.strategies.treasury_park import TreasuryParkStrategy

# Registry mapping: strategy name → (strategy class, config class)
_REGISTRY: dict[str, tuple[type, type]] = {
    "crypto_momentum": (CryptoMomentumStrategy, CryptoMomentumConfig),
    "etf_mean_reversion": (EtfMeanReversionStrategy, EtfMeanReversionConfig),
    "dual_momentum": (DualMomentumStrategy, DualMomentumConfig),
    "market_hedge": (MarketHedgeStrategy, MarketHedgeConfig),
    "treasury_park": (TreasuryParkStrategy, TreasuryParkConfig),
    "sector_momentum": (SectorMomentumStrategy, SectorMomentumConfig),
}


def get_strategy(name: str, config: Any | None = None, **kwargs: Any) -> Strategy:
    """Instantiate a strategy by name.

    Args:
        name: strategy name (e.g. 'crypto_momentum')
        config: optional pre-built config object
        **kwargs: passed to strategy constructor

    Returns:
        Instantiated Strategy object
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise StrategyNotFoundError(
            f"Strategy '{name}' not found. Available: {list(_REGISTRY.keys())}"
        )

    strategy_cls, config_cls = entry

    if config is None:
        config = config_cls()

    return strategy_cls(config=config, **kwargs)


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY.keys())


def register_strategy(name: str, strategy_cls: type, config_cls: type) -> None:
    """Register a new strategy at runtime."""
    _REGISTRY[name] = (strategy_cls, config_cls)
