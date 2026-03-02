"""Tests for strategy registry."""

import pytest

from hedgefund.core.exceptions import StrategyNotFoundError
from hedgefund.strategies.registry import get_strategy, list_strategies, register_strategy
from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy
from hedgefund.config.schemas import CryptoMomentumConfig


class TestStrategyRegistry:
    def test_list_strategies(self) -> None:
        names = list_strategies()
        assert "crypto_momentum" in names

    def test_get_crypto_momentum(self) -> None:
        strategy = get_strategy("crypto_momentum")
        assert isinstance(strategy, CryptoMomentumStrategy)
        assert strategy.name == "crypto_momentum"

    def test_get_with_config(self) -> None:
        config = CryptoMomentumConfig(lookback_days=30, top_n=5)
        strategy = get_strategy("crypto_momentum", config=config)
        assert strategy.config.lookback_days == 30  # type: ignore[union-attr]

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(StrategyNotFoundError, match="not found"):
            get_strategy("nonexistent")

    def test_register_custom(self) -> None:
        register_strategy("custom_test", CryptoMomentumStrategy, CryptoMomentumConfig)
        assert "custom_test" in list_strategies()
        strategy = get_strategy("custom_test")
        assert isinstance(strategy, CryptoMomentumStrategy)
