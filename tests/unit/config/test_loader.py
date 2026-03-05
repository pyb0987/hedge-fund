"""Tests for config loader."""

from pathlib import Path

import pytest

from hedgefund.config.loader import (
    load_all_strategy_configs,
    load_global_settings,
    load_strategy_config,
)
from hedgefund.config.schemas import CryptoMomentumConfig, DualMomentumConfig, GlobalSettings
from hedgefund.core.exceptions import ConfigError, ConfigFileNotFoundError

CONFIG_DIR = Path(__file__).parents[3] / "config"


class TestLoadGlobalSettings:
    def test_load_valid(self) -> None:
        settings = load_global_settings(CONFIG_DIR)
        assert isinstance(settings, GlobalSettings)
        assert settings.portfolio.initial_capital == 13_000_000
        assert settings.risk.max_portfolio_drawdown == 0.20
        assert settings.risk.max_symbol_exposure == 0.20

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigFileNotFoundError):
            load_global_settings(tmp_path)


class TestLoadStrategyConfig:
    def test_load_crypto_momentum(self) -> None:
        config = load_strategy_config("crypto_momentum", CONFIG_DIR)
        assert isinstance(config, CryptoMomentumConfig)
        assert config.lookback_days == 20
        assert config.top_n == 3

    def test_load_etf_mean_reversion(self) -> None:
        config = load_strategy_config("etf_mean_reversion", CONFIG_DIR)
        assert config.name == "etf_mean_reversion"  # type: ignore[union-attr]

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ConfigError, match="Unknown strategy"):
            load_strategy_config("nonexistent", CONFIG_DIR)


class TestLoadDualMomentumAssets:
    def test_loads_assets_from_yaml(self) -> None:
        """Verify YAML assets section is correctly flattened into config fields."""
        config = load_strategy_config("dual_momentum", CONFIG_DIR)
        assert isinstance(config, DualMomentumConfig)
        assert config.offensive_assets == ["KRW-BTC", "SPY"]
        assert config.defensive_assets == ["TLT", "GLD", "BIL"]
        assert config.defensive_weights == [0.4, 0.3, 0.3]

    def test_lookback_and_rebalance(self) -> None:
        config = load_strategy_config("dual_momentum", CONFIG_DIR)
        assert isinstance(config, DualMomentumConfig)
        assert config.lookback_days == 60
        assert config.rebalance_day == 1


class TestLoadAllStrategyConfigs:
    def test_loads_all(self) -> None:
        configs = load_all_strategy_configs(CONFIG_DIR)
        assert "crypto_momentum" in configs
        assert "etf_mean_reversion" in configs
        assert "dual_momentum" in configs
        assert "market_hedge" in configs
