"""YAML configuration loader with Pydantic validation."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from hedgefund.config.schemas import (
    CryptoMomentumConfig,
    DualMomentumConfig,
    EtfMeanReversionConfig,
    GlobalSettings,
)
from hedgefund.core.exceptions import ConfigError, ConfigFileNotFoundError

T = TypeVar("T", bound=BaseModel)

# Strategy name → config class mapping
STRATEGY_CONFIG_MAP: dict[str, type[BaseModel]] = {
    "crypto_momentum": CryptoMomentumConfig,
    "etf_mean_reversion": EtfMeanReversionConfig,
    "dual_momentum": DualMomentumConfig,
}

DEFAULT_CONFIG_DIR = Path("config")


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise ConfigFileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def load_global_settings(config_dir: Path = DEFAULT_CONFIG_DIR) -> GlobalSettings:
    """Load and validate global settings from settings.yaml."""
    path = config_dir / "settings.yaml"
    data = _load_yaml(path)
    try:
        return GlobalSettings(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid global settings: {e}") from e


def load_strategy_config(
    strategy_name: str,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> BaseModel:
    """Load and validate a strategy config by name.

    Args:
        strategy_name: one of 'crypto_momentum', 'etf_mean_reversion', 'dual_momentum'
        config_dir: base config directory

    Returns:
        Validated strategy config model
    """
    config_cls = STRATEGY_CONFIG_MAP.get(strategy_name)
    if config_cls is None:
        raise ConfigError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(STRATEGY_CONFIG_MAP.keys())}"
        )

    path = config_dir / "strategies" / f"{strategy_name}.yaml"
    data = _load_yaml(path)

    # Flatten nested 'parameters' key into top level for Pydantic
    if "parameters" in data:
        params = data.pop("parameters")
        data.update(params)

    # Flatten nested 'assets' key for dual_momentum (offensive/defensive → config fields)
    if "assets" in data:
        assets = data.pop("assets")
        if "offensive" in assets:
            data["offensive_assets"] = [a["symbol"] for a in assets["offensive"]]
        if "defensive" in assets:
            data["defensive_asset"] = assets["defensive"][0]["symbol"]

    try:
        return config_cls(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid strategy config '{strategy_name}': {e}") from e


def load_all_strategy_configs(
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> dict[str, BaseModel]:
    """Load all strategy configs."""
    result = {}
    for name in STRATEGY_CONFIG_MAP:
        path = config_dir / "strategies" / f"{name}.yaml"
        if path.exists():
            result[name] = load_strategy_config(name, config_dir)
    return result
