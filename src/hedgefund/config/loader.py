"""YAML configuration loader with Pydantic validation."""

import os
import re
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from hedgefund.config.schemas import (
    CryptoMomentumConfig,
    DualMomentumConfig,
    EtfMeanReversionConfig,
    GlobalSettings,
    MarketHedgeConfig,
    TreasuryParkConfig,
)
from hedgefund.core.exceptions import ConfigError, ConfigFileNotFoundError

T = TypeVar("T", bound=BaseModel)

# Strategy name → config class mapping
STRATEGY_CONFIG_MAP: dict[str, type[BaseModel]] = {
    "crypto_momentum": CryptoMomentumConfig,
    "etf_mean_reversion": EtfMeanReversionConfig,
    "dual_momentum": DualMomentumConfig,
    "market_hedge": MarketHedgeConfig,
    "treasury_park": TreasuryParkConfig,
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
    _resolve_env_vars(data)
    return data


def _resolve_env_vars(obj: object) -> None:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    env_pattern = re.compile(r"^\$\{(\w+)\}$")
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                match = env_pattern.match(value)
                if match:
                    env_name = match.group(1)
                    obj[key] = os.environ.get(env_name, "")
            elif isinstance(value, (dict, list)):
                _resolve_env_vars(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                match = env_pattern.match(item)
                if match:
                    env_name = match.group(1)
                    obj[i] = os.environ.get(env_name, "")
            elif isinstance(item, (dict, list)):
                _resolve_env_vars(item)


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
            data["defensive_assets"] = [a["symbol"] for a in assets["defensive"]]
            weights = [a.get("weight", 1.0 / len(assets["defensive"])) for a in assets["defensive"]]
            data["defensive_weights"] = weights

    # Flatten nested 'index_inverse_pairs' for market_hedge
    if "index_inverse_pairs" in data:
        pairs = data.pop("index_inverse_pairs")
        data["index_assets"] = [p["index"] for p in pairs]
        data["inverse_assets"] = [p["inverse"] for p in pairs]

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
