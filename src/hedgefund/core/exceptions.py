"""Custom exception hierarchy for the hedge fund system."""


class HedgeFundError(Exception):
    """Base exception for all hedge fund errors."""


# --- Configuration ---


class ConfigError(HedgeFundError):
    """Configuration loading or validation error."""


class ConfigFileNotFoundError(ConfigError):
    """Configuration file not found."""


# --- Data ---


class DataError(HedgeFundError):
    """Data fetching or processing error."""


class DataProviderError(DataError):
    """External data provider failure."""


class InsufficientDataError(DataError):
    """Not enough historical data for calculation."""


# --- Strategy ---


class StrategyError(HedgeFundError):
    """Strategy computation error."""


class StrategyNotFoundError(StrategyError):
    """Strategy not found in registry."""


# --- Risk ---


class RiskError(HedgeFundError):
    """Risk management violation."""


class DrawdownLimitError(RiskError):
    """Portfolio drawdown exceeded limit."""


class PositionLimitError(RiskError):
    """Position size exceeded limit."""


class OrderRejectedError(RiskError):
    """Order rejected by risk manager."""


# --- Execution ---


class ExecutionError(HedgeFundError):
    """Order execution failure."""


class InsufficientBalanceError(ExecutionError):
    """Not enough balance to execute order."""


class MinimumOrderError(ExecutionError):
    """Order below exchange minimum."""
