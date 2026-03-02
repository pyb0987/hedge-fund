"""Pydantic schemas for configuration validation."""

from pydantic import BaseModel, Field, field_validator

from hedgefund.core.enums import RebalanceFrequency


# --- Risk Limits ---


class RiskConfig(BaseModel):
    """Risk hard limits configuration."""

    max_portfolio_drawdown: float = Field(
        default=0.20, ge=0.01, le=0.50, description="Max portfolio DD before position reduction"
    )
    max_single_position: float = Field(
        default=0.15, ge=0.01, le=0.50, description="Max single position as fraction of portfolio"
    )
    max_strategy_allocation: float = Field(
        default=0.40, ge=0.10, le=1.0, description="Max allocation to any single strategy"
    )
    daily_var_95: float = Field(
        default=0.02, ge=0.005, le=0.10, description="95% daily VaR limit"
    )
    per_trade_stop_loss: float = Field(
        default=0.03, ge=0.01, le=0.10, description="Per-trade stop loss"
    )
    max_leverage: float = Field(default=1.0, ge=1.0, le=1.0, description="Max leverage (1.0 = no)")


# --- Allocation ---


class AllocationConfig(BaseModel):
    """Strategy allocation configuration."""

    crypto_momentum: float = Field(default=0.35, ge=0.0, le=1.0)
    etf_mean_reversion: float = Field(default=0.35, ge=0.0, le=1.0)
    dual_momentum: float = Field(default=0.30, ge=0.0, le=1.0)

    @field_validator("dual_momentum")
    @classmethod
    def allocations_sum_to_one(cls, v: float, info: object) -> float:
        data = info.data if hasattr(info, "data") else {}  # type: ignore[union-attr]
        total = data.get("crypto_momentum", 0.35) + data.get("etf_mean_reversion", 0.35) + v
        if abs(total - 1.0) > 0.01:
            msg = f"Allocations must sum to 1.0, got {total:.3f}"
            raise ValueError(msg)
        return v


# --- Schedule ---


class ScheduleConfig(BaseModel):
    """Scheduling configuration."""

    weekly_rebalance_day: str = Field(default="monday")
    monthly_rebalance_day: int = Field(default=1, ge=1, le=28)
    market_check_interval_minutes: int = Field(default=60, ge=1)


# --- Data ---


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    cache_ttl_seconds: int = Field(default=3600, ge=60)
    ohlcv_interval: str = Field(default="day")
    sqlite_path: str = Field(default="data/hedgefund.db")


# --- Monitoring ---


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    health_check_interval_minutes: int = Field(default=30, ge=1)


# --- Portfolio ---


class PortfolioConfig(BaseModel):
    """Portfolio-level configuration."""

    initial_capital: float = Field(ge=10000, description="Initial capital in KRW")
    base_currency: str = Field(default="KRW")


# --- Transaction Costs ---


class CostConfig(BaseModel):
    """Transaction cost model for a strategy."""

    commission_rate: float = Field(default=0.0, ge=0.0, le=0.05)
    slippage_rate: float = Field(default=0.0, ge=0.0, le=0.05)

    @property
    def total_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate


# --- Validation Criteria ---


class ValidationConfig(BaseModel):
    """Go/No-Go criteria for backtest validation."""

    min_sharpe: float = Field(default=0.8, ge=0.0)
    max_drawdown: float = Field(default=0.20, ge=0.01, le=1.0)
    min_profit_factor: float = Field(default=1.3, ge=1.0)


# --- Strategy Config ---


class CryptoMomentumConfig(BaseModel):
    """Strategy A: Crypto Momentum parameters."""

    name: str = "crypto_momentum"
    exchange: str = "upbit"
    market: str = "KRW"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    lookback_days: int = Field(default=20, ge=5, le=120)
    holding_days: int = Field(default=7, ge=1, le=30)
    top_n: int = Field(default=3, ge=1, le=10)
    universe_size: int = Field(default=15, ge=5, le=50)
    min_volume_krw: float = Field(default=1_000_000_000, ge=0)

    costs: CostConfig = CostConfig(commission_rate=0.0025, slippage_rate=0.0015)
    validation: ValidationConfig = ValidationConfig()


class EtfMeanReversionConfig(BaseModel):
    """Strategy B: US ETF Mean Reversion parameters."""

    name: str = "etf_mean_reversion"
    exchange: str = "alpaca"
    market: str = "US"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    lookback_days: int = Field(default=20, ge=5, le=120)
    z_entry_threshold: float = Field(default=-1.5, le=0)
    z_exit_threshold: float = Field(default=1.5, ge=0)
    holding_days: int = Field(default=7, ge=1, le=30)

    universe: list[str] = Field(default=["SPY", "QQQ", "TLT", "GLD", "IEF"])

    costs: CostConfig = CostConfig(commission_rate=0.0, slippage_rate=0.0005)
    validation: ValidationConfig = ValidationConfig()


class DualMomentumConfig(BaseModel):
    """Strategy C: Dual Momentum parameters."""

    name: str = "dual_momentum"
    exchange: str = "upbit+alpaca"
    market: str = "cross_asset"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY

    lookback_days: int = Field(default=60, ge=20, le=252)
    rebalance_day: int = Field(default=1, ge=1, le=28)

    costs: CostConfig = CostConfig(commission_rate=0.00125, slippage_rate=0.001)
    validation: ValidationConfig = ValidationConfig()


# --- Global Settings ---


class GlobalSettings(BaseModel):
    """Top-level configuration combining all sections."""

    portfolio: PortfolioConfig
    risk: RiskConfig = RiskConfig()
    allocation: AllocationConfig = AllocationConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    data: DataConfig = DataConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
