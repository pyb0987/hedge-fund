"""Pydantic schemas for configuration validation."""

from pydantic import BaseModel, Field, field_validator, model_validator

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
    max_symbol_exposure: float = Field(
        default=0.20, ge=0.05, le=0.50, description="Max cross-strategy symbol aggregate exposure"
    )
    daily_var_95: float = Field(default=0.02, ge=0.005, le=0.10, description="95% daily VaR limit")
    per_trade_stop_loss: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Per-trade stop loss"
    )
    max_leverage: float = Field(default=1.0, ge=1.0, le=2.0, description="Max gross leverage")


# --- Allocation ---


class AllocationConfig(BaseModel):
    """Strategy allocation configuration."""

    crypto_momentum: float = Field(default=0.10, ge=0.0, le=1.0)
    etf_mean_reversion: float = Field(default=0.20, ge=0.0, le=1.0)
    dual_momentum: float = Field(default=0.15, ge=0.0, le=1.0)
    market_hedge: float = Field(default=0.0, ge=0.0, le=1.0)
    treasury_park: float = Field(default=0.45, ge=0.0, le=1.0)
    sector_momentum: float = Field(default=0.0, ge=0.0, le=1.0)
    pairs_trading: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def allocations_sum_at_most_one(self) -> "AllocationConfig":
        total = (
            self.crypto_momentum
            + self.etf_mean_reversion
            + self.dual_momentum
            + self.market_hedge
            + self.treasury_park
            + self.sector_momentum
            + self.pairs_trading
        )
        if total > 1.01:
            msg = f"Allocations must sum to ≤ 1.0, got {total:.3f}"
            raise ValueError(msg)
        return self


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


class TelegramConfig(BaseModel):
    """Telegram notification configuration."""

    enabled: bool = Field(default=False)
    bot_token: str = Field(default="")
    chat_id: str = Field(default="")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    health_check_interval_minutes: int = Field(default=30, ge=1)
    telegram: TelegramConfig = TelegramConfig()


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

    costs: CostConfig = CostConfig(commission_rate=0.0005, slippage_rate=0.0015)
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
    drift_correction: bool = Field(default=False, description="Use drift-corrected z-score")
    drift_lookback_multiplier: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Multiplier for drift estimation window (drift_window = lookback * multiplier)",
    )

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

    offensive_assets: list[str] = Field(default=["KRW-BTC", "SPY"])
    defensive_assets: list[str] = Field(default=["TLT", "GLD", "BIL"])
    defensive_weights: list[float] = Field(default=[0.4, 0.3, 0.3])

    costs: CostConfig = CostConfig(commission_rate=0.00125, slippage_rate=0.001)
    validation: ValidationConfig = ValidationConfig()

    @field_validator("defensive_weights")
    @classmethod
    def defensive_weights_sum_to_one(cls, v: list[float]) -> list[float]:
        total = sum(v)
        if abs(total - 1.0) > 0.01:
            msg = f"Defensive weights must sum to 1.0, got {total:.3f}"
            raise ValueError(msg)
        return v


class SectorMomentumConfig(BaseModel):
    """Strategy: Sector ETF Momentum parameters."""

    name: str = "sector_momentum"
    exchange: str = "alpaca"
    market: str = "US"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    lookback_days: int = Field(default=20, ge=5, le=120)
    holding_days: int = Field(default=14, ge=7, le=60)
    top_n: int = Field(default=3, ge=1, le=5)

    universe: list[str] = Field(
        default=[
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "XLI",
            "XLC",
            "XLY",
            "XLP",
            "XLU",
            "XLRE",
            "XLB",
        ]
    )

    costs: CostConfig = CostConfig(commission_rate=0.0, slippage_rate=0.0005)
    validation: ValidationConfig = ValidationConfig()


class TreasuryParkConfig(BaseModel):
    """Strategy: Treasury Park — park idle cash in T-bill ETF."""

    name: str = "treasury_park"
    exchange: str = "alpaca"
    market: str = "US"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    symbol: str = Field(default="BIL", description="Short-term treasury ETF symbol")

    costs: CostConfig = CostConfig(commission_rate=0.0, slippage_rate=0.0001)
    validation: ValidationConfig = ValidationConfig()


class PairsTradingConfig(BaseModel):
    """Strategy: ETF Pairs Trading (market-neutral alpha)."""

    name: str = "pairs_trading"
    exchange: str = "alpaca"
    market: str = "US"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    lookback_days: int = Field(default=60, ge=20, le=252)
    holding_days: int = Field(default=5, ge=1, le=30)
    zscore_entry: float = Field(default=2.0, ge=1.0, le=4.0)
    zscore_exit: float = Field(default=0.5, ge=0.0, le=2.0)
    hedge_ratio_window: int = Field(default=60, ge=20, le=252)

    # Predefined cointegrated pairs: "SYMBOL_A,SYMBOL_B"
    pairs: list[str] = Field(
        default=["SPY,QQQ", "TLT,IEF"],
        description="Comma-separated pair strings, e.g. 'SPY,QQQ'",
    )

    costs: CostConfig = CostConfig(commission_rate=0.0, slippage_rate=0.001)
    validation: ValidationConfig = ValidationConfig()

    @field_validator("pairs")
    @classmethod
    def validate_pairs_format(cls, v: list[str]) -> list[str]:
        for pair_str in v:
            parts = pair_str.split(",")
            if len(parts) != 2 or not all(p.strip() for p in parts):
                msg = f"Each pair must be 'SYMBOL_A,SYMBOL_B', got '{pair_str}'"
                raise ValueError(msg)
        return v

    def parsed_pairs(self) -> list[tuple[str, str]]:
        """Return pairs as list of (symbol_a, symbol_b) tuples."""
        result = []
        for pair_str in self.pairs:
            a, b = pair_str.split(",")
            result.append((a.strip(), b.strip()))
        return result


class MarketHedgeConfig(BaseModel):
    """Strategy D: Market Hedge (Inverse ETF) parameters."""

    name: str = "market_hedge"
    exchange: str = "alpaca"
    market: str = "US"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY

    lookback_days: int = Field(default=60, ge=20, le=252)
    holding_days: int = Field(default=7, ge=1, le=30)

    # Index → Inverse ETF mapping
    index_assets: list[str] = Field(default=["SPY", "QQQ"])
    inverse_assets: list[str] = Field(default=["SH", "PSQ"])

    costs: CostConfig = CostConfig(commission_rate=0.0, slippage_rate=0.001)
    validation: ValidationConfig = ValidationConfig()


# --- Beta Hedge Overlay ---


class BetaHedgeConfig(BaseModel):
    """Portfolio-level beta hedging overlay configuration."""

    enabled: bool = Field(default=False)
    benchmark: str = Field(default="SPY")
    window: int = Field(default=60, ge=20, le=252)
    min_periods: int = Field(default=20, ge=10, le=60)
    max_hedge_ratio: float = Field(default=0.30, ge=0.0, le=1.0)


# --- Global Settings ---


class GlobalSettings(BaseModel):
    """Top-level configuration combining all sections."""

    portfolio: PortfolioConfig
    risk: RiskConfig = RiskConfig()
    allocation: AllocationConfig = AllocationConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    data: DataConfig = DataConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    beta_hedge: BetaHedgeConfig = BetaHedgeConfig()
