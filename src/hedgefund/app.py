"""Application wiring — assembles all components for paper trading.

Config → Strategies → Executors → RiskManager → PortfolioManager → run_cycle
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hedgefund.config.loader import load_all_strategy_configs, load_global_settings
from hedgefund.config.schemas import (
    CryptoMomentumConfig,
    DualMomentumConfig,
    EtfMeanReversionConfig,
    GlobalSettings,
)
from hedgefund.core.enums import Exchange
from hedgefund.data.collector import CollectionResult, DataCollector
from hedgefund.data.protocols import DataProvider
from hedgefund.data.store import DataStore
from hedgefund.execution.executors.paper_executor import PaperExecutor
from hedgefund.execution.protocols import Executor
from hedgefund.execution.state import (
    load_state,
    load_strategy_state,
    save_state,
    save_strategy_state,
)
from hedgefund.portfolio.manager import CycleResult, PortfolioManager
from hedgefund.risk.manager import RiskManager
from hedgefund.strategies.base import Strategy
from hedgefund.strategies.registry import get_strategy

logger = logging.getLogger(__name__)

STATE_DIR = Path("data/paper_state")
UPBIT_STATE = STATE_DIR / "upbit.json"
ALPACA_STATE = STATE_DIR / "alpaca.json"
STRATEGY_STATE = STATE_DIR / "strategies.json"


@dataclass(frozen=True)
class PaperRunResult:
    """Result of a paper trading batch run."""

    cycle_result: CycleResult
    collection: CollectionResult
    portfolio_value: float
    timestamp: datetime


def _create_strategies(
    config_dir: Path,
) -> dict[str, Strategy]:
    """Create strategy instances from YAML configs."""
    configs = load_all_strategy_configs(config_dir)
    strategies: dict[str, Strategy] = {}

    for name, config in configs.items():
        strategies[name] = get_strategy(name, config=config)

    return strategies


def _create_paper_executors(
    settings: GlobalSettings,
) -> dict[Exchange, PaperExecutor]:
    """Create paper executors, restoring state if available."""
    allocation = settings.allocation
    capital = settings.portfolio.initial_capital

    # Split capital by exchange: crypto → UPBIT, ETF → ALPACA
    upbit_share = allocation.crypto_momentum + allocation.dual_momentum * 0.5
    alpaca_share = allocation.etf_mean_reversion + allocation.dual_momentum * 0.5

    upbit_capital = capital * upbit_share
    alpaca_capital = capital * alpaca_share

    # Try to restore from saved state
    upbit_executor = load_state(UPBIT_STATE)
    if upbit_executor is None:
        upbit_executor = PaperExecutor(Exchange.UPBIT, upbit_capital)

    alpaca_executor = load_state(ALPACA_STATE)
    if alpaca_executor is None:
        alpaca_executor = PaperExecutor(Exchange.ALPACA, alpaca_capital)

    return {
        Exchange.UPBIT: upbit_executor,
        Exchange.ALPACA: alpaca_executor,
    }


def _create_providers() -> dict[str, DataProvider]:
    """Create data providers (lazy import to avoid API deps at import time)."""
    from hedgefund.data.providers.upbit_provider import UpbitProvider
    from hedgefund.data.providers.yfinance_provider import YFinanceProvider

    return {
        "upbit": UpbitProvider(),
        "yfinance": YFinanceProvider(),
    }


def _restore_strategy_state(strategies: dict[str, Strategy]) -> None:
    """Restore last_rebalance_date for strategies from saved state."""
    saved = load_strategy_state(STRATEGY_STATE)
    if not saved:
        return

    for name, last_date in saved.items():
        strategy = strategies.get(name)
        if strategy is not None and hasattr(strategy, "last_rebalance_date"):
            strategy.last_rebalance_date = last_date  # type: ignore[attr-defined]
            logger.info("Restored %s last_rebalance_date=%s", name, last_date)


def _setup_crypto_universe(
    strategies: dict[str, Strategy],
    providers: dict[str, DataProvider],
) -> None:
    """Set up crypto momentum universe from live volume data."""
    crypto_strategy = strategies.get("crypto_momentum")
    if crypto_strategy is None:
        return

    if not hasattr(crypto_strategy, "set_universe"):
        return

    upbit = providers.get("upbit")
    if upbit is None:
        return

    try:
        config = crypto_strategy.config  # type: ignore[attr-defined]
        top_symbols = upbit.get_top_volume_symbols(config.universe_size)  # type: ignore[attr-defined]
        crypto_strategy.set_universe(top_symbols)  # type: ignore[attr-defined]
        logger.info("Crypto universe set: %s", top_symbols[:5])
    except Exception:
        logger.exception("Failed to set crypto universe — using empty")


def run_once(
    config_dir: Path = Path("config"),
    dry_run: bool = False,
) -> PaperRunResult:
    """Execute a single paper trading cycle.

    1. Load config & create components
    2. Restore state
    3. Fetch market data
    4. Update executor prices
    5. Run portfolio cycle
    6. Persist all results (signals, trades, snapshots, state)

    Args:
        config_dir: path to config directory
        dry_run: if True, skip state saving

    Returns:
        PaperRunResult with cycle details
    """
    now = datetime.now()
    logger.info("=== Paper Trading Cycle Start: %s ===", now.strftime("%Y-%m-%d %H:%M"))

    # 1. Load config
    settings = load_global_settings(config_dir)

    # 2. Create components
    strategies = _create_strategies(config_dir)
    executors = _create_paper_executors(settings)
    providers = _create_providers()

    # 3. Restore strategy rebalancing state
    _restore_strategy_state(strategies)

    # 4. Set up crypto universe (requires live API call)
    _setup_crypto_universe(strategies, providers)

    # 5. Create risk manager & portfolio manager
    risk_manager = RiskManager(
        config=settings.risk,
        initial_capital=settings.portfolio.initial_capital,
    )

    portfolio_mgr = PortfolioManager(
        strategies=strategies,
        executors=executors,
        risk_manager=risk_manager,
        allocation=settings.allocation,
        risk_config=settings.risk,
    )

    # 6. Collect data
    collector = DataCollector(providers, strategies)
    collection = collector.collect_all()

    if collection.errors:
        for err in collection.errors:
            logger.warning("Data collection: %s", err)

    # 7. Update executor prices
    for exchange_type, executor in executors.items():
        exchange_prices = {
            sym: price for sym, price in collection.latest_prices.items()
            if _symbol_exchange(sym) == exchange_type
        }
        executor.set_prices(exchange_prices)

    # 8. Run cycle
    cycle_result = portfolio_mgr.run_cycle(collection.data, timestamp=now)

    # 9. Log results
    _log_cycle_result(cycle_result, executors)

    # 9b. Send Telegram notification
    _send_telegram_notification(settings, cycle_result, collection)

    # 10. Persist everything
    if not dry_run:
        save_state(executors[Exchange.UPBIT], UPBIT_STATE)
        save_state(executors[Exchange.ALPACA], ALPACA_STATE)
        save_strategy_state(strategies, STRATEGY_STATE)
        _persist_cycle_data(settings, cycle_result, executors, risk_manager)

    logger.info("=== Paper Trading Cycle Complete ===")

    return PaperRunResult(
        cycle_result=cycle_result,
        collection=collection,
        portfolio_value=cycle_result.portfolio_value,
        timestamp=now,
    )


def _send_telegram_notification(
    settings: GlobalSettings,
    result: CycleResult,
    collection: CollectionResult,
) -> None:
    """Send cycle summary via Telegram (no-op if disabled)."""
    tg_config = settings.monitoring.telegram
    if not tg_config.enabled or not tg_config.bot_token:
        return

    try:
        from hedgefund.monitoring.telegram import TelegramConfig, TelegramNotifier

        notifier = TelegramNotifier(TelegramConfig(
            bot_token=tg_config.bot_token,
            chat_id=tg_config.chat_id,
            enabled=True,
        ))
        notifier.notify_cycle(
            timestamp=result.timestamp,
            signals=len(result.signals),
            trades=result.num_trades,
            commission=result.total_commission,
            portfolio_value=result.portfolio_value,
            risk_blocked=result.risk_blocked,
            risk_reason=result.risk_reason,
            errors=tuple(collection.errors),
        )

        if result.risk_blocked:
            notifier.notify_risk_alert(
                reason=result.risk_reason or "Unknown",
                drawdown=0.0,
                timestamp=result.timestamp,
            )

        notifier.close()
    except Exception:
        logger.exception("Telegram notification failed — continuing")


def _symbol_exchange(symbol: str) -> Exchange:
    """Determine exchange for a symbol by naming convention."""
    if symbol.startswith("KRW-"):
        return Exchange.UPBIT
    return Exchange.ALPACA


def _log_cycle_result(
    result: CycleResult,
    executors: dict[Exchange, PaperExecutor],
) -> None:
    """Log human-readable cycle summary."""
    logger.info("Signals: %d, Trades: %d, Risk blocked: %s",
                len(result.signals), result.num_trades, result.risk_blocked)

    if result.risk_blocked:
        logger.warning("Risk reason: %s", result.risk_reason)

    for sig in result.signals:
        logger.info("  Signal: %s %s %s strength=%.2f",
                     sig.strategy_name, sig.symbol, sig.direction.value, sig.strength)

    for ex in result.executions:
        status = "OK" if ex.success else "FAIL"
        logger.info("  Exec: %s %s %s qty=%.4f price=%.2f cost=%.2f",
                     status, ex.order.symbol, ex.order.side.value,
                     ex.filled_quantity or 0, ex.filled_price or 0, ex.total_cost)

    for exchange_type, executor in executors.items():
        info = executor.get_account_info()
        logger.info("  %s: cash=%.0f, total=%.0f, positions=%s",
                     exchange_type.value, info.cash_balance, info.total_value,
                     dict(info.positions))


def _persist_cycle_data(
    settings: GlobalSettings,
    result: CycleResult,
    executors: dict[Exchange, PaperExecutor],
    risk_manager: RiskManager,
) -> None:
    """Persist signals, trades, and snapshot to SQLite for validation analysis."""
    try:
        store = DataStore(settings.data.sqlite_path)

        # A. Save signals (for Signal Fidelity validation)
        for signal in result.signals:
            store.save_signal(
                timestamp=signal.timestamp,
                strategy_name=signal.strategy_name,
                symbol=signal.symbol,
                exchange=signal.exchange.value,
                direction=signal.direction.value,
                strength=signal.strength,
                metadata=signal.metadata,
            )

        # B. Save trades (for Cost Accuracy validation)
        for execution in result.executions:
            if execution.success:
                store.save_trade(
                    symbol=execution.order.symbol,
                    exchange=execution.order.exchange.value,
                    side=execution.order.side.value,
                    quantity=execution.filled_quantity or 0,
                    price=execution.filled_price or 0,
                    commission=execution.commission,
                    slippage=execution.slippage,
                    strategy_name=execution.order.strategy_name,
                    timestamp=result.timestamp,
                    pnl=None,  # PnL computed from position tracking
                )

        # C. Save portfolio snapshot with real drawdown (for Performance validation)
        total_value = sum(
            e.get_account_info().total_value for e in executors.values()
        )
        total_cash = sum(
            e.get_account_info().cash_balance for e in executors.values()
        )

        dd_state = risk_manager.get_drawdown_state(total_value)

        store.save_snapshot(
            timestamp=result.timestamp,
            total_value=total_value,
            cash=total_cash,
            unrealized_pnl=total_value - total_cash,
            realized_pnl=result.total_commission,
            drawdown=dd_state.drawdown,
            peak_value=dd_state.peak_value,
        )

    except Exception:
        logger.exception("Failed to persist cycle data")
