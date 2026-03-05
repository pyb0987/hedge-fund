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
from hedgefund.strategies.base import RebalanceDecision, Strategy
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
    settings: GlobalSettings,
) -> dict[str, Strategy]:
    """Create strategy instances from YAML configs.

    Only loads strategies with allocation > 0 to avoid unnecessary API calls.
    """
    configs = load_all_strategy_configs(config_dir)
    strategies: dict[str, Strategy] = {}
    allocation = settings.allocation

    for name, config in configs.items():
        alloc = getattr(allocation, name, 0.0)
        if alloc > 0:
            strategies[name] = get_strategy(name, config=config)
        else:
            logger.debug("Skipping %s (allocation=0)", name)

    return strategies


def _create_paper_executors(
    settings: GlobalSettings,
) -> dict[Exchange, PaperExecutor]:
    """Create paper executors, restoring state if available."""
    allocation = settings.allocation
    capital = settings.portfolio.initial_capital

    # Split capital by exchange: crypto → UPBIT, ETF + hedge + treasury → ALPACA
    upbit_share = allocation.crypto_momentum + allocation.dual_momentum * 0.5
    alpaca_share = (
        allocation.etf_mean_reversion
        + allocation.dual_momentum * 0.5
        + allocation.market_hedge
        + allocation.treasury_park
        + allocation.sector_momentum
    )

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
    strategies = _create_strategies(config_dir, settings)
    executors = _create_paper_executors(settings)
    providers = _create_providers()

    # 3. Restore strategy rebalancing state
    _restore_strategy_state(strategies)

    # 4. Set up crypto universe (requires live API call)
    _setup_crypto_universe(strategies, providers)

    # 5. Create risk manager & portfolio manager
    # Use deployed capital (sum of executor capitals) as initial, not total initial_capital.
    # Otherwise unallocated cash buffer causes phantom drawdown on day 1.
    deployed_capital = sum(e.get_account_info().total_value for e in executors.values())
    risk_manager = RiskManager(
        config=settings.risk,
        initial_capital=deployed_capital,
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

    # 8. Capture rebalance decisions (before signals change strategy state)
    rebalance_decisions: dict[str, RebalanceDecision] = {}
    for name, strategy in strategies.items():
        if hasattr(strategy, "get_rebalance_decision"):
            rebalance_decisions[name] = strategy.get_rebalance_decision(now)  # type: ignore[attr-defined]

    # 9. Capture DD multiplier for audit
    dd_state = risk_manager.get_drawdown_state(deployed_capital)
    dd_multiplier = dd_state.position_multiplier

    # 10. Run cycle
    cycle_result = portfolio_mgr.run_cycle(collection.data, timestamp=now)

    # 11. Log results
    _log_cycle_result(cycle_result, executors)

    # 11b. Send Telegram notification
    _send_telegram_notification(settings, cycle_result, collection)

    # 12. Persist everything
    if not dry_run:
        save_state(executors[Exchange.UPBIT], UPBIT_STATE)
        save_state(executors[Exchange.ALPACA], ALPACA_STATE)
        save_strategy_state(strategies, STRATEGY_STATE)
        _persist_cycle_data(
            settings, cycle_result, executors, risk_manager,
            strategies=strategies, collection=collection,
            rebalance_decisions=rebalance_decisions,
            dd_multiplier=dd_multiplier,
        )

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
    strategies: dict[str, Strategy] | None = None,
    collection: CollectionResult | None = None,
    rebalance_decisions: dict[str, RebalanceDecision] | None = None,
    dd_multiplier: float = 1.0,
) -> None:
    """Persist signals, trades, positions, decisions, and OHLCV to SQLite."""
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

        # B. Save trades with PnL (for Cost Accuracy + Performance validation)
        cycle_realized_pnl = 0.0
        for execution in result.executions:
            if execution.success:
                trade_pnl = _find_trade_pnl(executors, execution)
                if trade_pnl is not None:
                    cycle_realized_pnl += trade_pnl

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
                    pnl=trade_pnl,
                )
            else:
                store.save_risk_event(
                    timestamp=result.timestamp,
                    event_type="execution_failed",
                    rule_name="order_rejected",
                    passed=False,
                    current_value=0.0,
                    limit_value=0.0,
                    message=execution.error_message or "Unknown execution error",
                    symbol=execution.order.symbol,
                    strategy_name=execution.order.strategy_name,
                )

        # C. Save portfolio snapshot with real drawdown
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
            realized_pnl=cycle_realized_pnl,
            drawdown=dd_state.drawdown,
            peak_value=dd_state.peak_value,
        )

        # D. Save risk events
        _persist_risk_events(store, result, risk_manager, total_value)

        # E. Save position snapshots (per-strategy attribution)
        if strategies is not None:
            _persist_position_snapshots(
                store, result.timestamp, strategies, executors,
            )

        # F. Save strategy decisions (rebalance gate + allocation audit)
        if strategies is not None and rebalance_decisions is not None:
            _persist_strategy_decisions(
                store, result, strategies, settings.allocation,
                rebalance_decisions, dd_multiplier, total_value,
            )

        # G. Save OHLCV market data for post-hoc verification
        if collection is not None:
            _persist_ohlcv(store, collection)

    except Exception:
        logger.exception("Failed to persist cycle data")


def _find_trade_pnl(
    executors: dict[Exchange, PaperExecutor],
    execution: "ExecutionResult",
) -> float | None:
    """Extract PnL from paper executor's trade records for a given execution."""
    from hedgefund.core.enums import OrderSide

    if execution.order.side != OrderSide.SELL:
        return None

    executor = executors.get(execution.order.exchange)
    if executor is None:
        return None

    # Search executor's trade history for matching trade
    for trade in reversed(executor._trades):
        if (trade.symbol == execution.order.symbol
                and trade.side == OrderSide.SELL
                and trade.pnl is not None):
            return trade.pnl

    return None


def _persist_risk_events(
    store: DataStore,
    result: "CycleResult",
    risk_manager: RiskManager,
    total_value: float,
) -> None:
    """Save risk system state as events for audit trail."""
    # Save drawdown state as risk event
    dd_state = risk_manager.get_drawdown_state(total_value)
    store.save_risk_event(
        timestamp=result.timestamp,
        event_type="drawdown_check",
        rule_name="portfolio_drawdown",
        passed=not dd_state.is_max_breached,
        current_value=dd_state.drawdown,
        limit_value=risk_manager.config.max_portfolio_drawdown,
        message=f"DD {dd_state.drawdown:.1%}, multiplier={dd_state.position_multiplier:.2f}",
    )

    # If risk blocked the entire cycle, record that
    if result.risk_blocked:
        store.save_risk_event(
            timestamp=result.timestamp,
            event_type="cycle_blocked",
            rule_name="max_drawdown",
            passed=False,
            current_value=dd_state.drawdown,
            limit_value=risk_manager.config.max_portfolio_drawdown,
            message=result.risk_reason or "Max drawdown breached",
        )


def _persist_position_snapshots(
    store: DataStore,
    timestamp: datetime,
    strategies: dict[str, Strategy],
    executors: dict[Exchange, PaperExecutor],
) -> None:
    """Save per-position snapshot with strategy attribution."""
    # Build symbol → strategy_name mapping from strategy universes
    symbol_strategy: dict[str, str] = {}
    for name, strategy in strategies.items():
        for sym in strategy.get_universe():
            # Last-write wins if symbols overlap (e.g., BIL in treasury + dual)
            symbol_strategy[sym] = name

    for executor in executors.values():
        for sym, pos in executor._positions.items():
            market_price = executor.get_current_price(sym)
            market_value = pos.quantity * market_price
            unrealized_pnl = (market_price - pos.avg_entry_price) * pos.quantity

            store.save_position_snapshot(
                timestamp=timestamp,
                symbol=sym,
                exchange=pos.exchange.value,
                strategy_name=symbol_strategy.get(sym, "unknown"),
                quantity=pos.quantity,
                avg_entry_price=pos.avg_entry_price,
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
            )


def _persist_strategy_decisions(
    store: DataStore,
    result: "CycleResult",
    strategies: dict[str, Strategy],
    allocation: "AllocationConfig",
    rebalance_decisions: dict[str, RebalanceDecision],
    dd_multiplier: float,
    total_value: float,
) -> None:
    """Save rebalancing gate decisions and allocation audit trail."""
    # Count signals per strategy
    signals_per_strategy: dict[str, int] = {}
    for signal in result.signals:
        signals_per_strategy[signal.strategy_name] = (
            signals_per_strategy.get(signal.strategy_name, 0) + 1
        )

    for name in strategies:
        decision = rebalance_decisions.get(name)
        if decision is None:
            continue

        target_alloc = getattr(allocation, name, 0.0)
        n_signals = signals_per_strategy.get(name, 0)

        action = "rebalance" if decision.should_rebalance else "hold"
        if n_signals == 0 and decision.should_rebalance:
            action = "skip_no_data"

        store.save_strategy_decision(
            timestamp=result.timestamp,
            strategy_name=name,
            action=action,
            reason=decision.reason,
            signals_generated=n_signals,
            target_allocation=target_alloc,
            actual_allocation=0.0,  # computed from position_snapshots in report
            dd_multiplier=dd_multiplier,
        )


def _persist_ohlcv(store: DataStore, collection: CollectionResult) -> None:
    """Save collected OHLCV data for post-hoc price verification."""
    for strategy_name, strategy_data in collection.data.items():
        for symbol, df in strategy_data.items():
            if df.empty:
                continue
            exchange = "upbit" if symbol.startswith("KRW-") else "alpaca"
            store.save_ohlcv(symbol, exchange, df)
