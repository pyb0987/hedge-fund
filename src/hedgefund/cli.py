"""CLI entrypoint for the hedge fund system.

Usage:
    uv run python -m hedgefund paper-run
    uv run python -m hedgefund paper-run --dry-run
    uv run python -m hedgefund paper-status
    uv run python -m hedgefund paper-report
    uv run python -m hedgefund daily-run
    uv run python -m hedgefund install-scheduler
"""

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="hedgefund",
    help="Small-capital quantitative hedge fund CLI",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


@app.command()
def paper_run(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run without saving state",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging",
    ),
) -> None:
    """Execute a single paper trading cycle."""
    _setup_logging(verbose)

    from hedgefund.app import run_once

    try:
        result = run_once(config_dir=config_dir, dry_run=dry_run)
        typer.echo(f"\nPortfolio Value: {result.portfolio_value:,.0f}")
        typer.echo(f"Signals: {len(result.cycle_result.signals)}")
        typer.echo(f"Trades: {result.cycle_result.num_trades}")
        typer.echo(f"Commission: {result.cycle_result.total_commission:,.0f}")

        if result.cycle_result.risk_blocked:
            typer.echo(f"Risk Blocked: {result.cycle_result.risk_reason}")

        if result.collection.errors:
            typer.echo(f"\nWarnings ({len(result.collection.errors)}):")
            for err in result.collection.errors[:10]:
                typer.echo(f"  - {err}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def paper_status(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
) -> None:
    """Show current paper trading portfolio status."""
    _setup_logging()

    from hedgefund.execution.state import load_state
    from hedgefund.app import UPBIT_STATE, ALPACA_STATE

    upbit = load_state(UPBIT_STATE)
    alpaca = load_state(ALPACA_STATE)

    if upbit is None and alpaca is None:
        typer.echo("No paper trading state found. Run 'paper-run' first.")
        raise typer.Exit(code=0)

    total = 0.0

    if upbit is not None:
        info = upbit.get_account_info()
        total += info.cash_balance
        typer.echo(f"\n[UPBIT] Cash: {info.cash_balance:,.0f} KRW")
        if info.positions:
            for sym, qty in info.positions.items():
                typer.echo(f"  {sym}: {qty:.6f}")
        else:
            typer.echo("  No positions")

    if alpaca is not None:
        info = alpaca.get_account_info()
        total += info.cash_balance
        typer.echo(f"\n[ALPACA] Cash: {info.cash_balance:,.2f} USD")
        if info.positions:
            for sym, qty in info.positions.items():
                typer.echo(f"  {sym}: {qty:.6f}")
        else:
            typer.echo("  No positions")

    typer.echo(f"\nTotal Cash: {total:,.0f}")


@app.command()
def paper_reset(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation",
    ),
) -> None:
    """Reset paper trading state (delete saved positions and strategy state)."""
    from hedgefund.app import UPBIT_STATE, ALPACA_STATE, STRATEGY_STATE

    if not confirm:
        typer.confirm("This will delete all paper trading state. Continue?", abort=True)

    for path in [UPBIT_STATE, ALPACA_STATE, STRATEGY_STATE]:
        if path.exists():
            path.unlink()
            typer.echo(f"Deleted: {path}")

    typer.echo("Paper trading state reset.")


@app.command()
def paper_report(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging",
    ),
) -> None:
    """Generate paper trading validation report (Go/No-Go analysis)."""
    _setup_logging(verbose)

    from hedgefund.config.loader import load_global_settings
    from hedgefund.data.store import DataStore
    from hedgefund.monitoring.paper_report import (
        ReportThresholds,
        format_report,
        generate_report,
    )

    try:
        settings = load_global_settings(config_dir)
        store = DataStore(settings.data.sqlite_path)

        report = generate_report(store, ReportThresholds())
        typer.echo(format_report(report))

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def daily_run(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
    force: bool = typer.Option(
        False, "--force", help="Skip idempotency guard (run even if already ran today)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging",
    ),
) -> None:
    """Run daily paper trading cycle (paper-run + report + Telegram digest).

    Includes UTC-based idempotency guard — safe to call multiple times.
    """
    _setup_logging(verbose)

    from hedgefund.scheduler.runner import run_daily_cycle

    try:
        executed = run_daily_cycle(config_dir=config_dir, force=force)
        if executed:
            typer.echo("Daily cycle completed successfully.")
        else:
            typer.echo("Already ran today — skipped. Use --force to override.")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def install_scheduler(
    hour: int = typer.Option(21, help="Hour to run (local time, default: 21)"),
    minute: int = typer.Option(0, help="Minute to run (default: 0)"),
) -> None:
    """Install launchd plist for daily paper trading at specified time.

    Default: 21:00 KST daily.
    """
    _setup_logging()

    from hedgefund.scheduler.runner import install_plist

    try:
        plist_path = install_plist(hour=hour, minute=minute)
        typer.echo(f"Scheduler installed: {plist_path}")
        typer.echo(f"Daily run at {hour:02d}:{minute:02d} (local time)")
        typer.echo("Logs: ~/Library/Logs/hedgefund/")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def validate_wf(
    config_dir: Path = typer.Option(
        Path("config"), help="Config directory path",
    ),
    days: int = typer.Option(
        1260, help="Historical data days to fetch (default: 1260 = ~5 years)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging",
    ),
) -> None:
    """Run portfolio-level walk-forward validation with real data.

    Fetches historical data, runs WF for each strategy + combined portfolio,
    and computes Deflated Sharpe Ratio.
    """
    _setup_logging(verbose)

    from hedgefund.backtest.engine import BacktestConfig
    from hedgefund.backtest.deflated_sharpe import compute_return_moments, deflated_sharpe_ratio
    from hedgefund.backtest.portfolio_backtest import run_portfolio_walk_forward
    from hedgefund.backtest.walk_forward import run_walk_forward
    from hedgefund.config.loader import load_all_strategy_configs, load_global_settings
    from hedgefund.strategies.registry import get_strategy

    settings = load_global_settings(config_dir)
    configs = load_all_strategy_configs(config_dir)

    # Build strategies
    strategies = {}
    for name, cfg in configs.items():
        alloc = getattr(settings.allocation, name, 0.0)
        if alloc > 0:
            strategies[name] = get_strategy(name, config=cfg)

    if not strategies:
        typer.echo("No active strategies found (all allocations are 0).")
        raise typer.Exit(code=1)

    typer.echo(f"Active strategies: {list(strategies.keys())}")
    typer.echo(f"Fetching {days} days of historical data...")

    # Fetch data
    from hedgefund.data.protocols import DataProvider
    try:
        from hedgefund.data.providers.yfinance_provider import YFinanceProvider
        yf_provider = YFinanceProvider()
    except ImportError:
        typer.echo("yfinance not available", err=True)
        raise typer.Exit(code=1)

    all_symbols: set[str] = set()
    for strategy in strategies.values():
        all_symbols.update(strategy.get_universe())

    data: dict[str, "pd.DataFrame"] = {}
    import pandas as pd
    for sym in sorted(all_symbols):
        typer.echo(f"  Fetching {sym}...", nl=False)
        try:
            if sym.startswith("KRW-"):
                from hedgefund.data.providers.upbit_provider import UpbitProvider
                provider = UpbitProvider()
            else:
                provider = yf_provider
            df = provider.get_ohlcv(sym, interval="day", count=days)
            if df is not None and not df.empty:
                data[sym] = df
                typer.echo(f" {len(df)} bars")
            else:
                typer.echo(" NO DATA")
        except Exception as e:
            typer.echo(f" ERROR: {e}")

    if not data:
        typer.echo("No data fetched. Cannot proceed.", err=True)
        raise typer.Exit(code=1)

    # Run individual strategy WF
    typer.echo(f"\n{'=' * 60}")
    typer.echo(" Walk-Forward Validation")
    typer.echo(f"{'=' * 60}")

    num_trials = 88  # total parameter grid combinations

    for name, strategy in strategies.items():
        if not hasattr(strategy, "backtest_weights"):
            continue

        universe = strategy.get_universe()
        strategy_data = {s: data[s] for s in universe if s in data}
        if not strategy_data:
            typer.echo(f"\n[{name}] No data available — skipped")
            continue

        # Build prices DataFrame
        common = sorted(set.intersection(*(set(df.index) for df in strategy_data.values())))
        if len(common) < 100:
            typer.echo(f"\n[{name}] Insufficient common dates ({len(common)}) — skipped")
            continue

        prices_idx = pd.DatetimeIndex(common)
        prices = pd.DataFrame({s: strategy_data[s].loc[prices_idx, "close"] for s in strategy_data})

        cfg = configs.get(name)
        cost_config = BacktestConfig(
            commission_rate=cfg.costs.commission_rate if hasattr(cfg, "costs") else 0.0025,
            slippage_rate=cfg.costs.slippage_rate if hasattr(cfg, "costs") else 0.0015,
        )

        try:
            result = run_walk_forward(
                prices, strategy.backtest_weights,
                is_ratio=0.30, oos_ratio=0.10, config=cost_config, min_windows=3,
            )
            moments = compute_return_moments(result.windows[0].oos_result.returns.values)
            dsr = deflated_sharpe_ratio(
                result.oos_combined_sharpe, num_trials,
                len(result.windows[0].oos_result.returns),
                **moments,
            )
            typer.echo(f"\n[{name}] Windows: {result.num_windows}")
            typer.echo(f"  OOS Sharpe: {result.oos_combined_sharpe:.3f}")
            typer.echo(f"  OOS Max DD: {result.oos_combined_max_dd:.2%}")
            typer.echo(f"  Efficiency: {result.efficiency_ratio:.3f} (>0.5 = OK)")
            typer.echo(f"  Deflated Sharpe p: {dsr:.4f} (>0.95 = genuine)")
            typer.echo(f"  Valid: {'YES' if result.is_valid else 'NO'}")
        except (ValueError, Exception) as e:
            typer.echo(f"\n[{name}] WF failed: {e}")

    # Run portfolio-level WF
    typer.echo(f"\n{'=' * 60}")
    typer.echo(" Portfolio-Level Walk-Forward")
    typer.echo(f"{'=' * 60}")

    allocation = {
        name: getattr(settings.allocation, name, 0.0)
        for name in strategies
    }
    typer.echo(f"Allocation: {allocation}")

    try:
        portfolio_result = run_portfolio_walk_forward(
            strategies=strategies,
            allocation=allocation,
            data=data,
            is_ratio=0.30,
            oos_ratio=0.10,
            min_windows=3,
        )
        moments = compute_return_moments(
            portfolio_result.windows[0].oos_result.returns.values
        )
        dsr = deflated_sharpe_ratio(
            portfolio_result.oos_combined_sharpe, num_trials,
            len(portfolio_result.windows[0].oos_result.returns),
            **moments,
        )
        typer.echo(f"Windows: {portfolio_result.num_windows}")
        typer.echo(f"OOS Sharpe: {portfolio_result.oos_combined_sharpe:.3f}")
        typer.echo(f"OOS Max DD: {portfolio_result.oos_combined_max_dd:.2%}")
        typer.echo(f"OOS Return: {portfolio_result.oos_combined_return:.2%}")
        typer.echo(f"Efficiency: {portfolio_result.efficiency_ratio:.3f}")
        typer.echo(f"Deflated Sharpe p: {dsr:.4f}")
        typer.echo(f"Valid: {'YES' if portfolio_result.is_valid else 'NO'}")
    except (ValueError, Exception) as e:
        typer.echo(f"Portfolio WF failed: {e}")

    typer.echo(f"\n{'=' * 60}")


@app.command()
def optimize(
    strategy_name: str = typer.Argument(help="Strategy to optimize (e.g., etf_mean_reversion)"),
    days: int = typer.Option(
        1260, help="Historical data days to fetch (default: 1260 = ~5 years)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging",
    ),
) -> None:
    """Run parameter grid search for a strategy with Deflated Sharpe correction.

    Example: uv run python -m hedgefund optimize etf_mean_reversion
    """
    _setup_logging(verbose)

    from hedgefund.backtest.engine import BacktestConfig
    from hedgefund.backtest.optimizer import run_grid_search
    from hedgefund.config.loader import load_all_strategy_configs
    from hedgefund.config.schemas import (
        CryptoMomentumConfig,
        DualMomentumConfig,
        EtfMeanReversionConfig,
    )
    from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy
    from hedgefund.strategies.dual_momentum import DualMomentumStrategy
    from hedgefund.strategies.etf_mean_reversion import EtfMeanReversionStrategy

    # Strategy-specific grids and factories
    GRIDS: dict[str, dict] = {
        "etf_mean_reversion": {
            "factory": lambda p: EtfMeanReversionStrategy(EtfMeanReversionConfig(**p)),
            "grid": {
                "lookback_days": [10, 15, 20, 25, 30],
                "z_entry_threshold": [-1.0, -1.5, -2.0, -2.5],
                "z_exit_threshold": [1.0, 1.5, 2.0],
            },
            "symbols": ["SPY", "QQQ", "TLT", "GLD", "IEF"],
            "cost_config": BacktestConfig(commission_rate=0.0, slippage_rate=0.0005),
        },
        "crypto_momentum": {
            "factory": lambda p: CryptoMomentumStrategy(CryptoMomentumConfig(**p)),
            "grid": {
                "lookback_days": [10, 14, 20, 30],
                "top_n": [2, 3, 5],
                "holding_days": [7, 14, 21],
            },
            "symbols": ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"],
            "cost_config": BacktestConfig(commission_rate=0.0025, slippage_rate=0.0015),
        },
        "dual_momentum": {
            "factory": lambda p: DualMomentumStrategy(DualMomentumConfig(**p)),
            "grid": {
                "lookback_days": [40, 60, 90, 120],
            },
            "symbols": ["KRW-BTC", "SPY", "TLT", "GLD", "BIL"],
            "cost_config": BacktestConfig(commission_rate=0.00125, slippage_rate=0.001),
        },
    }

    if strategy_name not in GRIDS:
        typer.echo(f"Unknown strategy: {strategy_name}. Available: {list(GRIDS.keys())}")
        raise typer.Exit(code=1)

    spec = GRIDS[strategy_name]
    grid_size = 1
    for values in spec["grid"].values():
        grid_size *= len(values)
    typer.echo(f"Strategy: {strategy_name}")
    typer.echo(f"Parameter grid: {grid_size} combinations")
    typer.echo(f"Fetching {days} days of data...")

    # Fetch data
    import pandas as pd

    try:
        from hedgefund.data.providers.yfinance_provider import YFinanceProvider

        yf_provider = YFinanceProvider()
    except ImportError:
        typer.echo("yfinance not available", err=True)
        raise typer.Exit(code=1)

    data: dict[str, pd.DataFrame] = {}
    for sym in spec["symbols"]:
        typer.echo(f"  Fetching {sym}...", nl=False)
        try:
            if sym.startswith("KRW-"):
                from hedgefund.data.providers.upbit_provider import UpbitProvider

                provider = UpbitProvider()
            else:
                provider = yf_provider
            df = provider.get_ohlcv(sym, interval="day", count=days)
            if df is not None and not df.empty:
                data[sym] = df
                typer.echo(f" {len(df)} bars")
            else:
                typer.echo(" NO DATA")
        except Exception as e:
            typer.echo(f" ERROR: {e}")

    if not data:
        typer.echo("No data fetched.", err=True)
        raise typer.Exit(code=1)

    # Run grid search
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f" Grid Search: {strategy_name}")
    typer.echo(f"{'=' * 60}")

    result = run_grid_search(
        strategy_factory=spec["factory"],
        data=data,
        param_grid=spec["grid"],
        is_fraction=0.70,
        backtest_config=spec["cost_config"],
    )

    typer.echo(f"\nTrials: {result.num_trials}")
    typer.echo(f"Deflated Sharpe p-value: {result.deflated_sharpe_p:.4f}")
    typer.echo(f"  (p > 0.95 = likely genuine skill)")

    # Show top 5
    typer.echo(f"\n{'=' * 60}")
    typer.echo(" Top 5 Parameter Sets (by OOS Sharpe)")
    typer.echo(f"{'=' * 60}")

    for i, r in enumerate(result.all_results[:5]):
        go_nogo = "GO" if r.passes_go_nogo else "NO"
        typer.echo(f"\n#{i + 1} [{go_nogo}] {r.params}")
        typer.echo(f"  IS Sharpe:  {r.is_sharpe:.3f}")
        typer.echo(f"  OOS Sharpe: {r.oos_sharpe:.3f}")
        typer.echo(f"  OOS Max DD: {r.oos_max_dd:.2%}")
        typer.echo(f"  OOS PF:     {r.oos_profit_factor:.2f}")
        typer.echo(f"  Efficiency: {r.efficiency:.3f}")

    if result.best and result.best.passes_go_nogo and result.deflated_sharpe_p > 0.95:
        typer.echo(f"\n✓ Best params pass Go/No-Go + DSR: {result.best.params}")
    elif result.best and result.best.passes_go_nogo:
        typer.echo(f"\n⚠ Best params pass Go/No-Go but DSR={result.deflated_sharpe_p:.4f} < 0.95")
    else:
        typer.echo("\n✗ No parameter set passes Go/No-Go criteria")

    typer.echo(f"\n{'=' * 60}")


@app.command()
def uninstall_scheduler() -> None:
    """Uninstall launchd plist (stop daily scheduling)."""
    _setup_logging()

    from hedgefund.scheduler.runner import uninstall_plist

    if uninstall_plist():
        typer.echo("Scheduler uninstalled.")
    else:
        typer.echo("No scheduler found to uninstall.")
