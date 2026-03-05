"""CLI entrypoint for the hedge fund system.

Usage:
    uv run python -m hedgefund paper-run
    uv run python -m hedgefund paper-run --dry-run
    uv run python -m hedgefund paper-status
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
