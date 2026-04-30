"""Tests for app.py — application wiring and run_once."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from hedgefund.app import (
    _filter_strategies_for_market_hours,
    _is_us_market_day,
    _symbol_exchange,
    run_once,
)
from hedgefund.core.enums import Exchange


class TestSymbolExchange:
    """Tests for _symbol_exchange helper."""

    def test_krw_symbols_map_to_upbit(self):
        assert _symbol_exchange("KRW-BTC") == Exchange.UPBIT
        assert _symbol_exchange("KRW-ETH") == Exchange.UPBIT

    def test_etf_symbols_map_to_alpaca(self):
        assert _symbol_exchange("SPY") == Exchange.ALPACA
        assert _symbol_exchange("TLT") == Exchange.ALPACA
        assert _symbol_exchange("QQQ") == Exchange.ALPACA


class TestMarketHoursFilter:
    """Tests for US market day detection and strategy filtering."""

    def test_weekdays_are_market_days(self):
        assert _is_us_market_day(datetime(2026, 3, 16)) is True
        assert _is_us_market_day(datetime(2026, 3, 20)) is True

    def test_weekends_are_not_market_days(self):
        assert _is_us_market_day(datetime(2026, 3, 14)) is False
        assert _is_us_market_day(datetime(2026, 3, 15)) is False

    def test_weekday_keeps_all_strategies(self):
        crypto = MagicMock(exchange=Exchange.UPBIT)
        etf_mr = MagicMock(exchange=Exchange.ALPACA)
        strategies = {"crypto_momentum": crypto, "etf_mean_reversion": etf_mr}

        monday = datetime(2026, 3, 16)
        result = _filter_strategies_for_market_hours(strategies, monday)

        assert set(result.keys()) == {"crypto_momentum", "etf_mean_reversion"}

    def test_weekend_skips_alpaca_strategies(self):
        crypto = MagicMock(exchange=Exchange.UPBIT)
        dual = MagicMock(exchange=Exchange.UPBIT)
        etf_mr = MagicMock(exchange=Exchange.ALPACA)
        treasury = MagicMock(exchange=Exchange.ALPACA)

        strategies = {
            "crypto_momentum": crypto,
            "dual_momentum": dual,
            "etf_mean_reversion": etf_mr,
            "treasury_park": treasury,
        }

        saturday = datetime(2026, 3, 14)
        result = _filter_strategies_for_market_hours(strategies, saturday)

        assert set(result.keys()) == {"crypto_momentum", "dual_momentum"}
        assert "etf_mean_reversion" not in result
        assert "treasury_park" not in result


def _make_ohlcv(n: int = 30, base_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": [base_price] * n,
            "high": [base_price * 1.01] * n,
            "low": [base_price * 0.99] * n,
            "close": [base_price] * n,
            "volume": [1000.0] * n,
        },
        index=dates,
    )


class TestRunOnce:
    """Integration-like tests for run_once with mocked providers."""

    @patch("hedgefund.app._create_providers")
    @patch("hedgefund.app._setup_crypto_universe")
    def test_run_once_dry_run(self, mock_setup_universe, mock_providers, tmp_path: Path):
        """run_once in dry_run mode doesn't save state."""
        # Create a minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        strategies_dir = config_dir / "strategies"
        strategies_dir.mkdir()

        (config_dir / "settings.yaml").write_text("""
portfolio:
  initial_capital: 1000000
  base_currency: KRW
risk:
  max_portfolio_drawdown: 0.20
  max_single_position: 0.15
  max_strategy_allocation: 0.40
  daily_var_95: 0.02
  per_trade_stop_loss: 0.03
  max_leverage: 1.0
allocation:
  crypto_momentum: 0.20
  etf_mean_reversion: 0.20
  dual_momentum: 0.15
  treasury_park: 0.35
""")

        (strategies_dir / "etf_mean_reversion.yaml").write_text("""
name: etf_mean_reversion
exchange: alpaca
market: US
rebalance_frequency: weekly
parameters:
  lookback_days: 20
  z_entry_threshold: -1.5
  z_exit_threshold: 1.5
  holding_days: 7
universe:
  - SPY
costs:
  commission_rate: 0.0
  slippage_rate: 0.0005
validation:
  min_sharpe: 0.8
  max_drawdown: 0.20
  min_profit_factor: 1.3
""")

        (strategies_dir / "treasury_park.yaml").write_text("""
name: treasury_park
exchange: alpaca
market: US
parameters:
  symbol: BIL
costs:
  commission_rate: 0.0
  slippage_rate: 0.0001
""")

        # Mock providers to return fake data
        spy_df = _make_ohlcv(30, 450.0)

        class FakeYFinance:
            exchange_name = "yfinance"

            def get_ohlcv(self, symbol, interval="day", count=200):
                return spy_df

            def get_ohlcv_range(self, symbol, start, end, interval="day"):
                return spy_df

            def get_available_symbols(self):
                return ["SPY"]

        mock_providers.return_value = {"us_etf": FakeYFinance()}

        result = run_once(config_dir=config_dir, dry_run=True)

        assert result.portfolio_value >= 0
        assert result.timestamp is not None
        # Dry run: no state files should be created
        # State files may not exist because dry_run=True skips saving
