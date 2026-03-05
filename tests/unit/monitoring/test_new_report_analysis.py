"""Tests for new report analysis functions (strategy perf, gates, holding periods)."""

from datetime import datetime

import pandas as pd
import pytest

from hedgefund.monitoring.paper_report import (
    RebalancingGateReport,
    StrategyPerformanceReport,
    analyze_holding_periods,
    analyze_rebalancing_gates,
    analyze_strategy_performance,
)


def _make_position_snapshots_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def _make_trades_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _make_decisions_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


class TestAnalyzeStrategyPerformance:
    def test_empty(self) -> None:
        result = analyze_strategy_performance(pd.DataFrame(), pd.DataFrame())
        assert result == ()

    def test_single_strategy(self) -> None:
        positions = _make_position_snapshots_df([
            {
                "timestamp": "2026-03-01", "symbol": "KRW-BTC",
                "exchange": "upbit", "strategy_name": "crypto_momentum",
                "quantity": 0.01, "avg_entry_price": 50e6,
                "market_price": 52e6, "market_value": 520_000,
                "unrealized_pnl": 20_000,
            },
        ])
        trades = _make_trades_df([
            {
                "timestamp": "2026-03-01", "symbol": "KRW-BTC",
                "strategy_name": "crypto_momentum", "commission": 1250,
                "slippage": 500, "side": "BUY", "quantity": 0.01,
                "price": 50e6, "pnl": None,
            },
        ])
        result = analyze_strategy_performance(positions, trades)
        assert len(result) == 1
        assert result[0].strategy_name == "crypto_momentum"
        assert result[0].total_value == 520_000
        assert result[0].total_pnl == 20_000
        assert result[0].trade_count == 1

    def test_multiple_strategies(self) -> None:
        positions = _make_position_snapshots_df([
            {
                "timestamp": "2026-03-01", "symbol": "KRW-BTC",
                "exchange": "upbit", "strategy_name": "crypto_momentum",
                "quantity": 0.01, "avg_entry_price": 50e6,
                "market_price": 52e6, "market_value": 520_000,
                "unrealized_pnl": 20_000,
            },
            {
                "timestamp": "2026-03-01", "symbol": "BIL",
                "exchange": "alpaca", "strategy_name": "treasury_park",
                "quantity": 100, "avg_entry_price": 91.5,
                "market_price": 91.6, "market_value": 9160,
                "unrealized_pnl": 10,
            },
        ])
        trades = pd.DataFrame()
        result = analyze_strategy_performance(positions, trades)
        assert len(result) == 2
        names = {r.strategy_name for r in result}
        assert names == {"crypto_momentum", "treasury_park"}


class TestAnalyzeRebalancingGates:
    def test_empty(self) -> None:
        assert analyze_rebalancing_gates(pd.DataFrame()) == ()

    def test_mixed_actions(self) -> None:
        decisions = _make_decisions_df([
            {
                "timestamp": "2026-03-01", "strategy_name": "crypto_momentum",
                "action": "rebalance", "reason": "first_run",
                "signals_generated": 5, "target_allocation": 0.10,
                "actual_allocation": 0.0, "dd_multiplier": 1.0,
            },
            {
                "timestamp": "2026-03-02", "strategy_name": "crypto_momentum",
                "action": "hold", "reason": "too_early",
                "signals_generated": 0, "target_allocation": 0.10,
                "actual_allocation": 0.0, "dd_multiplier": 1.0,
            },
            {
                "timestamp": "2026-03-03", "strategy_name": "crypto_momentum",
                "action": "hold", "reason": "too_early",
                "signals_generated": 0, "target_allocation": 0.10,
                "actual_allocation": 0.0, "dd_multiplier": 0.75,
            },
        ])
        result = analyze_rebalancing_gates(decisions)
        assert len(result) == 1
        gate = result[0]
        assert gate.strategy_name == "crypto_momentum"
        assert gate.rebalance_count == 1
        assert gate.hold_count == 2
        assert gate.skip_count == 0
        assert gate.avg_dd_multiplier == pytest.approx((1.0 + 1.0 + 0.75) / 3)


class TestAnalyzeHoldingPeriods:
    def test_empty(self) -> None:
        assert analyze_holding_periods(pd.DataFrame()) == ()

    def test_multi_day_position(self) -> None:
        positions = _make_position_snapshots_df([
            {
                "timestamp": "2026-03-01", "symbol": "KRW-BTC",
                "exchange": "upbit", "strategy_name": "crypto_momentum",
                "quantity": 0.01, "avg_entry_price": 50e6,
                "market_price": 50e6, "market_value": 500_000,
                "unrealized_pnl": 0,
            },
            {
                "timestamp": "2026-03-05", "symbol": "KRW-BTC",
                "exchange": "upbit", "strategy_name": "crypto_momentum",
                "quantity": 0.01, "avg_entry_price": 50e6,
                "market_price": 51e6, "market_value": 510_000,
                "unrealized_pnl": 10_000,
            },
        ])
        result = analyze_holding_periods(positions)
        assert len(result) == 1
        hp = result[0]
        assert hp.strategy_name == "crypto_momentum"
        assert hp.avg_days == 4  # Mar 1 → Mar 5
        assert hp.num_positions_tracked == 1

    def test_single_snapshot_position(self) -> None:
        """Single snapshot = minimum 1 day."""
        positions = _make_position_snapshots_df([
            {
                "timestamp": "2026-03-01", "symbol": "BIL",
                "exchange": "alpaca", "strategy_name": "treasury_park",
                "quantity": 100, "avg_entry_price": 91.5,
                "market_price": 91.5, "market_value": 9150,
                "unrealized_pnl": 0,
            },
        ])
        result = analyze_holding_periods(positions)
        assert len(result) == 1
        assert result[0].min_days == 1
