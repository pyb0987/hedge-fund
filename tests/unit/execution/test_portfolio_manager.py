"""Tests for portfolio manager run_cycle orchestration."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hedgefund.config.schemas import AllocationConfig, RiskConfig
from hedgefund.core.enums import Exchange, OrderSide, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.execution.executors.paper_executor import PaperExecutor
from hedgefund.portfolio.manager import CycleResult, PortfolioManager
from hedgefund.risk.manager import RiskManager


class StubStrategy:
    """Minimal strategy for testing PortfolioManager."""

    def __init__(
        self,
        name: str,
        exchange: Exchange,
        signals: list[Signal] | None = None,
        universe: list[str] | None = None,
    ) -> None:
        self._name = name
        self._exchange = exchange
        self._signals = signals or []
        self._universe = universe or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def exchange(self) -> Exchange:
        return self._exchange

    def generate_signals(
        self, data: dict[str, pd.DataFrame], timestamp: datetime
    ) -> list[Signal]:
        return list(self._signals)

    def get_universe(self) -> list[str]:
        return list(self._universe)


def _make_ohlcv(symbol: str, price: float = 50_000_000) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=30)
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000,
        },
        index=dates,
    )


@pytest.fixture
def upbit_executor() -> PaperExecutor:
    return PaperExecutor(
        Exchange.UPBIT,
        initial_cash=500_000,
        price_feed={"KRW-BTC": 50_000_000, "KRW-ETH": 3_000_000},
    )


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(config=RiskConfig(), initial_capital=500_000)


@pytest.fixture
def allocation() -> AllocationConfig:
    return AllocationConfig(
        crypto_momentum=0.25,
        etf_mean_reversion=0.25,
        dual_momentum=0.20,
        treasury_park=0.20,
    )


class TestCycleResult:
    def test_num_trades_counts_successful(self) -> None:
        from hedgefund.core.models import Order
        from hedgefund.execution.protocols import ExecutionResult

        success_result = ExecutionResult(
            order=Order(
                strategy_name="test",
                symbol="KRW-BTC",
                exchange=Exchange.UPBIT,
                side=OrderSide.BUY,
                quantity=0.001,
                timestamp=datetime(2024, 6, 1),
            ),
            success=True,
            commission=100,
        )
        fail_result = ExecutionResult(
            order=Order(
                strategy_name="test",
                symbol="KRW-ETH",
                exchange=Exchange.UPBIT,
                side=OrderSide.BUY,
                quantity=0.01,
                timestamp=datetime(2024, 6, 1),
            ),
            success=False,
        )

        result = CycleResult(
            timestamp=datetime(2024, 6, 1),
            signals=(),
            executions=(success_result, fail_result),
            portfolio_value=1_000_000,
            cash_balance=500_000,
            risk_blocked=False,
        )
        assert result.num_trades == 1
        assert result.total_commission == 100

    def test_no_executions(self) -> None:
        result = CycleResult(
            timestamp=datetime(2024, 6, 1),
            signals=(),
            executions=(),
            portfolio_value=1_000_000,
            cash_balance=500_000,
            risk_blocked=False,
        )
        assert result.num_trades == 0
        assert result.total_commission == 0


class TestRunCycle:
    def test_cycle_with_long_signal(
        self,
        upbit_executor: PaperExecutor,
        risk_manager: RiskManager,
        allocation: AllocationConfig,
    ) -> None:
        """Long signal generates buy order and executes."""
        now = datetime(2024, 6, 1)
        signal = Signal(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            direction=SignalDirection.LONG,
            strength=1.0,
            timestamp=now,
        )

        strategy = StubStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            signals=[signal],
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: upbit_executor},
            risk_manager=risk_manager,
            allocation=allocation,
        )

        data = {"crypto_momentum": {"KRW-BTC": _make_ohlcv("KRW-BTC")}}
        result = pm.run_cycle(data, timestamp=now)

        assert not result.risk_blocked
        assert len(result.signals) == 1
        assert result.signals[0].direction == SignalDirection.LONG

    def test_cycle_no_data_returns_empty(
        self,
        upbit_executor: PaperExecutor,
        risk_manager: RiskManager,
        allocation: AllocationConfig,
    ) -> None:
        """No data for strategy → no signals → empty result."""
        strategy = StubStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            signals=[],
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: upbit_executor},
            risk_manager=risk_manager,
            allocation=allocation,
        )

        result = pm.run_cycle({}, timestamp=datetime(2024, 6, 1))

        assert len(result.signals) == 0
        assert len(result.executions) == 0
        assert not result.risk_blocked

    def test_cycle_drawdown_blocked(
        self,
        upbit_executor: PaperExecutor,
        allocation: AllocationConfig,
    ) -> None:
        """Max drawdown breached blocks all trades."""
        # Set peak high and current value low → DD > 20%
        rm = RiskManager(config=RiskConfig(), initial_capital=1_000_000)
        rm.update_peak(1_000_000)

        # Simulate drawdown by depleting executor cash
        depleted_executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=700_000,  # 30% below peak
            price_feed={"KRW-BTC": 50_000_000},
        )

        signal = Signal(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            direction=SignalDirection.LONG,
            strength=1.0,
            timestamp=datetime(2024, 6, 1),
        )
        strategy = StubStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            signals=[signal],
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: depleted_executor},
            risk_manager=rm,
            allocation=allocation,
        )

        data = {"crypto_momentum": {"KRW-BTC": _make_ohlcv("KRW-BTC")}}
        result = pm.run_cycle(data, timestamp=datetime(2024, 6, 1))

        assert result.risk_blocked
        assert result.risk_reason is not None
        assert "drawdown" in result.risk_reason.lower()
        assert len(result.executions) == 0

    def test_cycle_flat_signal_sells_position(
        self,
        risk_manager: RiskManager,
        allocation: AllocationConfig,
    ) -> None:
        """FLAT signal triggers sell of existing position."""
        now = datetime(2024, 6, 1)
        executor = PaperExecutor(
            Exchange.UPBIT,
            initial_cash=1_000_000,
            price_feed={"KRW-BTC": 50_000_000},
        )

        # First buy a position
        from hedgefund.core.models import Order

        buy = Order(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            side=OrderSide.BUY,
            quantity=0.002,
            timestamp=now,
        )
        executor.submit_order(buy)

        # Now run cycle with FLAT signal
        signal = Signal(
            strategy_name="crypto_momentum",
            symbol="KRW-BTC",
            exchange=Exchange.UPBIT,
            direction=SignalDirection.FLAT,
            strength=0.0,
            timestamp=now,
        )
        strategy = StubStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            signals=[signal],
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: executor},
            risk_manager=risk_manager,
            allocation=allocation,
        )

        data = {"crypto_momentum": {"KRW-BTC": _make_ohlcv("KRW-BTC")}}
        result = pm.run_cycle(data, timestamp=now)

        assert not result.risk_blocked
        assert len(result.signals) == 1

    def test_cycle_history_accumulates(
        self,
        upbit_executor: PaperExecutor,
        risk_manager: RiskManager,
        allocation: AllocationConfig,
    ) -> None:
        """Each cycle adds to history."""
        strategy = StubStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            signals=[],
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: upbit_executor},
            risk_manager=risk_manager,
            allocation=allocation,
        )

        pm.run_cycle({}, timestamp=datetime(2024, 6, 1))
        pm.run_cycle({}, timestamp=datetime(2024, 6, 2))

        assert len(pm.cycle_history) == 2

    def test_signal_generation_failure_doesnt_crash(
        self,
        upbit_executor: PaperExecutor,
        risk_manager: RiskManager,
        allocation: AllocationConfig,
    ) -> None:
        """Strategy exception is caught gracefully."""

        class FailingStrategy(StubStrategy):
            def generate_signals(
                self, data: dict[str, pd.DataFrame], timestamp: datetime
            ) -> list[Signal]:
                raise RuntimeError("Strategy crashed")

        strategy = FailingStrategy(
            name="crypto_momentum",
            exchange=Exchange.UPBIT,
            universe=["KRW-BTC"],
        )

        pm = PortfolioManager(
            strategies={"crypto_momentum": strategy},
            executors={Exchange.UPBIT: upbit_executor},
            risk_manager=risk_manager,
            allocation=allocation,
        )

        data = {"crypto_momentum": {"KRW-BTC": _make_ohlcv("KRW-BTC")}}
        # Should not raise
        result = pm.run_cycle(data, timestamp=datetime(2024, 6, 1))
        assert len(result.signals) == 0
