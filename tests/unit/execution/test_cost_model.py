"""Tests for execution cost model."""

import pytest

from hedgefund.core.enums import Exchange
from hedgefund.execution.cost_model import (
    EXCHANGE_COSTS,
    estimate_commission,
    estimate_slippage,
    estimate_total_cost,
    get_exchange_costs,
    is_above_minimum,
)


class TestExchangeCosts:
    def test_upbit_costs_defined(self) -> None:
        costs = get_exchange_costs(Exchange.UPBIT)
        assert costs.commission_rate == 0.0025
        assert costs.slippage_rate == 0.0015
        assert costs.min_order_value == 5_000

    def test_alpaca_costs_defined(self) -> None:
        costs = get_exchange_costs(Exchange.ALPACA)
        assert costs.commission_rate == 0.0
        assert costs.slippage_rate == 0.0005
        assert costs.min_order_value == 1.0

    def test_round_trip_cost(self) -> None:
        upbit = get_exchange_costs(Exchange.UPBIT)
        assert upbit.total_round_trip == pytest.approx(0.008)  # 0.4% × 2

    def test_one_way_cost(self) -> None:
        upbit = get_exchange_costs(Exchange.UPBIT)
        assert upbit.total_one_way == pytest.approx(0.004)


class TestEstimates:
    def test_upbit_commission(self) -> None:
        cost = estimate_commission(100_000, Exchange.UPBIT)
        assert cost == pytest.approx(250)  # 0.25% of 100K

    def test_alpaca_commission_zero(self) -> None:
        cost = estimate_commission(10_000, Exchange.ALPACA)
        assert cost == 0.0

    def test_slippage(self) -> None:
        cost = estimate_slippage(100_000, Exchange.UPBIT)
        assert cost == pytest.approx(150)  # 0.15% of 100K

    def test_total_cost(self) -> None:
        cost = estimate_total_cost(100_000, Exchange.UPBIT)
        assert cost == pytest.approx(400)  # 0.40% of 100K


class TestMinimumOrder:
    def test_above_minimum_upbit(self) -> None:
        assert is_above_minimum(10_000, Exchange.UPBIT) is True

    def test_below_minimum_upbit(self) -> None:
        assert is_above_minimum(3_000, Exchange.UPBIT) is False

    def test_above_minimum_alpaca(self) -> None:
        assert is_above_minimum(5.0, Exchange.ALPACA) is True

    def test_below_minimum_alpaca(self) -> None:
        assert is_above_minimum(0.5, Exchange.ALPACA) is False
