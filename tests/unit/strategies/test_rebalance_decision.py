"""Tests for get_rebalance_decision() across all strategies."""

from datetime import datetime

import pytest

from hedgefund.config.schemas import (
    CryptoMomentumConfig,
    DualMomentumConfig,
    EtfMeanReversionConfig,
    SectorMomentumConfig,
    TreasuryParkConfig,
)
from hedgefund.strategies.base import RebalanceDecision
from hedgefund.strategies.crypto_momentum import CryptoMomentumStrategy
from hedgefund.strategies.dual_momentum import DualMomentumStrategy
from hedgefund.strategies.etf_mean_reversion import EtfMeanReversionStrategy
from hedgefund.strategies.sector_momentum import SectorMomentumStrategy
from hedgefund.strategies.treasury_park import TreasuryParkStrategy


class TestCryptoMomentumRebalanceDecision:
    def test_first_run(self) -> None:
        s = CryptoMomentumStrategy(CryptoMomentumConfig())
        d = s.get_rebalance_decision(datetime(2026, 3, 1))
        assert d.should_rebalance is True
        assert d.reason == "first_run"
        assert d.gate_days == 7

    def test_too_early(self) -> None:
        s = CryptoMomentumStrategy(CryptoMomentumConfig())
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 3, 3))
        assert d.should_rebalance is False
        assert d.reason == "too_early"
        assert d.days_since_last == 2

    def test_holding_period_met(self) -> None:
        s = CryptoMomentumStrategy(CryptoMomentumConfig())
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 3, 8))
        assert d.should_rebalance is True
        assert d.reason == "holding_period_met"
        assert d.days_since_last == 7


class TestDualMomentumRebalanceDecision:
    def test_first_run(self) -> None:
        s = DualMomentumStrategy(DualMomentumConfig())
        d = s.get_rebalance_decision(datetime(2026, 3, 1))
        assert d.should_rebalance is True
        assert d.reason == "first_run"

    def test_same_month(self) -> None:
        s = DualMomentumStrategy(DualMomentumConfig())
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 3, 15))
        assert d.should_rebalance is False
        assert d.reason == "same_month"

    def test_before_rebalance_day(self) -> None:
        config = DualMomentumConfig(rebalance_day=15)
        s = DualMomentumStrategy(config)
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 4, 10))
        assert d.should_rebalance is False
        assert d.reason == "before_rebalance_day"

    def test_monthly_gate(self) -> None:
        s = DualMomentumStrategy(DualMomentumConfig())
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 4, 1))
        assert d.should_rebalance is True
        assert d.reason == "monthly_gate"


class TestSectorMomentumRebalanceDecision:
    def test_first_run(self) -> None:
        s = SectorMomentumStrategy(SectorMomentumConfig())
        d = s.get_rebalance_decision(datetime(2026, 3, 1))
        assert d.should_rebalance is True
        assert d.reason == "first_run"
        assert d.gate_days == 14

    def test_too_early(self) -> None:
        s = SectorMomentumStrategy(SectorMomentumConfig())
        s.last_rebalance_date = datetime(2026, 3, 1)
        d = s.get_rebalance_decision(datetime(2026, 3, 10))
        assert d.should_rebalance is False
        assert d.reason == "too_early"
        assert d.days_since_last == 9


class TestNoGateStrategies:
    def test_etf_mean_reversion_no_gate(self) -> None:
        s = EtfMeanReversionStrategy(EtfMeanReversionConfig())
        d = s.get_rebalance_decision(datetime(2026, 3, 1))
        assert d.should_rebalance is True
        assert d.reason == "no_gate"

    def test_treasury_park_no_gate(self) -> None:
        s = TreasuryParkStrategy(TreasuryParkConfig())
        d = s.get_rebalance_decision(datetime(2026, 3, 1))
        assert d.should_rebalance is True
        assert d.reason == "no_gate"
