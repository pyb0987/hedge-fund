"""Tests for Kelly criterion position sizing."""

import numpy as np
import pytest

from hedgefund.core.kelly import (
    full_kelly_fraction,
    kelly_from_returns,
    quarter_kelly_fraction,
    volatility_adjusted_size,
)


class TestFullKelly:
    def test_coin_flip_even_odds(self) -> None:
        # 50% win rate, 1:1 payoff → Kelly = 0
        assert full_kelly_fraction(0.5, 1.0) == pytest.approx(0.0)

    def test_edge_positive(self) -> None:
        # 60% win rate, 1:1 payoff → Kelly = 0.2
        assert full_kelly_fraction(0.6, 1.0) == pytest.approx(0.2)

    def test_edge_negative(self) -> None:
        # 40% win rate, 1:1 payoff → Kelly = -0.2 (don't bet)
        assert full_kelly_fraction(0.4, 1.0) == pytest.approx(-0.2)

    def test_high_payoff(self) -> None:
        # 50% win rate, 2:1 payoff → Kelly = 0.25
        assert full_kelly_fraction(0.5, 2.0) == pytest.approx(0.25)

    def test_zero_win_loss_ratio(self) -> None:
        assert full_kelly_fraction(0.5, 0.0) == 0.0

    def test_negative_win_loss_ratio(self) -> None:
        assert full_kelly_fraction(0.5, -1.0) == 0.0


class TestQuarterKelly:
    def test_quarter_of_full(self) -> None:
        fk = full_kelly_fraction(0.6, 1.0)
        qk = quarter_kelly_fraction(0.6, 1.0)
        assert qk == pytest.approx(fk * 0.25)

    def test_negative_kelly_clamped_to_zero(self) -> None:
        assert quarter_kelly_fraction(0.4, 1.0) == 0.0

    def test_max_capped_at_025(self) -> None:
        # Very high edge
        qk = quarter_kelly_fraction(0.9, 5.0)
        assert qk <= 0.25


class TestKellyFromReturns:
    def test_insufficient_data(self) -> None:
        returns = np.array([0.01, -0.01, 0.02])
        assert kelly_from_returns(returns) == 0.0

    def test_positive_strategy(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 100)
        result = kelly_from_returns(returns)
        assert 0.0 <= result <= 0.25

    def test_all_winning(self) -> None:
        returns = np.array([0.01] * 20)
        # No losses → can't compute ratio
        assert kelly_from_returns(returns) == 0.0

    def test_all_losing(self) -> None:
        returns = np.array([-0.01] * 20)
        assert kelly_from_returns(returns) == 0.0

    def test_custom_fraction(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 100)
        half_kelly = kelly_from_returns(returns, fraction=0.5)
        quarter_kelly = kelly_from_returns(returns, fraction=0.25)
        assert half_kelly >= quarter_kelly


class TestVolatilityAdjustedSize:
    def test_high_vol_reduces_size(self) -> None:
        base = 0.10
        result = volatility_adjusted_size(base, current_volatility=0.30, target_volatility=0.15)
        assert result < base

    def test_low_vol_keeps_base(self) -> None:
        base = 0.10
        result = volatility_adjusted_size(base, current_volatility=0.10, target_volatility=0.15)
        # Low vol → scale would be >1, but capped at 1.0
        assert result == base

    def test_zero_vol(self) -> None:
        assert volatility_adjusted_size(0.10, 0.0) == 0.10

    def test_exact_target(self) -> None:
        base = 0.10
        result = volatility_adjusted_size(base, 0.15, 0.15)
        assert result == pytest.approx(base)
