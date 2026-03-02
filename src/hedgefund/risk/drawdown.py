"""Drawdown tracking and progressive position reduction.

드로다운이 깊어질수록 포지션을 점진적으로 줄여서 자본을 보존합니다.
20% DD 도달 시 전 포지션 50% 감소 (하드 리밋).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DrawdownState:
    """Current drawdown state — immutable snapshot."""

    current_value: float
    peak_value: float
    drawdown: float  # 0.0 ~ 1.0
    position_multiplier: float  # 0.0 ~ 1.0 (how much of target to hold)
    is_max_breached: bool


def compute_drawdown(current_value: float, peak_value: float) -> float:
    """Compute current drawdown as fraction from peak."""
    if peak_value <= 0:
        return 0.0
    return max(0.0, (peak_value - current_value) / peak_value)


def update_peak(current_value: float, previous_peak: float) -> float:
    """Update peak value (only increases)."""
    return max(current_value, previous_peak)


def position_multiplier(
    current_drawdown: float,
    max_drawdown: float = 0.20,
) -> float:
    """Progressive position reduction as drawdown deepens.

    | DD Level | Multiplier | Rationale |
    |----------|------------|-----------|
    | 0-5%     | 1.00       | Normal operations |
    | 5-10%    | 0.75       | Early warning — reduce exposure |
    | 10-15%   | 0.50       | Significant — cut in half |
    | 15-20%   | 0.25       | Severe — minimal exposure |
    | >20%     | 0.00       | STOP — max DD breached |
    """
    if current_drawdown <= 0.05:
        return 1.0
    elif current_drawdown <= 0.10:
        return 0.75
    elif current_drawdown <= 0.15:
        return 0.50
    elif current_drawdown <= max_drawdown:
        return 0.25
    else:
        return 0.0


def get_drawdown_state(
    current_value: float,
    peak_value: float,
    max_drawdown: float = 0.20,
) -> DrawdownState:
    """Compute full drawdown state snapshot."""
    new_peak = update_peak(current_value, peak_value)
    dd = compute_drawdown(current_value, new_peak)
    mult = position_multiplier(dd, max_drawdown)

    return DrawdownState(
        current_value=current_value,
        peak_value=new_peak,
        drawdown=dd,
        position_multiplier=mult,
        is_max_breached=dd > max_drawdown,
    )


def apply_drawdown_scaling(
    target_weights: NDArray[np.float64],
    dd_multiplier: float,
) -> NDArray[np.float64]:
    """Scale all target weights by drawdown multiplier.

    This reduces exposure proportionally across all positions.
    Freed capital goes to cash.
    """
    return target_weights * dd_multiplier
