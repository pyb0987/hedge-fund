"""Kelly Criterion position sizing.

Implements Quarter-Kelly for conservative position sizing.
Full Kelly is theoretically optimal but too aggressive in practice —
Quarter-Kelly sacrifices ~25% of growth for ~75% volatility reduction.
"""

import numpy as np
from numpy.typing import NDArray


def full_kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Compute full Kelly fraction.

    Kelly% = W - (1 - W) / R
    where W = win probability, R = win/loss ratio (avg_win / avg_loss)

    Args:
        win_rate: probability of winning (0.0 ~ 1.0)
        win_loss_ratio: average win / average loss (positive number)

    Returns:
        Optimal fraction of capital to risk. Can be negative (don't bet).
    """
    if win_loss_ratio <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / win_loss_ratio


def quarter_kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Quarter-Kelly: conservative 25% of full Kelly.

    Clamps result to [0.0, 0.25] — never risk more than 25% of capital
    and never go short via Kelly sizing.
    """
    fk = full_kelly_fraction(win_rate, win_loss_ratio)
    qk = fk * 0.25
    return max(0.0, min(qk, 0.25))


def kelly_from_returns(returns: NDArray[np.float64], fraction: float = 0.25) -> float:
    """Compute Kelly fraction directly from a return series.

    Args:
        returns: array of period returns
        fraction: Kelly multiplier (0.25 = quarter-kelly)

    Returns:
        Position size as fraction of capital [0.0, 0.25]
    """
    if len(returns) < 10:
        return 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    w = len(wins) / len(returns)
    avg_win = float(np.mean(wins))
    avg_loss = float(np.abs(np.mean(losses)))

    if avg_loss == 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    fk = full_kelly_fraction(w, win_loss_ratio)
    sized = fk * fraction
    return max(0.0, min(sized, 0.25))


def volatility_adjusted_size(
    base_size: float,
    current_volatility: float,
    target_volatility: float = 0.15,
) -> float:
    """Scale position size inversely with volatility.

    When volatility is high, reduce position size.
    When volatility is low, increase (up to base_size).

    Args:
        base_size: Kelly-derived position size
        current_volatility: current annualized volatility
        target_volatility: target annualized volatility (default 15%)

    Returns:
        Adjusted position size, clamped to [0.0, base_size]
    """
    if current_volatility <= 0:
        return base_size
    scale = target_volatility / current_volatility
    adjusted = base_size * min(scale, 1.0)  # never scale up beyond base
    return max(0.0, adjusted)
