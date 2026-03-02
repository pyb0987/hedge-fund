"""Position sizing — combines Kelly criterion with volatility scaling and risk limits.

파이프라인:
1. Kelly fraction → 기본 포지션 크기
2. Volatility scaling → 변동성에 반비례하여 조정
3. Drawdown multiplier → DD 레벨에 따라 감소
4. Hard limits → 최대 포지션/전략 배분 체크
"""

import numpy as np
from numpy.typing import NDArray

from hedgefund.config.schemas import RiskConfig
from hedgefund.core.kelly import kelly_from_returns, volatility_adjusted_size
from hedgefund.core.risk_metrics import annualized_volatility


def compute_position_size(
    returns: NDArray[np.float64],
    portfolio_value: float,
    drawdown_multiplier: float = 1.0,
    config: RiskConfig = RiskConfig(),
    target_volatility: float = 0.15,
    kelly_fraction: float = 0.25,
) -> float:
    """Full position sizing pipeline.

    Args:
        returns: historical return series for the asset/strategy
        portfolio_value: current portfolio value
        drawdown_multiplier: from drawdown control (0.0~1.0)
        config: risk configuration
        target_volatility: target annualized volatility
        kelly_fraction: Kelly multiplier (default quarter-Kelly)

    Returns:
        Position size in capital terms (KRW or USD)
    """
    if len(returns) < 20 or portfolio_value <= 0:
        return 0.0

    # Step 1: Kelly fraction
    base_fraction = kelly_from_returns(returns, fraction=kelly_fraction)

    # Step 2: Volatility scaling
    current_vol = annualized_volatility(returns)
    vol_adjusted = volatility_adjusted_size(base_fraction, current_vol, target_volatility)

    # Step 3: Drawdown scaling
    dd_scaled = vol_adjusted * drawdown_multiplier

    # Step 4: Clamp to hard limits
    max_position_fraction = config.max_single_position
    final_fraction = min(dd_scaled, max_position_fraction)

    return final_fraction * portfolio_value


def compute_strategy_weights(
    raw_weights: NDArray[np.float64],
    strategy_returns: NDArray[np.float64],
    drawdown_multiplier: float = 1.0,
    config: RiskConfig = RiskConfig(),
    target_volatility: float = 0.15,
) -> NDArray[np.float64]:
    """Adjust strategy-level weights with vol scaling and DD control.

    Args:
        raw_weights: target weights from strategy (sum <= 1.0)
        strategy_returns: historical returns of the strategy
        drawdown_multiplier: from drawdown control
        config: risk limits
        target_volatility: target vol

    Returns:
        Adjusted weights (may sum to less than original if scaled down)
    """
    if len(strategy_returns) < 20:
        return raw_weights

    current_vol = annualized_volatility(strategy_returns)
    if current_vol <= 0:
        vol_scale = 1.0
    else:
        vol_scale = min(1.0, target_volatility / current_vol)

    # Apply vol scale and DD multiplier
    adjusted = raw_weights * vol_scale * drawdown_multiplier

    # Clamp individual positions
    max_pos = config.max_single_position
    adjusted = np.minimum(adjusted, max_pos)

    # Clamp total strategy allocation
    total = float(np.sum(adjusted))
    if total > config.max_strategy_allocation:
        scale = config.max_strategy_allocation / total
        adjusted = adjusted * scale

    return adjusted
