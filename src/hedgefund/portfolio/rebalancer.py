"""Portfolio rebalancer — computes trades needed to reach target allocation.

리밸런싱 로직:
1. 현재 포지션 가중치 계산
2. 목표 가중치와의 차이(drift) 계산
3. Drift가 임계값 이상이면 리밸런싱 거래 생성
4. 최소 거래 크기 필터 적용 (거래 비용 > 기대 이익이면 무시)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RebalanceTrade:
    """Single rebalancing trade."""

    strategy_name: str
    current_weight: float
    target_weight: float
    trade_weight: float  # positive = buy, negative = sell
    trade_value: float  # in capital terms


@dataclass(frozen=True)
class RebalanceResult:
    """Complete rebalancing analysis."""

    trades: tuple[RebalanceTrade, ...]
    total_turnover: float  # sum of |trade_weight|
    estimated_cost: float  # estimated transaction cost
    needs_rebalancing: bool


def compute_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float,
    cost_rate: float = 0.004,  # default 0.4% (commission + slippage)
    drift_threshold: float = 0.03,  # 3% drift before rebalancing
    min_trade_value: float = 5000.0,  # KRW, Upbit minimum
) -> RebalanceResult:
    """Compute rebalancing trades.

    Args:
        current_weights: strategy → current weight
        target_weights: strategy → target weight
        portfolio_value: total portfolio value
        cost_rate: total round-trip cost rate
        drift_threshold: minimum drift to trigger rebalancing
        min_trade_value: minimum trade size (exchange minimum)

    Returns:
        RebalanceResult with trades and cost estimate
    """
    all_strategies = set(current_weights.keys()) | set(target_weights.keys())
    trades: list[RebalanceTrade] = []
    total_turnover = 0.0

    for strategy in sorted(all_strategies):
        current = current_weights.get(strategy, 0.0)
        target = target_weights.get(strategy, 0.0)
        diff = target - current

        trade_value = abs(diff) * portfolio_value

        # Skip if drift is below threshold or trade is too small
        if abs(diff) < drift_threshold or trade_value < min_trade_value:
            continue

        trades.append(RebalanceTrade(
            strategy_name=strategy,
            current_weight=current,
            target_weight=target,
            trade_weight=diff,
            trade_value=trade_value,
        ))
        total_turnover += abs(diff)

    estimated_cost = total_turnover * cost_rate * portfolio_value

    return RebalanceResult(
        trades=tuple(trades),
        total_turnover=total_turnover,
        estimated_cost=estimated_cost,
        needs_rebalancing=len(trades) > 0,
    )


def apply_target_weights(
    strategy_weights: dict[str, NDArray[np.float64]],
    allocation: dict[str, float],
) -> NDArray[np.float64]:
    """Combine per-strategy weights with strategy-level allocation.

    Args:
        strategy_weights: strategy → asset weight array within that strategy
        allocation: strategy → fraction of total capital

    Returns:
        Combined weight array (scaled by strategy allocation)
    """
    if not strategy_weights:
        return np.array([], dtype=np.float64)

    # Get total size
    first = next(iter(strategy_weights.values()))
    combined = np.zeros_like(first, dtype=np.float64)

    for name, weights in strategy_weights.items():
        alloc = allocation.get(name, 0.0)
        combined = combined + weights * alloc

    return combined
