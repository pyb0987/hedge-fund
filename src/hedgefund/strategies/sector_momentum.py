"""Strategy: Sector ETF Momentum.

11개 SPDR 섹터 ETF에서 lookback일 모멘텀 상위 top_n개를 매수.
나머지는 현금 보유. holding_days 간격으로 리밸런싱.

엣지 소스: 섹터 로테이션 프리미엄 (기관 리밸런싱 지연 + 행동 편향)
"""

from datetime import datetime

import pandas as pd

from hedgefund.config.schemas import SectorMomentumConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy, RebalanceDecision


class SectorMomentumStrategy(BaseStrategy):
    """Cross-sectional momentum on US sector ETFs."""

    def __init__(self, config: SectorMomentumConfig, **kwargs: object) -> None:
        self._config = config
        self._last_rebalance_date: datetime | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    @property
    def config(self) -> SectorMomentumConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return list(self._config.universe)

    @property
    def last_rebalance_date(self) -> datetime | None:
        return self._last_rebalance_date

    @last_rebalance_date.setter
    def last_rebalance_date(self, value: datetime | None) -> None:
        self._last_rebalance_date = value

    def should_rebalance(self, timestamp: datetime) -> bool:
        if self._last_rebalance_date is None:
            return True
        days_elapsed = (timestamp - self._last_rebalance_date).days
        return days_elapsed >= self._config.holding_days

    def get_rebalance_decision(self, timestamp: datetime) -> RebalanceDecision:
        if self._last_rebalance_date is None:
            return RebalanceDecision(
                should_rebalance=True, reason="first_run",
                days_since_last=None, gate_days=self._config.holding_days,
            )
        days_elapsed = (timestamp - self._last_rebalance_date).days
        if days_elapsed >= self._config.holding_days:
            return RebalanceDecision(
                should_rebalance=True, reason="holding_period_met",
                days_since_last=days_elapsed, gate_days=self._config.holding_days,
            )
        return RebalanceDecision(
            should_rebalance=False, reason="too_early",
            days_since_last=days_elapsed, gate_days=self._config.holding_days,
        )

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate sector momentum signals.

        1. Check rebalancing gate (holding_days interval)
        2. Compute lookback-day momentum for each sector ETF
        3. Rank by momentum descending
        4. Top N with positive momentum → LONG, rest → FLAT
        """
        if not self.should_rebalance(timestamp):
            return []

        scores = self._compute_momentum_scores(data)
        if not scores:
            return []

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        top_n = self._config.top_n

        signals: list[Signal] = []
        for i, (symbol, score) in enumerate(ranked):
            if i < top_n and score > 0:
                strength = self._normalize_strength(score, ranked)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={
                        "momentum_score": score,
                        "rank": float(i + 1),
                        "top_n": float(top_n),
                        "lookback_days": float(self._config.lookback_days),
                    },
                ))
            else:
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={
                        "momentum_score": score,
                        "rank": float(i + 1),
                        "top_n": float(top_n),
                        "lookback_days": float(self._config.lookback_days),
                    },
                ))

        if signals:
            self._last_rebalance_date = timestamp

        return signals

    def _compute_momentum_scores(
        self,
        data: dict[str, pd.DataFrame],
    ) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        lookback = self._config.lookback_days

        for symbol in self._config.universe:
            df = data.get(symbol)
            if df is None or len(df) < lookback + 1:
                continue
            momentum = self.compute_momentum(df["close"], lookback)
            scores.append((symbol, momentum))

        return scores

    @staticmethod
    def _normalize_strength(
        score: float,
        ranked: list[tuple[str, float]],
    ) -> float:
        scores = [s for _, s in ranked if s > 0]
        if not scores or len(scores) < 2:
            return 1.0 if score > 0 else 0.0
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return 1.0
        return max(0.0, min(1.0, (score - min_s) / (max_s - min_s)))

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix with holding_days rebalancing."""
        symbols = list(data.keys())
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)

        lookback = self._config.lookback_days
        top_n = self._config.top_n
        holding_days = self._config.holding_days

        days_since_rebalance = 0
        current_weights: dict[str, float] = {}

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            is_rebalance = not current_weights or days_since_rebalance >= holding_days

            if is_rebalance:
                scores: list[tuple[str, float]] = []
                for sym in symbols:
                    df = data[sym]
                    available = df.loc[df.index <= date]
                    if len(available) < lookback + 1:
                        continue
                    momentum = self.compute_momentum(available["close"], lookback)
                    scores.append((sym, momentum))

                ranked = sorted(scores, key=lambda x: x[1], reverse=True)
                selected = [(s, m) for s, m in ranked[:top_n] if m > 0]

                current_weights = {}
                if selected:
                    w = 1.0 / len(selected)
                    for sym, _ in selected:
                        current_weights[sym] = w

                days_since_rebalance = 0

            for sym, w in current_weights.items():
                weights.loc[date, sym] = w

            days_since_rebalance += 1

        return weights
