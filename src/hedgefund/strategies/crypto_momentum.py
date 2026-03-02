"""Strategy A: Crypto Momentum.

Upbit KRW 상위 코인 중 lookback일 모멘텀 상위 top_n개를 매수.
나머지는 KRW (현금) 보유.

엣지 소스: 암호화폐 시장의 행동 편향 (추세 추종 프리미엄)
"""

from datetime import datetime

import pandas as pd

from hedgefund.config.schemas import CryptoMomentumConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy


class CryptoMomentumStrategy(BaseStrategy):
    """Cross-sectional momentum on Upbit KRW crypto pairs."""

    def __init__(self, config: CryptoMomentumConfig, universe: list[str] | None = None) -> None:
        self._config = config
        self._universe = universe or []

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.UPBIT

    @property
    def config(self) -> CryptoMomentumConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return list(self._universe)

    def set_universe(self, symbols: list[str]) -> None:
        """Update the tradeable universe (e.g., from top volume filter)."""
        self._universe = list(symbols)

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate momentum signals.

        1. Compute lookback-day momentum for each symbol
        2. Rank by momentum (descending)
        3. Top N → LONG signal, rest → FLAT

        Args:
            data: symbol → OHLCV DataFrame (must have 'close' column)
            timestamp: evaluation time

        Returns:
            List of Signal objects
        """
        momentum_scores = self._compute_momentum_scores(data)

        if not momentum_scores:
            return []

        # Sort by momentum descending
        ranked = sorted(momentum_scores, key=lambda x: x[1], reverse=True)
        top_n = self._config.top_n

        signals: list[Signal] = []
        for i, (symbol, score) in enumerate(ranked):
            if i < top_n and score > 0:
                # Long signal — only buy positive momentum
                strength = self._normalize_strength(score, ranked)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={"momentum_score": score, "rank": float(i + 1)},
                ))
            else:
                # Flat — sell or stay out
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={"momentum_score": score, "rank": float(i + 1)},
                ))

        return signals

    def _compute_momentum_scores(
        self,
        data: dict[str, pd.DataFrame],
    ) -> list[tuple[str, float]]:
        """Compute momentum score for each symbol."""
        scores: list[tuple[str, float]] = []
        lookback = self._config.lookback_days

        for symbol in self._universe:
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
        """Normalize momentum score to 0-1 strength.

        Uses min-max scaling across the ranked list.
        """
        scores = [s for _, s in ranked if s > 0]
        if not scores or len(scores) < 2:
            return 1.0 if score > 0 else 0.0

        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score

        if range_score == 0:
            return 1.0
        return max(0.0, min(1.0, (score - min_score) / range_score))

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix for vectorized backtesting.

        Returns DataFrame with symbols as columns, dates as index,
        and weights (0.0 or equal-weight among top_n) as values.
        """
        symbols = list(data.keys())
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)

        lookback = self._config.lookback_days
        top_n = self._config.top_n

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            # Compute momentum for each symbol at this date
            scores: list[tuple[str, float]] = []
            for sym in symbols:
                df = data[sym]
                # Get data up to this date
                mask = df.index <= date
                available = df.loc[mask]
                if len(available) < lookback + 1:
                    continue
                momentum = self.compute_momentum(available["close"], lookback)
                scores.append((sym, momentum))

            # Rank and assign weights
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            selected = [(s, m) for s, m in ranked[:top_n] if m > 0]

            if selected:
                weight_per_asset = 1.0 / len(selected)
                for sym, _ in selected:
                    weights.loc[date, sym] = weight_per_asset

        return weights
