"""Strategy D: Market Hedge (Inverse ETF Overlay).

시장 하락 감지 시 인버스 ETF를 매수하여 포트폴리오 beta를 감소시킴.
SPY 60일 모멘텀 < 0 → SH (inverse S&P500) 매수
QQQ 60일 모멘텀 < 0 → PSQ (inverse QQQ) 매수
둘 다 양수 → 전량 현금 (FLAT)

엣지 소스: 포트폴리오 방향성 리스크 헷지 (alpha가 아닌 beta 감소 목적)
"""

from datetime import datetime

import pandas as pd

from hedgefund.config.schemas import MarketHedgeConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy


class MarketHedgeStrategy(BaseStrategy):
    """Trend-following inverse ETF strategy for beta hedging."""

    def __init__(self, config: MarketHedgeConfig, **kwargs: object) -> None:
        self._config = config
        self._index_assets = list(config.index_assets)
        self._inverse_assets = list(config.inverse_assets)
        self._last_rebalance_date: datetime | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    @property
    def config(self) -> MarketHedgeConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return self._index_assets + self._inverse_assets

    @property
    def last_rebalance_date(self) -> datetime | None:
        return self._last_rebalance_date

    @last_rebalance_date.setter
    def last_rebalance_date(self, value: datetime | None) -> None:
        self._last_rebalance_date = value

    def should_rebalance(self, timestamp: datetime) -> bool:
        """Check if enough days have elapsed since last rebalance."""
        if self._last_rebalance_date is None:
            return True
        days_elapsed = (timestamp - self._last_rebalance_date).days
        return days_elapsed >= self._config.holding_days

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate hedge signals based on index momentum.

        Logic:
        1. Check rebalancing gate (holding_days interval)
        2. For each index-inverse pair:
           - If index momentum < 0 → LONG inverse ETF (hedge active)
           - If index momentum >= 0 → FLAT inverse ETF (no hedge needed)
        """
        if not self.should_rebalance(timestamp):
            return []

        lookback = self._config.lookback_days
        signals: list[Signal] = []

        for index_sym, inverse_sym in zip(self._index_assets, self._inverse_assets):
            index_df = data.get(index_sym)
            if index_df is None or len(index_df) < lookback + 1:
                continue

            momentum = self.compute_momentum(index_df["close"], lookback)

            if momentum < 0:
                # Index declining → activate hedge (buy inverse ETF)
                strength = min(1.0, abs(momentum) / 0.10)  # scale: -10% → full strength
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=inverse_sym,
                    exchange=self.exchange,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={
                        "index_symbol": index_sym,
                        "index_momentum": momentum,
                        "regime": "hedge_active",
                        "lookback_days": float(lookback),
                    },
                ))
            else:
                # Index rising → deactivate hedge
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=inverse_sym,
                    exchange=self.exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={
                        "index_symbol": index_sym,
                        "index_momentum": momentum,
                        "regime": "hedge_inactive",
                        "lookback_days": float(lookback),
                    },
                ))

        if signals:
            self._last_rebalance_date = timestamp

        return signals

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix for vectorized backtesting.

        Inverse ETFs are bought (positive weights) when corresponding
        index has negative momentum. Equal weight among active hedges.
        """
        # Only use inverse symbols as columns (we don't trade the indices)
        weights = pd.DataFrame(0.0, index=dates, columns=self._inverse_assets)

        lookback = self._config.lookback_days
        holding_days = self._config.holding_days

        days_since_rebalance = 0
        current_weights: dict[str, float] = {}

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            is_rebalance = (
                not current_weights
                or days_since_rebalance >= holding_days
            )

            if is_rebalance:
                active_hedges: list[str] = []

                for index_sym, inverse_sym in zip(self._index_assets, self._inverse_assets):
                    index_df = data.get(index_sym)
                    if index_df is None:
                        continue
                    mask = index_df.index <= date
                    available = index_df.loc[mask]
                    if len(available) < lookback + 1:
                        continue

                    momentum = self.compute_momentum(available["close"], lookback)
                    if momentum < 0:
                        active_hedges.append(inverse_sym)

                current_weights = {}
                if active_hedges:
                    weight_per = 1.0 / len(active_hedges)
                    for sym in active_hedges:
                        current_weights[sym] = weight_per

                days_since_rebalance = 0

            for sym, w in current_weights.items():
                if sym in weights.columns:
                    weights.loc[date, sym] = w

            days_since_rebalance += 1

        return weights
