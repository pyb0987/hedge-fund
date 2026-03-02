"""Strategy C: Dual Momentum (Cross-Asset).

Gary Antonacci 듀얼 모멘텀 기반:
1. 절대 모멘텀: BTC와 SPY 각각의 lookback 수익률이 양수인지 확인
2. 상대 모멘텀: BTC vs SPY 중 수익률이 높은 쪽 선택
3. 둘 다 음수: 방어 자산(TLT)으로 전환

엣지 소스: 자산군 간 모멘텀 프리미엄 (학술적으로 가장 견고한 팩터 중 하나)
월간 리밸런싱으로 거래 비용 최소화.
"""

from datetime import datetime

import pandas as pd

from hedgefund.config.schemas import DualMomentumConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy

# Asset definitions
OFFENSIVE_ASSETS = ["KRW-BTC", "SPY"]
DEFENSIVE_ASSET = "TLT"


class DualMomentumStrategy(BaseStrategy):
    """Cross-asset dual momentum: absolute + relative momentum switching."""

    def __init__(self, config: DualMomentumConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        # Cross-asset strategy uses both exchanges
        return Exchange.UPBIT

    @property
    def config(self) -> DualMomentumConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return OFFENSIVE_ASSETS + [DEFENSIVE_ASSET]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate dual momentum signals.

        Decision tree:
        1. Compute absolute momentum for each offensive asset
        2. If both positive → invest in the higher one (relative momentum winner)
        3. If only one positive → invest in that one
        4. If both negative → defensive (TLT)
        """
        lookback = self._config.lookback_days
        momenta: dict[str, float] = {}

        for symbol in OFFENSIVE_ASSETS:
            df = data.get(symbol)
            if df is None or len(df) < lookback + 1:
                return []  # insufficient data — no signal
            momenta[symbol] = self.compute_momentum(df["close"], lookback)

        # Check if defensive asset data exists
        if DEFENSIVE_ASSET not in data or len(data[DEFENSIVE_ASSET]) < lookback + 1:
            return []

        signals: list[Signal] = []
        positive_assets = {s: m for s, m in momenta.items() if m > 0}

        if len(positive_assets) >= 2:
            # Both positive → relative momentum winner gets 100%
            winner = max(positive_assets, key=positive_assets.get)  # type: ignore[arg-type]
            for symbol in OFFENSIVE_ASSETS:
                direction = SignalDirection.LONG if symbol == winner else SignalDirection.FLAT
                strength = 1.0 if symbol == winner else 0.0
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=exchange,
                    direction=direction,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={"momentum": momenta[symbol], "regime": "offensive"},
                ))
            signals.append(Signal(
                strategy_name=self.name,
                symbol=DEFENSIVE_ASSET,
                exchange=Exchange.ALPACA,
                direction=SignalDirection.FLAT,
                strength=0.0,
                timestamp=timestamp,
                metadata={"regime": "offensive"},
            ))

        elif len(positive_assets) == 1:
            # Only one positive → invest in that one
            winner = list(positive_assets.keys())[0]
            for symbol in OFFENSIVE_ASSETS:
                direction = SignalDirection.LONG if symbol == winner else SignalDirection.FLAT
                strength = 1.0 if symbol == winner else 0.0
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=exchange,
                    direction=direction,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={"momentum": momenta[symbol], "regime": "mixed"},
                ))
            signals.append(Signal(
                strategy_name=self.name,
                symbol=DEFENSIVE_ASSET,
                exchange=Exchange.ALPACA,
                direction=SignalDirection.FLAT,
                strength=0.0,
                timestamp=timestamp,
                metadata={"regime": "mixed"},
            ))

        else:
            # Both negative → defensive mode (TLT)
            for symbol in OFFENSIVE_ASSETS:
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={"momentum": momenta[symbol], "regime": "defensive"},
                ))
            signals.append(Signal(
                strategy_name=self.name,
                symbol=DEFENSIVE_ASSET,
                exchange=Exchange.ALPACA,
                direction=SignalDirection.LONG,
                strength=1.0,
                timestamp=timestamp,
                metadata={"regime": "defensive"},
            ))

        return signals

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix for vectorized backtesting.

        Monthly rebalancing: compute decision once per month,
        hold until next rebalancing date.
        """
        all_symbols = OFFENSIVE_ASSETS + [DEFENSIVE_ASSET]
        available_symbols = [s for s in all_symbols if s in data]
        weights = pd.DataFrame(0.0, index=dates, columns=available_symbols)

        lookback = self._config.lookback_days
        rebalance_day = self._config.rebalance_day

        current_allocation: dict[str, float] = {}

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            # Only rebalance on the designated day (or first available day of month)
            is_rebalance = (
                date.day == rebalance_day
                or (i > 0 and dates[i - 1].month != date.month and date.day >= rebalance_day)
                or not current_allocation  # first time
            )

            if is_rebalance:
                current_allocation = self._compute_allocation(data, date, lookback)

            for sym, w in current_allocation.items():
                if sym in weights.columns:
                    weights.loc[date, sym] = w

        return weights

    def _compute_allocation(
        self,
        data: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        lookback: int,
    ) -> dict[str, float]:
        """Compute target allocation at a given date."""
        momenta: dict[str, float] = {}

        for symbol in OFFENSIVE_ASSETS:
            df = data.get(symbol)
            if df is None:
                continue
            mask = df.index <= date
            available = df.loc[mask]
            if len(available) < lookback + 1:
                continue
            momenta[symbol] = self.compute_momentum(available["close"], lookback)

        if len(momenta) < len(OFFENSIVE_ASSETS):
            return {}  # insufficient data

        positive = {s: m for s, m in momenta.items() if m > 0}

        if len(positive) >= 1:
            winner = max(positive, key=positive.get)  # type: ignore[arg-type]
            return {winner: 1.0}
        else:
            return {DEFENSIVE_ASSET: 1.0}
