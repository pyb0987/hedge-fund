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
from hedgefund.strategies.base import BaseStrategy, RebalanceDecision


class DualMomentumStrategy(BaseStrategy):
    """Cross-asset dual momentum: absolute + relative momentum switching."""

    def __init__(self, config: DualMomentumConfig) -> None:
        self._config = config
        self._offensive_assets = list(config.offensive_assets)
        self._defensive_assets = list(config.defensive_assets)
        self._defensive_weights = list(config.defensive_weights)
        self._last_rebalance_date: datetime | None = None

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
        return self._offensive_assets + list(self._defensive_assets)

    @property
    def last_rebalance_date(self) -> datetime | None:
        return self._last_rebalance_date

    @last_rebalance_date.setter
    def last_rebalance_date(self, value: datetime | None) -> None:
        self._last_rebalance_date = value

    def should_rebalance(self, timestamp: datetime) -> bool:
        """Check if it's time for monthly rebalancing.

        Matches backtest_weights() logic: rebalance on rebalance_day of each month.
        """
        if self._last_rebalance_date is None:
            return True

        # Same month → no rebalance
        if (
            timestamp.year == self._last_rebalance_date.year
            and timestamp.month == self._last_rebalance_date.month
        ):
            return False

        # New month — rebalance on or after rebalance_day
        return timestamp.day >= self._config.rebalance_day

    def get_rebalance_decision(self, timestamp: datetime) -> RebalanceDecision:
        if self._last_rebalance_date is None:
            return RebalanceDecision(
                should_rebalance=True,
                reason="first_run",
                days_since_last=None,
                gate_days=None,
            )
        days_elapsed = (timestamp - self._last_rebalance_date).days
        same_month = (
            timestamp.year == self._last_rebalance_date.year
            and timestamp.month == self._last_rebalance_date.month
        )
        if same_month:
            return RebalanceDecision(
                should_rebalance=False,
                reason="same_month",
                days_since_last=days_elapsed,
                gate_days=None,
            )
        if timestamp.day < self._config.rebalance_day:
            return RebalanceDecision(
                should_rebalance=False,
                reason="before_rebalance_day",
                days_since_last=days_elapsed,
                gate_days=None,
            )
        return RebalanceDecision(
            should_rebalance=True,
            reason="monthly_gate",
            days_since_last=days_elapsed,
            gate_days=None,
        )

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate dual momentum signals.

        Monthly rebalancing gate: returns empty list between rebalance dates.

        Decision tree:
        1. Check rebalancing gate (monthly on rebalance_day)
        2. Compute absolute momentum for each offensive asset
        3. If both positive → invest in the higher one (relative momentum winner)
        4. If only one positive → invest in that one
        5. If both negative → defensive (TLT)
        """
        if not self.should_rebalance(timestamp):
            return []

        lookback = self._config.lookback_days
        momenta: dict[str, float] = {}

        for symbol in self._offensive_assets:
            df = data.get(symbol)
            if df is None or len(df) < lookback + 1:
                return []  # insufficient data — no signal
            raw_mom = self.compute_momentum(df["close"], lookback)
            # Vol-adjusted momentum for fairer cross-asset comparison
            import numpy as np

            log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
            if len(log_ret) >= lookback:
                vol = float(log_ret.iloc[-lookback:].std())
                momenta[symbol] = raw_mom / vol if vol > 0 else raw_mom
            else:
                momenta[symbol] = raw_mom

        # Check if at least one defensive asset has data
        defensive_with_data = [
            s for s in self._defensive_assets if s in data and len(data[s]) >= lookback + 1
        ]
        if not defensive_with_data:
            return []

        signals: list[Signal] = []
        positive_assets = {s: m for s, m in momenta.items() if m > 0}

        if len(positive_assets) >= 2:
            # Both positive → relative momentum winner gets 100%
            winner = max(positive_assets, key=positive_assets.get)  # type: ignore[arg-type]
            for symbol in self._offensive_assets:
                direction = SignalDirection.LONG if symbol == winner else SignalDirection.FLAT
                strength = 1.0 if symbol == winner else 0.0
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=symbol,
                        exchange=exchange,
                        direction=direction,
                        strength=strength,
                        timestamp=timestamp,
                        metadata={
                            "momentum": momenta[symbol],
                            "regime": "offensive",
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )
            for def_sym in self._defensive_assets:
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=def_sym,
                        exchange=Exchange.ALPACA,
                        direction=SignalDirection.FLAT,
                        strength=0.0,
                        timestamp=timestamp,
                        metadata={
                            "regime": "offensive",
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )

        elif len(positive_assets) == 1:
            # Only one positive → invest in that one
            winner = list(positive_assets.keys())[0]
            for symbol in self._offensive_assets:
                direction = SignalDirection.LONG if symbol == winner else SignalDirection.FLAT
                strength = 1.0 if symbol == winner else 0.0
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=symbol,
                        exchange=exchange,
                        direction=direction,
                        strength=strength,
                        timestamp=timestamp,
                        metadata={
                            "momentum": momenta[symbol],
                            "regime": "mixed",
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )
            for def_sym in self._defensive_assets:
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=def_sym,
                        exchange=Exchange.ALPACA,
                        direction=SignalDirection.FLAT,
                        strength=0.0,
                        timestamp=timestamp,
                        metadata={
                            "regime": "mixed",
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )

        else:
            # Both negative → defensive basket mode
            for symbol in self._offensive_assets:
                exchange = Exchange.UPBIT if "KRW" in symbol else Exchange.ALPACA
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=symbol,
                        exchange=exchange,
                        direction=SignalDirection.FLAT,
                        strength=0.0,
                        timestamp=timestamp,
                        metadata={
                            "momentum": momenta[symbol],
                            "regime": "defensive",
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )
            for def_sym, def_weight in zip(self._defensive_assets, self._defensive_weights):
                signals.append(
                    Signal(
                        strategy_name=self.name,
                        symbol=def_sym,
                        exchange=Exchange.ALPACA,
                        direction=SignalDirection.LONG,
                        strength=def_weight,
                        timestamp=timestamp,
                        metadata={
                            "regime": "defensive",
                            "weight": def_weight,
                            "lookback_days": float(self._config.lookback_days),
                        },
                    )
                )

        # Mark rebalance date for monthly gate
        if signals:
            self._last_rebalance_date = timestamp

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
        all_symbols = self._offensive_assets + list(self._defensive_assets)
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

        for symbol in self._offensive_assets:
            df = data.get(symbol)
            if df is None:
                continue
            mask = df.index <= date
            available = df.loc[mask]
            if len(available) < lookback + 1:
                continue
            raw_mom = self.compute_momentum(available["close"], lookback)
            # Vol-adjusted momentum for fairer cross-asset comparison
            import numpy as np

            log_ret = np.log(available["close"] / available["close"].shift(1)).dropna()
            if len(log_ret) >= lookback:
                vol = float(log_ret.iloc[-lookback:].std())
                momenta[symbol] = raw_mom / vol if vol > 0 else raw_mom
            else:
                momenta[symbol] = raw_mom

        if len(momenta) < len(self._offensive_assets):
            return {}  # insufficient data

        positive = {s: m for s, m in momenta.items() if m > 0}

        if len(positive) >= 1:
            winner = max(positive, key=positive.get)  # type: ignore[arg-type]
            return {winner: 1.0}
        else:
            # Defensive basket allocation
            return {sym: w for sym, w in zip(self._defensive_assets, self._defensive_weights)}
