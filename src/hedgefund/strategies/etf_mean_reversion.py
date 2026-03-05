"""Strategy B: US ETF Mean Reversion.

SPY, QQQ, TLT, GLD, IEF 바스켓에서 20일 수익률 z-score 기반 평균회귀.
z-score < -1.5 → 매수 (과매도), z-score > +1.5 → 청산 (과매수).

엣지 소스: 기관 리밸런싱 플로우 (분기말/월말 대량 매매 → 일시적 가격 왜곡 → 회귀)
"""

from datetime import datetime

import numpy as np
import pandas as pd

from hedgefund.config.schemas import EtfMeanReversionConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy


class EtfMeanReversionStrategy(BaseStrategy):
    """Z-score based mean reversion on US ETF basket."""

    def __init__(self, config: EtfMeanReversionConfig, universe: list[str] | None = None) -> None:
        self._config = config
        self._universe = universe or list(config.universe)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    @property
    def config(self) -> EtfMeanReversionConfig:
        return self._config

    def get_universe(self) -> list[str]:
        return list(self._universe)

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate mean-reversion signals based on z-scores.

        Logic:
        1. Compute rolling z-score of returns for each ETF
        2. z < entry_threshold → LONG (oversold, expect reversion up)
        3. z > exit_threshold → FLAT (overbought, close position)
        4. Otherwise → maintain current signal direction
        """
        signals: list[Signal] = []

        for symbol in self._universe:
            df = data.get(symbol)
            if df is None or len(df) < self._config.lookback_days + 1:
                continue

            z = self._compute_rolling_zscore(df["close"])

            if z <= self._config.z_entry_threshold:
                # Oversold — mean reversion buy
                strength = self._zscore_to_strength(z, is_entry=True)
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.LONG,
                    strength=strength,
                    timestamp=timestamp,
                    metadata={"z_score": z},
                ))
            elif z >= self._config.z_exit_threshold:
                # Overbought — close position
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={"z_score": z},
                ))
            else:
                # Neutral zone — no strong signal
                signals.append(Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    exchange=self.exchange,
                    direction=SignalDirection.FLAT,
                    strength=0.0,
                    timestamp=timestamp,
                    metadata={"z_score": z},
                ))

        return signals

    def _compute_rolling_zscore(self, prices: pd.Series) -> float:
        """Compute z-score of cumulative log-return over lookback window.

        Uses log returns for mathematical consistency:
        sum(log_returns) is the exact log of the price ratio,
        and CLT gives us mean/std scaling for the sum.
        """
        lookback = self._config.lookback_days

        if len(prices) < lookback + 1:
            return 0.0

        # Log returns: ln(P_t / P_{t-1})
        log_returns = np.log(prices / prices.shift(1)).dropna()

        if len(log_returns) < lookback:
            return 0.0

        window = log_returns.iloc[-lookback:]
        cumulative_log_return = float(window.sum())

        mean = float(window.mean()) * lookback
        std = float(window.std()) * np.sqrt(lookback)

        if std == 0:
            return 0.0
        return (cumulative_log_return - mean) / std

    def _zscore_to_strength(self, z: float, is_entry: bool) -> float:
        """Convert z-score magnitude to signal strength (0-1).

        Deeper oversold/overbought → stronger signal.
        """
        threshold = abs(self._config.z_entry_threshold if is_entry else self._config.z_exit_threshold)
        max_z = threshold * 2  # z of -3.0 at threshold -1.5 → max strength
        magnitude = abs(z)
        return min(1.0, max(0.0, (magnitude - threshold) / (max_z - threshold)))

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix for vectorized backtesting.

        Mean reversion: equal-weight all symbols with z < entry_threshold.
        If no symbols qualify, stay in cash (all zeros).
        """
        symbols = list(data.keys())
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)

        lookback = self._config.lookback_days
        z_entry = self._config.z_entry_threshold
        z_exit = self._config.z_exit_threshold

        # Track which symbols are currently in position
        in_position: dict[str, bool] = {s: False for s in symbols}

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            active_symbols: list[str] = []

            for sym in symbols:
                df = data[sym]
                mask = df.index <= date
                available = df.loc[mask]
                if len(available) < lookback + 1:
                    continue

                # Use same z-score as generate_signals() for consistency
                z = self._compute_rolling_zscore(available["close"])

                if z <= z_entry:
                    in_position[sym] = True
                elif z >= z_exit:
                    in_position[sym] = False

                if in_position[sym]:
                    active_symbols.append(sym)

            if active_symbols:
                weight_per = 1.0 / len(active_symbols)
                for sym in active_symbols:
                    weights.loc[date, sym] = weight_per

        return weights
