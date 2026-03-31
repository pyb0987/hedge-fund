"""Strategy: ETF Pairs Trading (market-neutral alpha).

사전 정의된 코인티그레이션 ETF 쌍(GLD/GDX, XLE/USO, TLT/IEF)에서
스프레드 z-score 기반 평균회귀 트레이딩.

엣지 소스: 구조적 코인티그레이션 (동일 기초자산 노출) → 일시적 괴리 → 회귀.
Long undervalued leg + Short overvalued leg → market-neutral alpha.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from hedgefund.config.schemas import PairsTradingConfig
from hedgefund.core.enums import Exchange, SignalDirection
from hedgefund.core.models import Signal
from hedgefund.strategies.base import BaseStrategy, RebalanceDecision


class PairsTradingStrategy(BaseStrategy):
    """Spread z-score based pairs trading on cointegrated ETF pairs."""

    def __init__(self, config: PairsTradingConfig | None = None) -> None:
        self._config = config or PairsTradingConfig()
        self._parsed_pairs = self._config.parsed_pairs()
        self._last_rebalance_date: datetime | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def exchange(self) -> Exchange:
        return Exchange.ALPACA

    @property
    def config(self) -> PairsTradingConfig:
        return self._config

    def get_universe(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for a, b in self._parsed_pairs:
            for sym in (a, b):
                if sym not in seen:
                    seen.add(sym)
                    result.append(sym)
        return result

    def get_rebalance_decision(self, timestamp: datetime) -> RebalanceDecision:
        return self._holding_day_rebalance_decision(
            timestamp,
            self._last_rebalance_date,
            self._config.holding_days,
        )

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime,
    ) -> list[Signal]:
        """Generate pairs trading signals.

        For each pair (A, B):
        1. Compute rolling OLS hedge ratio: β = cov(A,B) / var(B)
        2. Spread = log(A) - β × log(B)
        3. Z-score spread against rolling mean/std
        4. z > entry → SHORT A + LONG B (spread overvalued)
        5. z < -entry → LONG A + SHORT B (spread undervalued)
        6. |z| < exit → FLAT both legs
        """
        signals: list[Signal] = []
        min_data = self._config.lookback_days + 1

        for sym_a, sym_b in self._parsed_pairs:
            df_a = data.get(sym_a)
            df_b = data.get(sym_b)
            if df_a is None or df_b is None:
                continue
            if len(df_a) < min_data or len(df_b) < min_data:
                continue

            prices_a = df_a["close"]
            prices_b = df_b["close"]

            beta = self._compute_hedge_ratio(
                prices_a,
                prices_b,
                self._config.hedge_ratio_window,
            )
            z = self._compute_spread_zscore(
                prices_a,
                prices_b,
                self._config.lookback_days,
            )

            if not np.isfinite(z):
                continue

            metadata = {
                "spread_zscore": z,
                "hedge_ratio": beta,
                "pair": f"{sym_a}/{sym_b}",
            }

            if z > self._config.zscore_entry:
                # Spread overvalued: SHORT A, LONG B
                signals.extend(
                    [
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_a,
                            exchange=self.exchange,
                            direction=SignalDirection.SHORT,
                            strength=self._zscore_to_strength(z),
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_b,
                            exchange=self.exchange,
                            direction=SignalDirection.LONG,
                            strength=self._zscore_to_strength(z),
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                    ]
                )
            elif z < -self._config.zscore_entry:
                # Spread undervalued: LONG A, SHORT B
                signals.extend(
                    [
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_a,
                            exchange=self.exchange,
                            direction=SignalDirection.LONG,
                            strength=self._zscore_to_strength(abs(z)),
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_b,
                            exchange=self.exchange,
                            direction=SignalDirection.SHORT,
                            strength=self._zscore_to_strength(abs(z)),
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                    ]
                )
            else:
                # Neutral zone or exit → FLAT both legs
                signals.extend(
                    [
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_a,
                            exchange=self.exchange,
                            direction=SignalDirection.FLAT,
                            strength=0.0,
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                        Signal(
                            strategy_name=self.name,
                            symbol=sym_b,
                            exchange=self.exchange,
                            direction=SignalDirection.FLAT,
                            strength=0.0,
                            timestamp=timestamp,
                            metadata=metadata,
                        ),
                    ]
                )

        if signals:
            self._last_rebalance_date = timestamp

        return signals

    def backtest_weights(
        self,
        data: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate weight matrix for vectorized backtesting.

        Pairs trading: +weight on long leg, -weight on short leg.
        Gross exposure per active pair = 1/num_pairs.
        """
        all_symbols = self.get_universe()
        weights = pd.DataFrame(0.0, index=dates, columns=all_symbols)

        lookback = self._config.lookback_days
        z_entry = self._config.zscore_entry
        z_exit = self._config.zscore_exit
        num_pairs = len(self._parsed_pairs)

        # Track position state per pair: 1=long spread, -1=short spread, 0=flat
        pair_state: dict[tuple[str, str], int] = {pair: 0 for pair in self._parsed_pairs}

        for i, date in enumerate(dates):
            if i < lookback:
                continue

            for sym_a, sym_b in self._parsed_pairs:
                if sym_a not in data or sym_b not in data:
                    continue

                df_a = data[sym_a]
                df_b = data[sym_b]
                mask_a = df_a.index <= date
                mask_b = df_b.index <= date
                avail_a = df_a.loc[mask_a]
                avail_b = df_b.loc[mask_b]

                if len(avail_a) < lookback + 1 or len(avail_b) < lookback + 1:
                    continue

                # Use same z-score as generate_signals
                z = self._compute_spread_zscore(
                    avail_a["close"],
                    avail_b["close"],
                    lookback,
                )

                if not np.isfinite(z):
                    continue

                pair_key = (sym_a, sym_b)
                state = pair_state[pair_key]

                # Entry/exit logic
                if state == 0:
                    if z > z_entry:
                        pair_state[pair_key] = -1  # short spread
                    elif z < -z_entry:
                        pair_state[pair_key] = 1  # long spread
                else:
                    if abs(z) < z_exit:
                        pair_state[pair_key] = 0  # exit
                    elif state == 1 and z > z_entry:
                        pair_state[pair_key] = -1  # flip
                    elif state == -1 and z < -z_entry:
                        pair_state[pair_key] = 1  # flip

                # Assign weights based on state
                state = pair_state[pair_key]
                weight_per_leg = 0.5 / num_pairs  # gross = 1.0 across all pairs

                if state == 1:
                    # Long spread: LONG A, SHORT B
                    weights.loc[date, sym_a] = weight_per_leg
                    weights.loc[date, sym_b] = -weight_per_leg
                elif state == -1:
                    # Short spread: SHORT A, LONG B
                    weights.loc[date, sym_a] = -weight_per_leg
                    weights.loc[date, sym_b] = weight_per_leg
                # state == 0 → weights stay 0

        return weights

    def _compute_hedge_ratio(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        window: int,
    ) -> float:
        """Compute rolling OLS hedge ratio: β = cov(A,B) / var(B).

        Uses log prices for the regression.
        """
        log_a = np.log(prices_a.iloc[-window:].values.astype(np.float64))
        log_b = np.log(prices_b.iloc[-window:].values.astype(np.float64))

        min_len = min(len(log_a), len(log_b))
        if min_len < 10:
            return 1.0

        log_a = log_a[-min_len:]
        log_b = log_b[-min_len:]

        var_b = np.var(log_b, ddof=1)
        if var_b == 0:
            return 1.0

        cov_ab = np.cov(log_a, log_b, ddof=1)[0, 1]
        return float(cov_ab / var_b)

    def _compute_spread_zscore(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        window: int,
    ) -> float:
        """Compute z-score of the log-price spread.

        Spread = log(A) - β × log(B)
        Z = (spread[-1] - mean(spread[-window:])) / std(spread[-window:])
        """
        beta = self._compute_hedge_ratio(prices_a, prices_b, window)

        log_a = np.log(prices_a.values.astype(np.float64))
        log_b = np.log(prices_b.values.astype(np.float64))
        min_len = min(len(log_a), len(log_b))
        log_a = log_a[-min_len:]
        log_b = log_b[-min_len:]

        spread = log_a - beta * log_b

        if len(spread) < window:
            return 0.0

        spread_window = spread[-window:]
        mean = np.mean(spread_window)
        std = np.std(spread_window, ddof=1)

        if std == 0:
            return 0.0

        return float((spread[-1] - mean) / std)

    def _zscore_to_strength(self, z_abs: float) -> float:
        """Convert |z-score| to signal strength (0-1)."""
        entry = self._config.zscore_entry
        max_z = entry * 2
        return min(1.0, max(0.0, (z_abs - entry) / (max_z - entry)))
