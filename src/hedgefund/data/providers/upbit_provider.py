"""Upbit data provider — fetches KRW crypto OHLCV via pyupbit."""

import logging
import time
from datetime import datetime

import pandas as pd
import pyupbit
import requests

from hedgefund.core.exceptions import DataProviderError, InsufficientDataError
from hedgefund.core.timeout import call_with_timeout

logger = logging.getLogger(__name__)

_API_TIMEOUT = 30  # seconds per API call
_RATE_LIMIT_DELAY = 0.15  # seconds between API calls (Upbit ~10 req/s)
_MAX_RETRIES = 3

# Upbit interval mapping
_INTERVAL_MAP = {
    "day": "day",
    "week": "week",
    "minute1": "minute1",
    "minute3": "minute3",
    "minute5": "minute5",
    "minute15": "minute15",
    "minute60": "minute60",
    "minute240": "minute240",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pyupbit DataFrame to standard format."""
    column_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=column_map)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataProviderError(f"Missing columns from Upbit: {missing}")
    return df[required].sort_index()


class UpbitProvider:
    """Upbit OHLCV data provider."""

    @property
    def exchange_name(self) -> str:
        return "upbit"

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Upbit."""
        upbit_interval = _INTERVAL_MAP.get(interval)
        if upbit_interval is None:
            raise DataProviderError(
                f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP.keys())}"
            )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                time.sleep(_RATE_LIMIT_DELAY)
                df = call_with_timeout(
                    pyupbit.get_ohlcv,
                    timeout=_API_TIMEOUT,
                    args=(symbol,),
                    kwargs={"interval": upbit_interval, "count": count},
                )
            except TimeoutError:
                raise DataProviderError(
                    f"Upbit API timeout ({_API_TIMEOUT}s) for {symbol}"
                ) from None
            except Exception as e:
                raise DataProviderError(f"Upbit API error for {symbol}: {e}") from e

            if df is not None and not df.empty:
                break

            if attempt < _MAX_RETRIES:
                wait = attempt * 0.5
                logger.warning(
                    "Upbit retry %d/%d for %s (wait %.1fs)",
                    attempt,
                    _MAX_RETRIES,
                    symbol,
                    wait,
                )
                time.sleep(wait)

        if df is None or df.empty:
            raise InsufficientDataError(f"No data returned for {symbol} from Upbit")

        return _normalize_columns(df)

    def get_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Fetch OHLCV for date range from Upbit.

        pyupbit의 get_ohlcv는 'to' 파라미터로 끝 시점을 지정할 수 있으나
        시작 시점 지정이 어려우므로, 충분한 데이터를 가져온 후 필터링합니다.
        """
        days_diff = (end - start).days + 1
        count = min(days_diff + 50, 400)  # 여유분 포함, Upbit 최대 400

        df = self.get_ohlcv(symbol, interval=interval, count=count)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        filtered = df.loc[mask]

        if filtered.empty:
            raise InsufficientDataError(f"No data for {symbol} in range {start} ~ {end}")
        return filtered

    def get_available_symbols(self) -> list[str]:
        """Get KRW market tickers from Upbit."""
        try:
            tickers = call_with_timeout(
                pyupbit.get_tickers,
                timeout=_API_TIMEOUT,
                kwargs={"fiat": "KRW"},
            )
        except TimeoutError:
            raise DataProviderError(
                f"Upbit tickers API timeout ({_API_TIMEOUT}s)"
            ) from None
        except Exception as e:
            raise DataProviderError(f"Failed to get Upbit tickers: {e}") from e

        if tickers is None:
            return []
        return sorted(tickers)

    def get_top_volume_symbols(self, top_n: int = 15) -> list[str]:
        """Get top N symbols by 24h KRW trading volume.

        Upbit ticker API로 1회 호출하여 전체 거래대금을 조회합니다.
        기존: 243개 심볼 × get_ohlcv → rate limit 초과
        개선: /v1/ticker 1회 호출 → acc_trade_price_24h로 정렬
        """
        symbols = self.get_available_symbols()
        if not symbols:
            return []

        try:
            resp = requests.get(
                "https://api.upbit.com/v1/ticker",
                params={"markets": ",".join(symbols)},
                timeout=_API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Upbit ticker API failed, falling back to empty: %s", e)
            return []

        volumes = [
            (item["market"], item.get("acc_trade_price_24h", 0))
            for item in data
            if item.get("market", "").startswith("KRW-")
        ]
        volumes.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in volumes[:top_n]]
