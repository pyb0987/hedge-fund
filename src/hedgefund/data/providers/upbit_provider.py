"""Upbit data provider — fetches KRW crypto OHLCV via pyupbit."""

from datetime import datetime

import pandas as pd
import pyupbit

from hedgefund.core.exceptions import DataProviderError, InsufficientDataError

_API_TIMEOUT = 30  # seconds per API call

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

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    pyupbit.get_ohlcv, symbol, interval=upbit_interval, count=count
                )
                df = future.result(timeout=_API_TIMEOUT)
        except FuturesTimeout:
            raise DataProviderError(f"Upbit API timeout ({_API_TIMEOUT}s) for {symbol}")
        except Exception as e:
            raise DataProviderError(f"Upbit API error for {symbol}: {e}") from e

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
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(pyupbit.get_tickers, fiat="KRW")
                tickers = future.result(timeout=_API_TIMEOUT)
        except FuturesTimeout:
            raise DataProviderError(f"Upbit tickers API timeout ({_API_TIMEOUT}s)")
        except Exception as e:
            raise DataProviderError(f"Failed to get Upbit tickers: {e}") from e

        if tickers is None:
            return []
        return sorted(tickers)

    def get_top_volume_symbols(self, top_n: int = 15) -> list[str]:
        """Get top N symbols by 24h KRW trading volume.

        Useful for crypto momentum universe selection.
        """
        symbols = self.get_available_symbols()
        volumes: list[tuple[str, float]] = []

        for sym in symbols:
            try:
                df = self.get_ohlcv(sym, interval="day", count=1)
                if not df.empty:
                    vol = float(df["close"].iloc[-1] * df["volume"].iloc[-1])
                    volumes.append((sym, vol))
            except (DataProviderError, InsufficientDataError):
                continue

        volumes.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in volumes[:top_n]]
