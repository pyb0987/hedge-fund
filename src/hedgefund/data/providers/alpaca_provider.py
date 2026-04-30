"""Alpaca data provider — fetches US ETF/stock OHLCV via Alpaca Markets data API.

yfinance가 차단/장애 빈발하여 신뢰할 수 없는 운영 데이터 소스가 됨에 따라,
실거래용 키와 동일한 Alpaca 자격증명으로 historical bars를 받아온다.
무료 (IEX) feed로도 일별 close는 종가에 충분히 근접한다.
"""

from datetime import datetime, timedelta

import pandas as pd

from hedgefund.core.exceptions import DataProviderError, InsufficientDataError
from hedgefund.core.timeout import call_with_timeout

_API_TIMEOUT = 30


_INTERVAL_MAP = {
    "day": "1Day",
    "week": "1Week",
    "month": "1Month",
    "minute60": "1Hour",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Alpaca bars DataFrame to the standard OHLCV layout."""
    if df.empty:
        return df

    required = ["open", "high", "low", "close", "volume"]
    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        missing = set(required) - set(available)
        raise DataProviderError(f"Missing columns from Alpaca bars: {missing}")

    out = df[required].copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.sort_index()


class AlpacaProvider:
    """Alpaca historical bars provider for US ETFs/stocks."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        feed: str = "iex",
    ) -> None:
        if not api_key or not secret_key:
            raise DataProviderError("Alpaca API key and secret are required")

        try:
            import alpaca_trade_api as tradeapi
        except ImportError as e:
            raise DataProviderError(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            ) from e

        self._api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        self._feed = feed

    @property
    def exchange_name(self) -> str:
        return "alpaca"

    def _resolve_timeframe(self, interval: str):  # type: ignore[no-untyped-def]
        from alpaca_trade_api import TimeFrame

        mapping = {
            "day": TimeFrame.Day,
            "week": TimeFrame.Week,
            "month": TimeFrame.Month,
            "minute60": TimeFrame.Hour,
        }
        tf = mapping.get(interval)
        if tf is None:
            raise DataProviderError(
                f"Unsupported interval '{interval}'. Supported: {list(mapping.keys())}"
            )
        return tf

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame:
        """Fetch the most recent ``count`` bars for ``symbol``."""
        timeframe = self._resolve_timeframe(interval)
        end = datetime.utcnow()
        # Wide window to absorb weekends/holidays — we tail() at the end.
        if interval == "day":
            start = end - timedelta(days=count * 2 + 30)
        elif interval == "week":
            start = end - timedelta(weeks=count * 2)
        elif interval == "month":
            start = end - timedelta(days=count * 62)
        else:
            start = end - timedelta(hours=count * 2)

        try:
            bars = call_with_timeout(
                self._api.get_bars,
                timeout=_API_TIMEOUT,
                args=(symbol, timeframe),
                kwargs={
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "adjustment": "raw",
                    "feed": self._feed,
                },
            )
        except TimeoutError:
            raise DataProviderError(
                f"Alpaca data API timeout ({_API_TIMEOUT}s) for {symbol}"
            ) from None
        except Exception as e:
            raise DataProviderError(f"Alpaca data error for {symbol}: {e}") from e

        df = getattr(bars, "df", None)
        if df is None or df.empty:
            raise InsufficientDataError(f"No data returned for {symbol} from Alpaca")

        df = _normalize_columns(df)
        return df.tail(count)

    def get_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Fetch bars for a date range (inclusive)."""
        timeframe = self._resolve_timeframe(interval)
        # Pad start to absorb weekends/holidays then trim to exact range.
        start_padded = start - timedelta(days=5)

        try:
            bars = call_with_timeout(
                self._api.get_bars,
                timeout=_API_TIMEOUT,
                args=(symbol, timeframe),
                kwargs={
                    "start": start_padded.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "adjustment": "raw",
                    "feed": self._feed,
                },
            )
        except TimeoutError:
            raise DataProviderError(
                f"Alpaca data API timeout ({_API_TIMEOUT}s) for {symbol}"
            ) from None
        except Exception as e:
            raise DataProviderError(f"Alpaca data error for {symbol}: {e}") from e

        df = getattr(bars, "df", None)
        if df is None or df.empty:
            raise InsufficientDataError(f"No data for {symbol} in range {start} ~ {end}")

        df = _normalize_columns(df)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]

    def get_available_symbols(self) -> list[str]:
        """Default ETF universe used by the strategies."""
        return ["SPY", "QQQ", "TLT", "GLD", "IEF", "BIL", "GDX", "SLV", "SH", "VTI"]
