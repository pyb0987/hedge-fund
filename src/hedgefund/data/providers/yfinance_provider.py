"""Yahoo Finance data provider — fetches US ETF/stock OHLCV via yfinance."""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from hedgefund.core.exceptions import DataProviderError, InsufficientDataError

_INTERVAL_MAP = {
    "day": "1d",
    "week": "1wk",
    "month": "1mo",
    "minute60": "60m",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance DataFrame to standard format."""
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_map)
    required = ["open", "high", "low", "close", "volume"]
    available = [c for c in required if c in df.columns]
    if len(available) < len(required):
        missing = set(required) - set(available)
        raise DataProviderError(f"Missing columns from yfinance: {missing}")
    return df[required].sort_index()


class YFinanceProvider:
    """Yahoo Finance data provider for US ETFs and stocks."""

    @property
    def exchange_name(self) -> str:
        return "yfinance"

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "day",
        count: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance."""
        yf_interval = _INTERVAL_MAP.get(interval)
        if yf_interval is None:
            raise DataProviderError(
                f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP.keys())}"
            )

        # Estimate period needed for count bars
        days_needed = count * 2 if interval == "day" else count * 10
        period = f"{days_needed}d"

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
        except Exception as e:
            raise DataProviderError(f"yfinance error for {symbol}: {e}") from e

        if df is None or df.empty:
            raise InsufficientDataError(f"No data returned for {symbol} from yfinance")

        df = _normalize_columns(df)
        return df.tail(count)

    def get_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Fetch OHLCV for date range from Yahoo Finance."""
        yf_interval = _INTERVAL_MAP.get(interval)
        if yf_interval is None:
            raise DataProviderError(f"Unsupported interval '{interval}'")

        # Add buffer days to handle weekends/holidays
        start_with_buffer = start - timedelta(days=5)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_with_buffer.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=yf_interval,
            )
        except Exception as e:
            raise DataProviderError(f"yfinance error for {symbol}: {e}") from e

        if df is None or df.empty:
            raise InsufficientDataError(
                f"No data for {symbol} in range {start} ~ {end}"
            )

        df = _normalize_columns(df)
        # Filter to exact range (tz-naive comparison)
        df.index = df.index.tz_localize(None)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask]

    def get_available_symbols(self) -> list[str]:
        """Return default ETF universe."""
        return ["SPY", "QQQ", "TLT", "GLD", "IEF", "VTI", "BND", "EFA", "EEM", "DBC"]
