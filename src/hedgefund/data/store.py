"""SQLite data store for persisting OHLCV data and portfolio state."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from hedgefund.core.exceptions import DataError

_OHLCV_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    interval TEXT NOT NULL DEFAULT 'day',
    PRIMARY KEY (symbol, exchange, timestamp, interval)
)
"""

_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    slippage REAL NOT NULL,
    strategy_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    pnl REAL
)
"""

_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    drawdown REAL NOT NULL,
    peak_value REAL NOT NULL
)
"""


class DataStore:
    """SQLite-backed persistent data store."""

    def __init__(self, db_path: str = "data/hedgefund.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.execute(_OHLCV_TABLE)
            conn.execute(_TRADES_TABLE)
            conn.execute(_SNAPSHOTS_TABLE)

    def save_ohlcv(
        self,
        symbol: str,
        exchange: str,
        df: pd.DataFrame,
        interval: str = "day",
    ) -> int:
        """Save OHLCV data, skipping duplicates.

        Returns number of rows inserted.
        """
        if df.empty:
            return 0

        rows = []
        for ts, row in df.iterrows():
            rows.append((
                symbol,
                exchange,
                str(ts),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                interval,
            ))

        with self._connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO ohlcv "
                "(symbol, exchange, timestamp, open, high, low, close, volume, interval) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return len(rows)

    def load_ohlcv(
        self,
        symbol: str,
        exchange: str,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Load OHLCV data from store."""
        query = (
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = ? AND exchange = ? AND interval = ?"
        )
        params: list = [symbol, exchange, interval]

        if start is not None:
            query += " AND timestamp >= ?"
            params.append(str(start))
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(str(end))

        query += " ORDER BY timestamp ASC"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    def has_data(
        self,
        symbol: str,
        exchange: str,
        interval: str = "day",
    ) -> bool:
        """Check if any data exists for a symbol."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND exchange = ? AND interval = ?",
                (symbol, exchange, interval),
            )
            count = cursor.fetchone()[0]
        return count > 0

    def save_snapshot(
        self,
        timestamp: datetime,
        total_value: float,
        cash: float,
        unrealized_pnl: float,
        realized_pnl: float,
        drawdown: float,
        peak_value: float,
    ) -> None:
        """Save a portfolio snapshot."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, total_value, cash, unrealized_pnl, realized_pnl, drawdown, peak_value) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), total_value, cash, unrealized_pnl, realized_pnl, drawdown, peak_value),
            )

    def load_snapshots(self) -> pd.DataFrame:
        """Load all portfolio snapshots."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        return df
