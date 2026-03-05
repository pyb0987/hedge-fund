"""SQLite data store for persisting OHLCV data and portfolio state."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd


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

_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    direction TEXT NOT NULL,
    strength REAL NOT NULL,
    metadata TEXT
)
"""

_POSITION_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    market_price REAL NOT NULL,
    market_value REAL NOT NULL,
    unrealized_pnl REAL NOT NULL
)
"""

_STRATEGY_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS strategy_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    signals_generated INTEGER NOT NULL,
    target_allocation REAL NOT NULL,
    actual_allocation REAL NOT NULL,
    dd_multiplier REAL NOT NULL
)
"""

_RISK_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    symbol TEXT,
    strategy_name TEXT,
    rule_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    current_value REAL NOT NULL,
    limit_value REAL NOT NULL,
    message TEXT NOT NULL
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
            conn.execute(_SIGNALS_TABLE)
            conn.execute(_POSITION_SNAPSHOTS_TABLE)
            conn.execute(_STRATEGY_DECISIONS_TABLE)
            conn.execute(_RISK_EVENTS_TABLE)

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
        params: list[object] = [symbol, exchange, interval]

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

    def save_trade(
        self,
        symbol: str,
        exchange: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
        strategy_name: str,
        timestamp: datetime,
        pnl: float | None = None,
    ) -> None:
        """Save a single trade record."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(symbol, exchange, side, quantity, price, commission, slippage, "
                "strategy_name, timestamp, pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, exchange, side, quantity, price, commission, slippage,
                 strategy_name, str(timestamp), pnl),
            )

    def load_trades(self) -> pd.DataFrame:
        """Load all trade records."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save_signal(
        self,
        timestamp: datetime,
        strategy_name: str,
        symbol: str,
        exchange: str,
        direction: str,
        strength: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Save a single signal record."""
        meta_str = json.dumps(metadata) if metadata else None
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO signals "
                "(timestamp, strategy_name, symbol, exchange, direction, strength, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), strategy_name, symbol, exchange, direction,
                 strength, meta_str),
            )

    def load_signals(self) -> pd.DataFrame:
        """Load all signal records."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save_risk_event(
        self,
        timestamp: datetime,
        event_type: str,
        rule_name: str,
        passed: bool,
        current_value: float,
        limit_value: float,
        message: str,
        symbol: str | None = None,
        strategy_name: str | None = None,
    ) -> None:
        """Save a risk check event (pass or fail)."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO risk_events "
                "(timestamp, event_type, symbol, strategy_name, rule_name, "
                "passed, current_value, limit_value, message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), event_type, symbol, strategy_name,
                 rule_name, int(passed), current_value, limit_value, message),
            )

    def load_risk_events(self) -> pd.DataFrame:
        """Load all risk events."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM risk_events ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save_position_snapshot(
        self,
        timestamp: datetime,
        symbol: str,
        exchange: str,
        strategy_name: str,
        quantity: float,
        avg_entry_price: float,
        market_price: float,
        market_value: float,
        unrealized_pnl: float,
    ) -> None:
        """Save a single position snapshot row."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO position_snapshots "
                "(timestamp, symbol, exchange, strategy_name, quantity, "
                "avg_entry_price, market_price, market_value, unrealized_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), symbol, exchange, strategy_name,
                 quantity, avg_entry_price, market_price, market_value,
                 unrealized_pnl),
            )

    def load_position_snapshots(self) -> pd.DataFrame:
        """Load all position snapshots."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM position_snapshots ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def save_strategy_decision(
        self,
        timestamp: datetime,
        strategy_name: str,
        action: str,
        reason: str,
        signals_generated: int,
        target_allocation: float,
        actual_allocation: float,
        dd_multiplier: float,
    ) -> None:
        """Save a strategy rebalancing/allocation decision."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO strategy_decisions "
                "(timestamp, strategy_name, action, reason, signals_generated, "
                "target_allocation, actual_allocation, dd_multiplier) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), strategy_name, action, reason,
                 signals_generated, target_allocation, actual_allocation,
                 dd_multiplier),
            )

    def load_strategy_decisions(self) -> pd.DataFrame:
        """Load all strategy decisions."""
        with self._connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM strategy_decisions ORDER BY timestamp ASC",
                conn,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
