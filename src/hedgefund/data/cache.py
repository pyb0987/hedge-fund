"""TTL-based in-memory cache for data provider results."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """Immutable cache entry with expiration."""

    data: Any
    expires_at: datetime


class TTLCache:
    """Simple TTL-based cache.

    Thread-safety note: This is not thread-safe. For concurrent access,
    wrap with a lock or use a thread-safe alternative.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        """Get value if present and not expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if datetime.now() >= entry.expires_at:
            del self._store[key]
            return None
        return entry.data

    def set(self, key: str, value: Any) -> None:
        """Store value with TTL."""
        expires_at = datetime.now() + self._ttl
        # Create new dict entry (immutable entry)
        self._store[key] = CacheEntry(data=value, expires_at=expires_at)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        self._store = {}

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        now = datetime.now()
        expired_keys = [k for k, v in self._store.items() if now >= v.expires_at]
        for key in expired_keys:
            del self._store[key]
        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired)."""
        return len(self._store)

    @staticmethod
    def make_key(prefix: str, *parts: str) -> str:
        """Build a cache key from parts."""
        return f"{prefix}:{':'.join(parts)}"
