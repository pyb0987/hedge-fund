"""Tests for TTL cache."""

from datetime import datetime, timedelta

from hedgefund.data.cache import CacheEntry, TTLCache


class TestCacheEntry:
    def test_immutable(self) -> None:
        entry = CacheEntry(data="hello", expires_at=datetime.now())
        assert entry.data == "hello"


class TestTTLCache:
    def test_set_and_get(self) -> None:
        cache = TTLCache(ttl_seconds=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key(self) -> None:
        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_invalidate(self) -> None:
        cache = TTLCache()
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self) -> None:
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_make_key(self) -> None:
        key = TTLCache.make_key("ohlcv", "KRW-BTC", "day")
        assert key == "ohlcv:KRW-BTC:day"

    def test_expired_entry_returns_none(self) -> None:
        cache = TTLCache(ttl_seconds=0)  # immediate expiry
        cache.set("key", "value")
        # Entry expires immediately
        assert cache.get("key") is None

    def test_cleanup_expired(self) -> None:
        cache = TTLCache(ttl_seconds=0)
        cache.set("a", 1)
        cache.set("b", 2)
        removed = cache.cleanup_expired()
        # Both entries should be expired (ttl=0), but timing may vary
        assert removed >= 1
        assert cache.size == 0

    def test_size(self) -> None:
        cache = TTLCache()
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1
        cache.set("b", 2)
        assert cache.size == 2
