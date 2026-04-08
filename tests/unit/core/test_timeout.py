"""Tests for hedgefund.core.timeout.call_with_timeout."""

import threading
import time

import pytest

from hedgefund.core.timeout import call_with_timeout


def test_returns_value_on_success() -> None:
    assert call_with_timeout(lambda x: x * 2, timeout=1.0, args=(21,)) == 42


def test_kwargs_are_forwarded() -> None:
    def fn(a: int, *, b: int) -> int:
        return a + b

    assert call_with_timeout(fn, timeout=1.0, args=(1,), kwargs={"b": 2}) == 3


def test_propagates_exception_from_fn() -> None:
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        call_with_timeout(boom, timeout=1.0)


def test_raises_timeout_when_fn_hangs() -> None:
    started = threading.Event()

    def hang() -> None:
        started.set()
        time.sleep(5.0)

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        call_with_timeout(hang, timeout=0.2)
    elapsed = time.monotonic() - t0

    # Must return promptly — the broken ThreadPoolExecutor pattern would have
    # blocked here for the full 5s waiting on shutdown(wait=True).
    assert started.is_set()
    assert elapsed < 1.0, f"call_with_timeout blocked for {elapsed:.2f}s"


def test_zero_or_negative_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        call_with_timeout(lambda: None, timeout=0)
    with pytest.raises(ValueError):
        call_with_timeout(lambda: None, timeout=-1)


def test_abandoned_thread_is_daemon() -> None:
    """Hung worker must not block interpreter exit."""
    captured: list[threading.Thread] = []

    def hang() -> None:
        captured.append(threading.current_thread())
        time.sleep(5.0)

    with pytest.raises(TimeoutError):
        call_with_timeout(hang, timeout=0.1)

    assert captured, "worker should have started"
    assert captured[0].daemon is True
