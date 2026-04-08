"""Hard-timeout primitives for blocking external calls.

배경
----
`concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=...)` 패턴은
`with ThreadPoolExecutor(...) as pool:` 구문과 함께 쓰면 **timeout이 사실상 무력화**
된다. `with` block의 `__exit__`이 `shutdown(wait=True)`를 호출하므로, future가
TimeoutError를 던져도 워커 스레드가 자연 종료할 때까지 무한 대기한다. 워커 스레드는
daemon이 아니므로 인터프리터 종료도 막는다.

본 모듈은 그 대신 명시적인 daemon `threading.Thread`를 사용한다. timeout 시
스레드를 단순히 버리고 호출자에게 `TimeoutError`를 즉시 던진다. 버려진 스레드는
daemon이므로 프로세스 종료를 막지 않는다.

사용 예:
    from hedgefund.core.timeout import call_with_timeout

    try:
        result = call_with_timeout(some_blocking_api, timeout=30, args=(symbol,))
    except TimeoutError as e:
        ...
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def call_with_timeout(
    fn: Callable[..., T],
    timeout: float,
    *,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> T:
    """Run ``fn(*args, **kwargs)`` with a hard wall-clock timeout.

    Parameters
    ----------
    fn:
        Callable to invoke.
    timeout:
        Seconds to wait. Must be positive.
    args, kwargs:
        Forwarded to ``fn``.

    Raises
    ------
    TimeoutError
        If ``fn`` does not return within ``timeout`` seconds.
    Exception
        Any exception raised by ``fn`` is re-raised on the caller's thread.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    kwargs = kwargs or {}
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — propagate everything
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True, name=f"timeout-{fn.__name__}")
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"{fn.__name__} did not return within {timeout}s (thread abandoned)"
        )

    if "error" in box:
        raise box["error"]
    return box["value"]
