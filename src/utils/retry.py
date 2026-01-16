import time
from typing import Callable, TypeVar


T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    backoff: float = 2.0,
    on_error: Callable[[Exception, int], None] | None = None,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # pylint: disable=broad-except
            if attempt >= max_attempts:
                raise
            if on_error:
                on_error(exc, attempt)
            time.sleep(backoff ** attempt)
