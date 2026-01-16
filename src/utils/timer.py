import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[dict]:
    start = time.time()
    data = {"start": start, "end": None, "elapsed_ms": None}
    try:
        yield data
    finally:
        end = time.time()
        data["end"] = end
        data["elapsed_ms"] = int((end - start) * 1000)
