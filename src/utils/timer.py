"""Timer utilities for measuring execution time and latency."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Generator
import statistics

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for a collection of timing measurements."""
    
    measurements: list[float] = field(default_factory=list)
    
    def add(self, duration_ms: float) -> None:
        """Add a measurement in milliseconds."""
        self.measurements.append(duration_ms)
    
    @property
    def count(self) -> int:
        return len(self.measurements)
    
    @property
    def total_ms(self) -> float:
        return sum(self.measurements)
    
    @property
    def mean_ms(self) -> float:
        if not self.measurements:
            return 0.0
        return statistics.mean(self.measurements)
    
    @property
    def median_ms(self) -> float:
        if not self.measurements:
            return 0.0
        return statistics.median(self.measurements)
    
    @property
    def p95_ms(self) -> float:
        if not self.measurements:
            return 0.0
        sorted_m = sorted(self.measurements)
        idx = int(len(sorted_m) * 0.95)
        return sorted_m[min(idx, len(sorted_m) - 1)]
    
    @property
    def min_ms(self) -> float:
        if not self.measurements:
            return 0.0
        return min(self.measurements)
    
    @property
    def max_ms(self) -> float:
        if not self.measurements:
            return 0.0
        return max(self.measurements)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


class Timer:
    """Timer for measuring execution time."""
    
    def __init__(self, name: Optional[str] = None, log: bool = False):
        """
        Initialize timer.
        
        Args:
            name: Optional name for logging
            log: If True, log timing when stopped
        """
        self.name = name
        self.log = log
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return duration in milliseconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        
        self._end_time = time.perf_counter()
        duration_ms = self.duration_ms
        
        if self.log and self.name:
            logger.info(f"{self.name}: {duration_ms:.2f}ms")
        
        return duration_ms
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self._start_time is None:
            return 0.0
        
        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000
    
    def __enter__(self) -> "Timer":
        """Start timer when entering context."""
        return self.start()
    
    def __exit__(self, *args) -> None:
        """Stop timer when exiting context."""
        self.stop()


@contextmanager
def timed(name: str, log: bool = True) -> Generator[Timer, None, None]:
    """
    Context manager for timing a block of code.
    
    Usage:
        with timed("my_operation") as t:
            # do stuff
        print(f"Duration: {t.duration_ms}ms")
    """
    timer = Timer(name=name, log=log)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


# Global timing collectors
_timing_collectors: dict[str, TimingStats] = {}


def get_timing_stats(name: str) -> TimingStats:
    """Get or create a timing statistics collector."""
    if name not in _timing_collectors:
        _timing_collectors[name] = TimingStats()
    return _timing_collectors[name]


def record_timing(name: str, duration_ms: float) -> None:
    """Record a timing measurement."""
    get_timing_stats(name).add(duration_ms)


def get_all_timing_stats() -> dict[str, dict]:
    """Get all timing statistics as a dictionary."""
    return {name: stats.to_dict() for name, stats in _timing_collectors.items()}


def reset_timing_stats() -> None:
    """Reset all timing statistics."""
    _timing_collectors.clear()
