"""Retry utilities for handling API failures."""

import time
import random
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay
        exceptions: Tuple of exception types to catch
        on_retry: Optional callback called on each retry with (exception, attempt_number)
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class RetryableError(Exception):
    """Exception that signals the operation should be retried."""
    pass


class NonRetryableError(Exception):
    """Exception that signals the operation should NOT be retried."""
    pass


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    error_str = str(exception).lower()
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "throttl",
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


def is_transient_error(exception: Exception) -> bool:
    """Check if an exception is likely transient and worth retrying."""
    error_str = str(exception).lower()
    transient_indicators = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "unavailable",
        "503",
        "502",
        "500",
        "504",
    ]
    return any(indicator in error_str for indicator in transient_indicators)


class APIRetryHandler:
    """Context manager for handling API retries with detailed logging."""
    
    def __init__(
        self,
        operation_name: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        self.operation_name = operation_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.attempts = 0
        self.last_error = None
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        for attempt in range(self.max_retries + 1):
            self.attempts = attempt + 1
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.last_error = e
                
                if attempt >= self.max_retries:
                    logger.error(f"{self.operation_name}: Failed after {self.attempts} attempts")
                    raise
                
                # Check if we should retry
                if is_rate_limit_error(e):
                    delay = self.base_delay * (2 ** attempt) * 2  # Extra delay for rate limits
                    logger.warning(
                        f"{self.operation_name}: Rate limited. Waiting {delay:.1f}s..."
                    )
                elif is_transient_error(e):
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"{self.operation_name}: Transient error. Retrying in {delay:.1f}s..."
                    )
                else:
                    # Unknown error, try with normal backoff
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"{self.operation_name}: Error ({type(e).__name__}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                
                time.sleep(delay)
        
        raise self.last_error
