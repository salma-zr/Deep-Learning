"""Disk-based caching utilities to avoid redundant API calls."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps

from diskcache import Cache

from src.utils.io_utils import get_project_root


class DiskCache:
    """
    Disk-based cache for expensive operations (API calls, generations, etc.).
    
    Uses diskcache for persistence and thread-safety.
    """
    
    def __init__(self, cache_dir: Optional[str | Path] = None, namespace: str = "default"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache storage. Defaults to .cache/ in project root.
            namespace: Namespace to separate different cache types.
        """
        if cache_dir is None:
            cache_dir = get_project_root() / ".cache"
        
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.cache_dir))
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a unique cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set a value in cache with optional expiration (seconds)."""
        self._cache.set(key, value, expire=expire)
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        expire: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and store."""
        value = self.get(key)
        if value is not None:
            return value
        
        value = compute_fn()
        self.set(key, value, expire=expire)
        return value
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "directory": str(self.cache_dir),
        }


# Global cache instances
_caches: dict[str, DiskCache] = {}


def get_cache(namespace: str = "default") -> DiskCache:
    """Get or create a cache instance for the given namespace."""
    if namespace not in _caches:
        _caches[namespace] = DiskCache(namespace=namespace)
    return _caches[namespace]


def cached(namespace: str = "default", expire: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        expire: Optional expiration time in seconds
        
    Usage:
        @cached(namespace="api_calls")
        def expensive_api_call(prompt: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache(namespace)
            key = cache._make_key(func.__name__, *args, **kwargs)
            
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(key, result, expire=expire)
            return result
        
        return wrapper
    return decorator


def cache_key_for_generation(
    model: str,
    prompt: str,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Create a unique cache key for a generation request."""
    key_data = {
        "model": model,
        "prompt": prompt,
        "question": question,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]
