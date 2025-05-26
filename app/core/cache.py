from redis import asyncio as aioredis
import os
import json
import pickle
import structlog
from typing import Any, Optional, Union
from datetime import timedelta
import hashlib
from functools import wraps
from app.core.monitoring import track_cache_operation

# Initialize logger
logger = structlog.get_logger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"

# Default cache settings
DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))  # 1 hour
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1048576"))  # 1MB


class CacheError(Exception):
    """Base class for cache-related errors."""

    pass


class RedisCache:
    """Production-grade Redis cache implementation."""

    def __init__(self):
        """Initialize Redis connection pool."""
        self._redis = aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
            password=REDIS_PASSWORD,
            ssl=REDIS_SSL,
            encoding="utf-8",
            decode_responses=False,
            max_connections=20,
        )

    async def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return await self._redis.ping()
        except Exception as e:
            logger.error("Redis ping failed", error=str(e))
            return False

    @track_cache_operation
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            data = await self._redis.get(key)
            if data is None:
                return None

            return pickle.loads(data)

        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None

    @track_cache_operation
    async def set(
        self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds or timedelta

        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize value
            data = pickle.dumps(value)

            # Check size limit
            if len(data) > MAX_CACHE_SIZE:
                logger.warning(
                    "Cache value exceeds size limit",
                    key=key,
                    size=len(data),
                    limit=MAX_CACHE_SIZE,
                )
                return False

            # Convert timedelta to seconds
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())

            # Set with TTL
            await self._redis.set(key, data, ex=ttl or DEFAULT_CACHE_TTL)

            return True

        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(await self._redis.delete(key))
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self._redis.exists(key))
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            await self._redis.flushdb()
            return True
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False

    async def get_or_set(
        self, key: str, value_func, ttl: Optional[Union[int, timedelta]] = None
    ) -> Any:
        """
        Get value from cache or compute and store it.

        Args:
            key: Cache key
            value_func: Async function to compute value if not cached
            ttl: Time to live in seconds or timedelta

        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        try:
            value = await value_func()
        except Exception as e:
            logger.error("Cache value computation error", key=key, error=str(e))
            raise

        # Store in cache
        await self.set(key, value, ttl)

        return value

    async def close(self):
        """Close Redis connection."""
        await self._redis.close()


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.

    Example:
        @cached()
        async def get_user(user_id: int):
            pass

        Cache key will be: "get_user:123"
    """
    # Convert args and kwargs to strings
    arg_strings = [str(arg) for arg in args]
    kwarg_strings = [f"{k}={v}" for k, v in sorted(kwargs.items())]

    # Join all arguments
    key_parts = arg_strings + kwarg_strings
    key = ":".join(key_parts)

    # Hash if too long
    if len(key) > 200:
        key = hashlib.sha256(key.encode()).hexdigest()

    return key


def cached(
    ttl: Optional[Union[int, timedelta]] = None, key_prefix: Optional[str] = None
):
    """
    Cache decorator for async functions.

    Args:
        ttl: Time to live in seconds or timedelta
        key_prefix: Prefix for cache key

    Example:
        @cached(ttl=300)  # Cache for 5 minutes
        async def get_user(user_id: int):
            pass
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_args = list(args)
            if cache_args and hasattr(cache_args[0], "__class__"):
                cache_args = cache_args[1:]  # Remove self/cls

            key = cache_key(*cache_args, **kwargs)
            if key_prefix:
                key = f"{key_prefix}:{key}"
            else:
                key = f"{func.__name__}:{key}"

            # Get or compute value
            cache = RedisCache()
            try:
                return await cache.get_or_set(key, lambda: func(*args, **kwargs), ttl)
            finally:
                await cache.close()

        return wrapper

    return decorator


# Create global cache instance
cache = RedisCache()
