"""Recommendation cache: Redis when REDIS_URL is set, otherwise an in-process TTL cache.

Invalidation uses a per-user *generation* number that is part of every key.
A new rating bumps the user's generation, so all of their cached results
become unreachable at once, without scanning or deleting keys.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Cache(Protocol):
    backend: str

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl: int) -> None: ...
    def incr(self, key: str) -> int: ...
    def ping(self) -> bool: ...


class MemoryCache:
    backend = "memory"

    def __init__(self, max_items: int = 10_000) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:  # crude bound: drop expired, then the oldest
                now = time.monotonic()
                for k in [k for k, (e, _) in self._data.items() if e < now]:
                    del self._data[k]
                while len(self._data) >= self.max_items:
                    self._data.pop(next(iter(self._data)))
            self._data[key] = (time.monotonic() + ttl, value)

    def incr(self, key: str) -> int:
        with self._lock:
            _, value = self._data.get(key, (0.0, "0"))
            new = int(value) + 1
            self._data[key] = (float("inf"), str(new))
            return new

    def ping(self) -> bool:
        return True


class RedisCache:
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)

    def get(self, key: str) -> str | None:
        try:
            return self.client.get(key)
        except Exception:  # a cache outage must never break recommendations
            return None

    def set(self, key: str, value: str, ttl: int) -> None:
        try:
            self.client.set(key, value, ex=ttl)
        except Exception:
            pass

    def incr(self, key: str) -> int:
        try:
            return int(self.client.incr(key))
        except Exception:
            return 0

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def make_cache(redis_url: str | None) -> Cache:
    return RedisCache(redis_url) if redis_url else MemoryCache()
