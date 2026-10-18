"""Small process-local byte-result cache for remote ComfyUI artifacts."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any


class ByteResultCache:
    """Bounded LRU cache for serialized remote artifacts.

    The cache deliberately stores bytes, not tensors/models, so cache hits do not
    keep GPU objects alive and do not couple peer memory cleanup to local reuse.
    """

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        if max_entries < 1:
            raise ValueError("ByteResultCache max_entries must be at least 1")
        if max_bytes < 1:
            raise ValueError("ByteResultCache max_bytes must be at least 1")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple[Any, ...], tuple[bytes, ...]] = OrderedDict()
        self._sizes: dict[tuple[Any, ...], int] = {}
        self._total_bytes = 0
        self._lock = RLock()

    def get(self, key: tuple[Any, ...]) -> tuple[bytes, ...] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return tuple(bytes(item) for item in value)

    def put(self, key: tuple[Any, ...], parts: tuple[bytes, ...]) -> None:
        payload = tuple(bytes(part) for part in parts)
        size = sum(len(part) for part in payload)
        with self._lock:
            if key in self._entries:
                self._total_bytes -= self._sizes.pop(key)
                self._entries.pop(key)
            if size > self._max_bytes:
                return
            self._entries[key] = payload
            self._sizes[key] = size
            self._total_bytes += size
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries or self._total_bytes > self._max_bytes:
                old_key, _old_value = self._entries.popitem(last=False)
                self._total_bytes -= self._sizes.pop(old_key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._sizes.clear()
            self._total_bytes = 0
