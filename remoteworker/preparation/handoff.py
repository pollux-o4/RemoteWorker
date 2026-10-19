"""Bounded one-shot per-job results; refresh never becomes cross-job reuse."""

import hashlib
import io
import json
from collections import OrderedDict
from threading import Lock


def signature(values):
    def normalize(value):
        if hasattr(value, "detach"):
            import torch

            tensor = value.detach().cpu().contiguous()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(
                    tensor.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(v) for v in value]
        return value

    values = {
        k: v
        for k, v in values.items()
        if k not in {"self", "unique_id", "extra_pnginfo", "timeout_seconds"}
    }
    return hashlib.sha256(
        json.dumps(normalize(values), sort_keys=True).encode()
    ).hexdigest()


def pack(result, max_bytes=None):
    import torch

    def size(value):
        if hasattr(value, "numel"):
            if getattr(value, "is_nested", False):
                return sum(size(v) for v in value.unbind())
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(v) for v in value)
        return 0

    if max_bytes is not None and size(result) > max_bytes:
        return None
    buffer = io.BytesIO()
    torch.save(result, buffer)
    if max_bytes is not None and buffer.tell() > max_bytes:
        return None
    return buffer.getvalue()


def unpack(blob):
    import torch

    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)


class Handoffs:
    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.limit = max_bytes
        self.entries = OrderedDict()
        self.lock = Lock()
        self.bytes_used = 0

    def put(self, key, stamp, blob):
        with self.lock:
            self._drop(key)
            if len(blob) > self.limit:
                return False
            self.entries[key] = (stamp, blob)
            self.bytes_used += len(blob)
            while self.bytes_used > self.limit:
                self._drop(next(iter(self.entries)))
            return True

    def take(self, key, stamp):
        with self.lock:
            entry = self.entries.get(key)
            self._drop(key)
            return entry[1] if entry and entry[0] == stamp else None

    def retain(self, prompt_ids):
        with self.lock:
            for key in list(self.entries):
                if key[0] not in prompt_ids:
                    self._drop(key)

    def _drop(self, key):
        old = self.entries.pop(key, None)
        if old:
            self.bytes_used -= len(old[1])
