"""Comfy media validation and lossless request encoding."""

from __future__ import annotations

import io
import re
from typing import Any

from .. import _native
from .protocol import (
    CACHE_POLICIES,
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    _parse_env_positive_int,
)

_MAX_IMAGE_BYTES = _parse_env_positive_int("REMOTE_CONDITIONING_MAX_IMAGE_BYTES", 32 * 1024 * 1024)
_MAX_IMAGE_ATTACHMENTS = 64
_DYNAMIC_IMAGE_NAME = re.compile(r"^image_[1-9][0-9]*$")


class DynamicImageInputs(dict[str, tuple[str]]):
    """Declare numbered IMAGE sockets without capping future Strategy cardinality."""

    def __contains__(self, key: object) -> bool:
        return _native.image_socket_name(key)

    def __getitem__(self, key: str) -> tuple[str]:
        if key not in self:
            raise KeyError(key)
        return ("IMAGE",)


class DynamicImageAndPolicyInputs(DynamicImageInputs):
    """Backward-compatible optional settings plus numbered IMAGE sockets."""

    _FIXED = {
        "peer_memory_policy": (
            list(PEER_MEMORY_POLICIES),
            {"default": DEFAULT_PEER_MEMORY_POLICY},
        ),
        "cache_policy": (list(CACHE_POLICIES), {"default": DEFAULT_CACHE_POLICY}),
    }

    def __init__(self) -> None:
        super().__init__(self._FIXED)

    def __contains__(self, key: object) -> bool:
        return key in self._FIXED or super().__contains__(key)

    def __getitem__(self, key: str) -> tuple[Any, ...]:
        if key in self._FIXED:
            return self._FIXED[key]
        return super().__getitem__(key)


def ordered_image_frames(image_count: int, **images: Any) -> list[Any]:
    from .._native import ordered_image_frames as ordered

    return ordered(image_count, images)


def encode_image_frame_png(frame: Any) -> bytes:
    """Encode one HWC ComfyUI frame without resizing it."""
    try:
        import numpy as np
        from PIL import Image

        array = frame.detach().to(device="cpu").float().numpy()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Remote Conditioning image attachment must be a tensor frame") from exc
    mode = _native.image_array_mode(array.shape, bool(np.isfinite(array).all()))
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode).save(output, format="PNG")
    data = output.getvalue()
    _native.check_image_budget(len(data), _MAX_IMAGE_BYTES)
    return data
