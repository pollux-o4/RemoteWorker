"""Comfy media validation and lossless request encoding."""

from __future__ import annotations

import io
import re
from typing import Any

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
        return isinstance(key, str) and _DYNAMIC_IMAGE_NAME.fullmatch(key) is not None

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
    """Flatten numbered IMAGE batches while preserving socket and batch order."""
    if (
        not isinstance(image_count, int)
        or isinstance(image_count, bool)
        or not 0 <= image_count <= _MAX_IMAGE_ATTACHMENTS
    ):
        raise ValueError(f"Remote Conditioning image_count must be 0 to {_MAX_IMAGE_ATTACHMENTS}")
    frames: list[Any] = []
    for index in range(1, image_count + 1):
        name = f"image_{index}"
        if name not in images or images[name] is None:
            raise ValueError(f"Remote Conditioning requires connected {name}")
        batch = images[name]
        shape = getattr(batch, "shape", None)
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ValueError(f"Remote Conditioning {name} must be a BHWC IMAGE tensor")
        count, height, width, channels = shape
        if not all(isinstance(value, int) and value > 0 for value in (count, height, width)) or channels not in {3, 4}:
            raise ValueError(f"Remote Conditioning {name} has an invalid IMAGE shape: {shape!r}")
        frames.extend(batch[batch_index] for batch_index in range(count))
        if len(frames) > _MAX_IMAGE_ATTACHMENTS:
            raise ValueError(
                f"Remote Conditioning accepts at most {_MAX_IMAGE_ATTACHMENTS} image attachments per request"
            )
    return frames


def encode_image_frame_png(frame: Any) -> bytes:
    """Encode one HWC ComfyUI frame without resizing it."""
    try:
        import numpy as np
        from PIL import Image

        array = frame.detach().to(device="cpu").float().numpy()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Remote Conditioning image attachment must be a tensor frame") from exc
    if array.ndim != 3 or array.shape[2] not in {3, 4} or not np.isfinite(array).all():
        raise ValueError(f"Remote Conditioning image attachment has invalid HWC data: {array.shape!r}")
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, "RGBA" if pixels.shape[2] == 4 else "RGB").save(output, format="PNG")
    data = output.getvalue()
    if not data or len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("Remote Conditioning encoded image exceeds the 32 MiB limit")
    return data
