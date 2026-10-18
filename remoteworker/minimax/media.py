"""MiniMax reference socket validation, media encoding and fingerprints."""
from __future__ import annotations

import hashlib
import io
import wave
from typing import Any

from ..core.media import encode_image_frame_png
from ..core.protocol import CACHE_POLICIES, DEFAULT_CACHE_POLICY
from .common import (
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    VAE_NAME_OVERRIDE_SOCKET,
)

_MAX_REF_IMAGES = 9
_MAX_REF_VIDEOS = 3
_MAX_REF_AUDIOS = 3


class DynamicReferenceMedia(dict[str, tuple[str]]):
    """Expose the official MiniMax ref2va autogrow socket families to ComfyUI."""

    _KINDS = {
        "ref_image": "IMAGE",
        "ref_video": "IMAGE",
        "ref_video_audio": "AUDIO",
        "ref_audio": "AUDIO",
    }

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        prefix, separator, suffix = key.rpartition("_")
        return bool(separator and prefix in self._KINDS and suffix.isdigit() and int(suffix) >= 0)

    def __getitem__(self, key: str) -> tuple[str]:
        if key not in self:
            raise KeyError(key)
        prefix, _, _ = key.rpartition("_")
        return (self._KINDS[prefix],)


class DynamicReferenceAndVaeInputs(DynamicReferenceMedia):
    """Dynamic MiniMax media sockets plus explicit remote VAE-name overrides."""

    _FIXED = {
        "vae_name_override": (VAE_NAME_OVERRIDE_SOCKET, {"forceInput": True}),
        "audio_vae_name_override": (VAE_NAME_OVERRIDE_SOCKET, {"forceInput": True}),
        "peer_memory_policy": (list(PEER_MEMORY_POLICIES), {"default": DEFAULT_PEER_MEMORY_POLICY}),
        "cache_policy": (list(CACHE_POLICIES), {"default": DEFAULT_CACHE_POLICY}),
    }

    def __init__(self) -> None:
        super().__init__(self._FIXED)

    def __contains__(self, key: object) -> bool:
        return key in self._FIXED or super().__contains__(key)

    def __getitem__(self, key: str) -> tuple[str]:
        if key in self._FIXED:
            return self._FIXED[key]
        return super().__getitem__(key)


def _connected_slots(prefix: str, media: dict[str, Any], maximum: int) -> list[tuple[str, Any]]:
    """Read contiguous autogrow sockets; their presence, not a counter widget, is the SSoT."""
    entries: list[tuple[int, str, Any]] = []
    for name, value in media.items():
        head, separator, tail = name.rpartition("_")
        if separator and head == prefix and tail.isdigit() and int(tail) >= 0:
            entries.append((int(tail), name, value))
    entries.sort()
    indices = [index for index, _, _ in entries]
    if len(entries) > maximum:
        raise ValueError(f"Remote MiniMax H3 accepts at most {maximum} {prefix} references")
    if indices != list(range(len(entries))):
        raise ValueError(f"Remote MiniMax H3 {prefix} sockets must be connected without gaps")
    return [(name, value) for _, name, value in entries]


def _ordered_images(prefix: str, media: dict[str, Any], *, batches: bool, maximum: int) -> list[Any]:
    ordered: list[Any] = []
    for name, image in _connected_slots(prefix, media, maximum):
        shape = getattr(image, "shape", None)
        if image is None:
            raise ValueError(f"Remote MiniMax H3 requires connected {name}")
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ValueError(f"Remote MiniMax H3 {name} must be a BHWC IMAGE tensor")
        frames, height, width, channels = shape
        if not all(isinstance(value, int) and value > 0 for value in (frames, height, width)) or channels not in {3, 4}:
            raise ValueError(f"Remote MiniMax H3 {name} has an invalid IMAGE shape: {shape!r}")
        if not batches and frames != 1:
            raise ValueError(f"Remote MiniMax H3 {name} must contain exactly one reference image")
        if batches and frames < 5:
            raise ValueError(f"Remote MiniMax H3 {name} needs at least 5 frames")
        ordered.append(image)
    return ordered


def _ordered_audio(prefix: str, media: dict[str, Any], maximum: int) -> list[Any]:
    result: list[Any] = []
    for name, audio in _connected_slots(prefix, media, maximum):
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError(f"Remote MiniMax H3 {name} must be a connected AUDIO value")
        result.append(audio)
    return result


def encode_audio_wav(audio: Any) -> bytes:
    """Encode one ComfyUI AUDIO object as portable PCM WAV without GPU retention."""
    try:
        import numpy as np

        waveform = audio["waveform"].detach().to(device="cpu").float().numpy()
        sample_rate = audio["sample_rate"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Remote MiniMax H3 AUDIO must contain a tensor waveform and sample_rate") from exc
    if not isinstance(sample_rate, int) or not 1 <= sample_rate <= 384000:
        raise ValueError("Remote MiniMax H3 audio sample_rate is invalid")
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError("Remote MiniMax H3 AUDIO waveform must have shape [1, channels, samples]")
    if not np.isfinite(waveform).all():
        raise ValueError("Remote MiniMax H3 AUDIO waveform contains non-finite values")
    pcm = np.rint(np.clip(waveform[0].T, -1.0, 1.0) * 32767).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _bytes_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _image_batch_fingerprint(batch: Any) -> tuple[str, ...]:
    return tuple(_bytes_fingerprint(encode_image_frame_png(batch[index])) for index in range(batch.shape[0]))


def _audio_fingerprint(audio: Any) -> str:
    return _bytes_fingerprint(encode_audio_wav(audio))


