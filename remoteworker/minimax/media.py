"""MiniMax reference socket validation, media encoding and fingerprints."""

from __future__ import annotations

import hashlib
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

    def __getitem__(self, key: str) -> tuple[str]:
        if key in self._FIXED:
            return self._FIXED[key]
        return super().__getitem__(key)


def _connected_slots(prefix: str, media: dict[str, Any], maximum: int) -> list[tuple[str, Any]]:
    from .._native import connected_slots_native

    return connected_slots_native(prefix, media, maximum)


def _ordered_images(prefix: str, media: dict[str, Any], *, batches: bool, maximum: int) -> list[Any]:
    from .._native import ordered_images_native

    return ordered_images_native(prefix, media, batches, maximum, False)


def _ordered_audio(prefix: str, media: dict[str, Any], maximum: int) -> list[Any]:
    from .._native import ordered_audio_native

    return ordered_audio_native(prefix, media, maximum)


def encode_audio_wav(audio: Any) -> bytes:
    from .._native import encode_audio_wav_native

    try:
        waveform = audio["waveform"].detach().to(device="cpu").float().numpy()
        sample_rate = audio["sample_rate"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Remote MiniMax H3 AUDIO must contain a tensor waveform and sample_rate") from exc
    return bytes(
        encode_audio_wav_native(
            sample_rate,
            list(waveform.shape),
            waveform.astype("<f4", copy=False).tobytes(),
        )
    )


def _bytes_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _image_batch_fingerprint(batch: Any) -> tuple[str, ...]:
    return tuple(_bytes_fingerprint(encode_image_frame_png(batch[index])) for index in range(batch.shape[0]))


def _audio_fingerprint(audio: Any) -> str:
    return _bytes_fingerprint(encode_audio_wav(audio))
