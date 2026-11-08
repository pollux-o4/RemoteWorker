"""Shared peer validation and artifact decoding for MiniMax H3 remote strategies."""

from __future__ import annotations

import io
from typing import Any

from ..core.artifacts import RemoteArtifactSpec

# Rust runtime callback exports.
from ..core.protocol import (
    CACHE_POLICIES as CACHE_POLICIES,
)
from ..core.protocol import (
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    ClipSelection,
    _parse_env_positive_int,
    validate_cleanup_capability,
)
from ..core.protocol import (
    PEER_MEMORY_POLICIES as PEER_MEMORY_POLICIES,
)
from ..core.result_cache import ByteResultCache
from ..defaults import default_value

LATENT_ARTIFACT = RemoteArtifactSpec("latent", "save_latent", "latents", ".latent")
MINIMAX_CLIP_BASENAME = default_value("minimax_clip", "")
MINIMAX_VIDEO_VAE_BASENAME = default_value("minimax_video_vae", "")
MINIMAX_AUDIO_VAE_BASENAME = default_value("minimax_audio_vae", "")


def _clean_model_name(value: Any, field_name: str) -> str:
    from .._native import clean_model_name_native

    return clean_model_name_native(value, "Remote MiniMax H3", field_name, True)


MINIMAX_CLIP_CHOICES = [MINIMAX_CLIP_BASENAME]
MINIMAX_VIDEO_VAE_CHOICES = [MINIMAX_VIDEO_VAE_BASENAME]
MINIMAX_AUDIO_VAE_CHOICES = [MINIMAX_AUDIO_VAE_BASENAME]
VAE_NAME_OVERRIDE_SOCKET = "*"
_DEFAULT_MINIMAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
_MINIMAX_RESULT_CACHE = ByteResultCache(
    max_entries=4,
    max_bytes=_parse_env_positive_int("REMOTE_MINIMAX_CACHE_BYTES", _DEFAULT_MINIMAX_CACHE_BYTES),
)


def validate_peer_memory_policy(peer_memory_policy: str) -> str:
    from .._native import minimax_policy_native

    return minimax_policy_native(peer_memory_policy, False)


def validate_cache_policy(cache_policy: str) -> str:
    from .._native import minimax_policy_native

    return minimax_policy_native(cache_policy, True)


def get_cached_minimax_outputs(
    cache_key: tuple[Any, ...],
) -> tuple[bytes, bytes] | None:
    from .._native import minimax_cached_native

    return minimax_cached_native(_MINIMAX_RESULT_CACHE, cache_key)


def put_cached_minimax_outputs(cache_key: tuple[Any, ...], conditioning: bytes, latent: bytes) -> None:
    _MINIMAX_RESULT_CACHE.put(cache_key, (conditioning, latent))


def clear_minimax_result_cache() -> None:
    _MINIMAX_RESULT_CACHE.clear()


def _clean_vae_name(value: Any) -> str:
    return _clean_model_name(value, "VAE name")


def validate_minimax_loader_inputs(
    *,
    clip_name: Any,
    clip_type: Any,
    device: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
    cache_policy: str = DEFAULT_CACHE_POLICY,
    vae_names: tuple[Any, ...] = (),
) -> bool | str:
    from .._native import encoder_local_inputs_native

    return encoder_local_inputs_native("minimax", locals())


def resolve_vae_name(
    widget_value: str,
    connected_value: Any | None = None,
    *,
    input_name: str | None = None,
    unique_id: Any = None,
    extra_pnginfo: Any = None,
) -> str:
    from .._native import resolve_vae_name_native

    return resolve_vae_name_native(widget_value, connected_value, input_name, unique_id, extra_pnginfo)


def add_release_cleanup(workflow: dict[str, Any], peer_memory_policy: str) -> None:
    from .._native import add_release_cleanup_native

    add_release_cleanup_native(workflow, peer_memory_policy)


def validate_release_cleanup_capability(api: Any, peer_memory_policy: str) -> None:
    from .._native import release_capability_native

    release_capability_native(api, peer_memory_policy, validate_cleanup_capability)


def validate_peer_inputs(api: Any, clip: ClipSelection, *vae_names: str) -> None:
    import sys

    from .._native import peer_loaders_native

    peer_loaders_native("minimax", sys.modules[__name__], api, clip, vae_names)


def decode_remote_outputs(conditioning: bytes, latent: bytes) -> tuple[Any, dict[str, Any]]:
    """Restore the CPU-only artifacts while preserving MiniMax NestedTensor state."""
    import torch

    state = torch.load(io.BytesIO(latent), map_location="cpu", weights_only=False)
    from .._native import validate_latent_artifact_native

    validate_latent_artifact_native(state, "Remote MiniMax H3")
    return torch.load(io.BytesIO(conditioning), map_location="cpu", weights_only=False), state
