"""Shared peer validation and artifact decoding for MiniMax H3 remote strategies."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.artifacts import RemoteArtifactSpec
from ..core.protocol import (
    CACHE_POLICIES,
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    ClipSelection,
    _parse_env_positive_int,
    build_release_cleanup_node,
    validate_cleanup_capability,
)
from ..core.result_cache import ByteResultCache
from ..defaults import default_value

LATENT_ARTIFACT = RemoteArtifactSpec("latent", "save_latent", "latents", ".latent")
MINIMAX_CLIP_BASENAME = default_value("minimax_clip", "")
MINIMAX_VIDEO_VAE_BASENAME = default_value("minimax_video_vae", "")
MINIMAX_AUDIO_VAE_BASENAME = default_value("minimax_audio_vae", "")


def _clean_model_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"Remote MiniMax H3 {field_name} must be a non-empty string")
    return value.strip()


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
    if not isinstance(peer_memory_policy, str) or peer_memory_policy not in PEER_MEMORY_POLICIES:
        choices = ", ".join(PEER_MEMORY_POLICIES)
        raise ValueError(f"Remote MiniMax H3 peer_memory_policy must be one of: {choices}")
    return peer_memory_policy


def validate_cache_policy(cache_policy: str) -> str:
    if not isinstance(cache_policy, str) or cache_policy not in CACHE_POLICIES:
        choices = ", ".join(CACHE_POLICIES)
        raise ValueError(f"Remote MiniMax H3 cache_policy must be one of: {choices}")
    return cache_policy


def get_cached_minimax_outputs(cache_key: tuple[Any, ...]) -> tuple[bytes, bytes] | None:
    parts = _MINIMAX_RESULT_CACHE.get(cache_key)
    if parts is None:
        return None
    if len(parts) != 2:
        _MINIMAX_RESULT_CACHE.clear()
        return None
    return parts[0], parts[1]


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
    """Keep local prompt validation flexible; peer validation remains authoritative."""
    try:
        _clean_model_name(clip_name, "clip_name")
        _clean_model_name(clip_type, "clip_type")
        _clean_model_name(device, "device")
        for value in vae_names:
            _clean_vae_name(value)
        validate_peer_memory_policy(peer_memory_policy)
        validate_cache_policy(cache_policy)
    except ValueError as exc:
        return str(exc)
    return True


def _workflow_from_extra_pnginfo(extra_pnginfo: Any) -> Mapping[str, Any] | None:
    if not isinstance(extra_pnginfo, Mapping):
        return None
    workflow = extra_pnginfo.get("workflow")
    if isinstance(workflow, Mapping):
        return workflow
    nested = extra_pnginfo.get("extra_pnginfo")
    if isinstance(nested, Mapping) and isinstance(nested.get("workflow"), Mapping):
        return nested["workflow"]
    return None


def _nodes_by_id(workflow: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if isinstance(node, Mapping) and node.get("id") is not None:
            result[str(node["id"])] = node
    return result


def _link_origin_id(workflow: Mapping[str, Any], link_id: Any) -> str | None:
    links = workflow.get("links")
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        return None
    for link in links:
        if isinstance(link, Mapping):
            if str(link.get("id")) == str(link_id) and link.get("origin_id") is not None:
                return str(link["origin_id"])
        elif isinstance(link, Sequence) and not isinstance(link, (str, bytes)) and len(link) >= 2:
            if str(link[0]) == str(link_id):
                return str(link[1])
    return None


def _connected_source_node(workflow: Mapping[str, Any], unique_id: Any, input_name: str) -> Mapping[str, Any] | None:
    nodes = _nodes_by_id(workflow)
    current = nodes.get(str(unique_id))
    if current is None:
        return None
    inputs = current.get("inputs")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        return None
    for item in inputs:
        if isinstance(item, Mapping) and item.get("name") == input_name and item.get("link") is not None:
            origin_id = _link_origin_id(workflow, item["link"])
            if origin_id is not None:
                return nodes.get(origin_id)
    return None


def _vae_name_from_loader_node(node: Mapping[str, Any]) -> str | None:
    node_type = str(node.get("type") or node.get("class_type") or node.get("title") or "")
    if "VAELoader" not in node_type and "VAE 로드" not in node_type:
        return None
    widgets_values = node.get("widgets_values")
    if isinstance(widgets_values, Sequence) and not isinstance(widgets_values, (str, bytes)) and widgets_values:
        return _clean_vae_name(widgets_values[0])
    return None


def _connected_vae_loader_name(*, input_name: str | None, unique_id: Any, extra_pnginfo: Any) -> str | None:
    if input_name is None or unique_id is None:
        return None
    workflow = _workflow_from_extra_pnginfo(extra_pnginfo)
    if workflow is None:
        return None
    source = _connected_source_node(workflow, unique_id, input_name)
    if source is None:
        return None
    return _vae_name_from_loader_node(source)


def resolve_vae_name(
    widget_value: str,
    connected_value: Any | None = None,
    *,
    input_name: str | None = None,
    unique_id: Any = None,
    extra_pnginfo: Any = None,
) -> str:
    """Resolve VAE override from a remote-name string or a connected VAELoader node."""
    candidate_value = connected_value if connected_value is not None else widget_value
    if isinstance(candidate_value, str):
        return _clean_vae_name(candidate_value)

    candidate = _connected_vae_loader_name(input_name=input_name, unique_id=unique_id, extra_pnginfo=extra_pnginfo)
    if candidate is not None:
        return candidate
    raise ValueError(
        "Remote MiniMax H3 VAE override must be a VAE name string or a directly connected VAELoader; "
        "use Remote VAE Name Selector for peer-only loading"
    )


def add_release_cleanup(workflow: dict[str, Any], peer_memory_policy: str) -> None:
    """Append peer cleanup after both remote artifacts are serialized."""
    validate_peer_memory_policy(peer_memory_policy)
    if peer_memory_policy != DEFAULT_PEER_MEMORY_POLICY:
        return
    try:
        save_latent_inputs = workflow["save_latent"]["inputs"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Remote MiniMax H3 workflow is missing SaveRemoteLatent for cleanup ordering") from exc
    save_latent_inputs["dependency"] = ["save", 0]
    workflow["cleanup"] = build_release_cleanup_node(["save_latent", 0])


def validate_release_cleanup_capability(api: Any, peer_memory_policy: str) -> None:
    validate_peer_memory_policy(peer_memory_policy)
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        validate_cleanup_capability(api.require_node("VRAMCleanup"))


def validate_peer_inputs(api: Any, clip: ClipSelection, *vae_names: str) -> None:
    """Reject unavailable peer loader values before any media upload begins."""
    options = api.clip_loader_options()
    for field, value in zip(("clip_name", "clip_type", "device"), clip):
        if value not in options.get(field, []):
            raise ValueError(f"Remote MiniMax H3 peer does not provide {field}: {value!r}")
    vae_info = api.require_node("VAELoader")
    try:
        vae_options = vae_info["input"]["required"]["vae_name"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Remote MiniMax H3 peer VAELoader capability is malformed") from exc
    if not isinstance(vae_options, list) or any(value not in vae_options for value in vae_names):
        raise ValueError("Remote MiniMax H3 peer does not provide the selected VAE")


def decode_remote_outputs(conditioning: bytes, latent: bytes) -> tuple[Any, dict[str, Any]]:
    """Restore the CPU-only artifacts while preserving MiniMax NestedTensor state."""
    import torch

    state = torch.load(io.BytesIO(latent), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "samples" not in state:
        raise RuntimeError("Remote MiniMax H3 latent artifact is invalid")
    return torch.load(io.BytesIO(conditioning), map_location="cpu", weights_only=False), state
