"""Peer data contracts, option validation and memory/cache policy."""

from __future__ import annotations

import os
import re
from typing import Any, NamedTuple
from urllib.parse import urlsplit

DEFAULT_PEER_MEMORY_POLICY = "release_after_result"
PEER_MEMORY_POLICIES = (DEFAULT_PEER_MEMORY_POLICY, "keep")
DEFAULT_CACHE_POLICY = "reuse_same_inputs"
CACHE_POLICIES = (DEFAULT_CACHE_POLICY, "always_refresh")


def _parse_env_positive_int(env_var: str, default: int) -> int:
    """Parse a positive integer environment variable safely; fallback to default on invalid values."""
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
        return val if val > 0 else default
    except ValueError:
        return default


class ClipSelection(NamedTuple):
    """The same three values exposed by ComfyUI's CLIPLoader."""

    clip_name: str
    clip_type: str
    device: str


def parse_clip_loader_options(object_info: dict[str, Any]) -> dict[str, list[str]]:
    """Extract the peer CLIPLoader choices into the Remote Conditioning contract."""
    try:
        inputs = object_info["CLIPLoader"]["input"]
        fields = {
            "clip_name": inputs["required"]["clip_name"][0],
            "clip_type": inputs["required"]["type"][0],
            "device": inputs["optional"]["device"][0],
        }
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("Remote Conditioning CLIPLoader object_info is malformed") from exc
    parsed: dict[str, list[str]] = {}
    for name, values in fields.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise RuntimeError(f"Remote Conditioning CLIPLoader {name} options are invalid")
        parsed[name] = list(dict.fromkeys(values))
    return parsed


def parse_vae_loader_options(object_info: dict[str, Any]) -> dict[str, list[str]]:
    """Extract the peer VAELoader choices without resolving local model paths."""
    try:
        values = object_info["VAELoader"]["input"]["required"]["vae_name"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("Remote Conditioning VAELoader object_info is malformed") from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise RuntimeError("Remote Conditioning VAELoader vae_name options are invalid")
    return {"vae_name": list(dict.fromkeys(values))}


def _validate_clip_selection(clip: ClipSelection) -> None:
    if not isinstance(clip, ClipSelection) or any(not isinstance(value, str) or not value.strip() for value in clip):
        raise ValueError("Remote Conditioning CLIP selection requires non-empty clip_name, clip_type, and device")
    if any("\x00" in value for value in clip):
        raise ValueError("Remote Conditioning CLIP selection contains a null byte")


def _validate_policy(value: Any, allowed: tuple[str, ...], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"Remote Conditioning {field_name} must be one of: {choices}")
    return value


def _validate_server_url(server_url: str) -> str:
    parsed = urlsplit(server_url) if isinstance(server_url, str) else None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Remote Conditioning server_url must be an absolute HTTP(S) origin")
    return server_url.rstrip("/")


def build_release_cleanup_node(anything: list[Any]) -> dict[str, Any]:
    """Build the single supported peer cleanup payload for release_after_result."""
    return {
        "class_type": "VRAMCleanup",
        "inputs": {
            "anything": anything,
            "offload_model": True,
            "offload_cache": True,
            "reset_execution_cache": False,
        },
    }


def validate_cleanup_capability(node_info: dict[str, Any]) -> None:
    """Validate only the shared VRAMCleanup inputs needed by release policy."""
    try:
        inputs = node_info["input"]
        required = inputs["required"]
        optional = inputs["optional"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Remote Conditioning release_after_result requires a valid VRAMCleanup capability") from exc
    if not isinstance(required, dict) or not isinstance(optional, dict):
        raise RuntimeError("Remote Conditioning VRAMCleanup capability has malformed inputs")
    cleanup_inputs = set(required) | set(optional)
    expected_inputs = {"offload_model", "offload_cache", "reset_execution_cache"}
    if not expected_inputs.issubset(cleanup_inputs) or "anything" not in optional:
        raise RuntimeError(
            "Remote Conditioning release_after_result requires VRAMCleanup inputs "
            "offload_model, offload_cache, reset_execution_cache, and optional anything"
        )


_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def is_valid_request_id(value: Any) -> bool:
    """The same request key is used by uploads, workflows and cleanup routes."""
    return isinstance(value, str) and _REQUEST_ID.fullmatch(value) is not None
