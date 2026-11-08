"""Peer data contracts, option validation and memory/cache policy."""

from __future__ import annotations

import os
from typing import NamedTuple

from .._native import (
    build_release_cleanup_node as build_release_cleanup_node,
)
from .._native import (
    is_valid_request_id as is_valid_request_id,
)
from .._native import (
    parse_clip_loader_options as parse_clip_loader_options,
)
from .._native import (
    parse_vae_loader_options as parse_vae_loader_options,
)
from .._native import (
    resolve_peer_model_name as resolve_peer_model_name,
)
from .._native import (
    validate_cleanup_capability as validate_cleanup_capability,
)
from .._native import (
    validate_clip_selection_native,
)
from .._native import (
    validate_policy as _validate_policy,
)
from .._native import (
    validate_server_url as _validate_server_url,
)

DEFAULT_PEER_MEMORY_POLICY = "release_after_result"
PEER_MEMORY_POLICIES = (DEFAULT_PEER_MEMORY_POLICY, "keep")
DEFAULT_CACHE_POLICY = "reuse_same_inputs"
CACHE_POLICIES = (DEFAULT_CACHE_POLICY, "always_refresh")


def _parse_env_positive_int(env_var: str, default: int) -> int:
    from .._native import positive_env

    return positive_env(os.environ.get(env_var), default)


class ClipSelection(NamedTuple):
    """The same three values exposed by ComfyUI's CLIPLoader."""

    clip_name: str
    clip_type: str
    device: str


__all__ = ["_validate_policy", "_validate_server_url"]


def _validate_clip_selection(clip: ClipSelection) -> None:
    validate_clip_selection_native(clip, ClipSelection)
