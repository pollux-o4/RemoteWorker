"""MiniMax reference workflow construction."""

from __future__ import annotations

from typing import Any

from .._native import workflow_build
from ..core.protocol import ClipSelection
from .common import DEFAULT_PEER_MEMORY_POLICY


def build_workflow(
    *,
    clip: ClipSelection,
    vae_name: str,
    audio_vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    image_refs: list[str],
    video_refs: list[list[str]],
    video_audio_refs: list[str],
    audio_refs: list[str],
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    return workflow_build("minimax-reference", locals())
