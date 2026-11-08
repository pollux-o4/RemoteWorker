"""Pure CLIP and grounded-conditioning workflow strategies."""

from __future__ import annotations

from .._native import workflow_build

from pathlib import Path
from typing import Any, NamedTuple

from ..core.artifacts import REMOTE_CONDITIONING_ARTIFACT as CONDITIONING_ARTIFACT
from ..core.artifacts import extract_artifact_paths
from ..core.protocol import (
    DEFAULT_PEER_MEMORY_POLICY,
    ClipSelection,
)

_ROOT = Path(__file__).resolve().parents[2]
_TEXT_MARKER = "__REMOTE_CONDITIONING_TEXT__"
_CLIP_NAME_MARKER = "__REMOTE_CONDITIONING_CLIP_NAME__"
_CLIP_TYPE_MARKER = "__REMOTE_CONDITIONING_CLIP_TYPE__"
_CLIP_DEVICE_MARKER = "__REMOTE_CONDITIONING_CLIP_DEVICE__"
_REQUEST_MARKER = "__REMOTE_CONDITIONING_REQUEST_ID__"
_TEMPLATE_PATH = (
    _ROOT / "templates" / "remote-conditioning" / "clip-conditioning.api.json"
)


class ConditioningStrategySpec(NamedTuple):
    name: str
    minimum_images: int
    maximum_images: int | None
    remote_node: str


STRATEGIES = {
    "text": ConditioningStrategySpec("text", 0, 0, "CLIPTextEncode"),
    "krea2-grounded": ConditioningStrategySpec(
        "krea2-grounded", 1, 2, "Krea2EditGroundedEncode"
    ),
}


def build_workflow(
    clip: ClipSelection,
    text: str,
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    return workflow_build(
        "conditioning",
        dict(
            clip=clip,
            prompt=text,
            request_id=request_id,
            peer_memory_policy=peer_memory_policy,
        ),
        str(_TEMPLATE_PATH),
    )


def build_strategy_workflow(
    *,
    strategy_name: str,
    clip: ClipSelection,
    prompt: str,
    request_id: str,
    image_refs: list[str],
    grounding_px: int,
    system_prompt: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    return workflow_build("conditioning", locals(), str(_TEMPLATE_PATH))


def validate_strategy_capability(
    strategy: ConditioningStrategySpec, node_info: dict[str, Any], image_count: int
) -> None:
    from .._native import workflow_validate

    workflow_validate("conditioning", node_info, strategy.name, image_count)


def extract_artifact_path(
    history: dict[str, Any], prompt_id: str, request_id: str
) -> str:
    """Return the sole request-scoped `.cond` result from a completed history entry."""
    return extract_artifact_paths(
        history,
        prompt_id,
        request_id,
        (CONDITIONING_ARTIFACT,),
    )["conditioning"]
