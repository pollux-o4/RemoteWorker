"""Pure CLIP and grounded-conditioning workflow strategies."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, NamedTuple

from ..core.artifacts import CONDITIONING_ARTIFACT, extract_artifact_paths
from ..core.protocol import (
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    ClipSelection,
    _validate_clip_selection,
    _validate_policy,
    build_release_cleanup_node,
    is_valid_request_id,
)

_ROOT = Path(__file__).resolve().parents[2]
_TEXT_MARKER = "__REMOTE_CONDITIONING_TEXT__"
_CLIP_NAME_MARKER = "__REMOTE_CONDITIONING_CLIP_NAME__"
_CLIP_TYPE_MARKER = "__REMOTE_CONDITIONING_CLIP_TYPE__"
_CLIP_DEVICE_MARKER = "__REMOTE_CONDITIONING_CLIP_DEVICE__"
_REQUEST_MARKER = "__REMOTE_CONDITIONING_REQUEST_ID__"
_TEMPLATE_PATH = _ROOT / "templates" / "remote-conditioning" / "clip-conditioning.api.json"


class ConditioningStrategySpec(NamedTuple):
    name: str
    minimum_images: int
    maximum_images: int | None
    remote_node: str


STRATEGIES = {
    "text": ConditioningStrategySpec("text", 0, 0, "CLIPTextEncode"),
    "krea2-grounded": ConditioningStrategySpec("krea2-grounded", 1, 2, "Krea2EditGroundedEncode"),
}


def build_workflow(
    clip: ClipSelection,
    text: str,
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    """Build one API-format CLIPLoader → encode → save workflow."""
    _validate_clip_selection(clip)
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Remote Conditioning prompt must be a non-empty string")
    if not is_valid_request_id(request_id):
        raise ValueError("Remote Conditioning request_id must be 1-64 URL-safe characters")
    try:
        template = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Remote Conditioning template is invalid: {_TEMPLATE_PATH}") from exc
    workflow = copy.deepcopy(template)
    try:
        inputs = workflow["clip"]["inputs"]
        encoded_text = workflow["encode"]["inputs"]["text"]
        filename_prefix = workflow["save"]["inputs"]["filename_prefix"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Remote Conditioning template is malformed") from exc
    if (
        inputs.get("clip_name") != _CLIP_NAME_MARKER
        or inputs.get("type") != _CLIP_TYPE_MARKER
        or inputs.get("device") != _CLIP_DEVICE_MARKER
        or encoded_text != _TEXT_MARKER
        or filename_prefix.count(_REQUEST_MARKER) != 1
    ):
        raise RuntimeError("Remote Conditioning template has unexpected injection markers")
    inputs["clip_name"], inputs["type"], inputs["device"] = clip
    workflow["encode"]["inputs"]["text"] = text
    workflow["save"]["inputs"]["filename_prefix"] = filename_prefix.replace(_REQUEST_MARKER, request_id)
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        workflow["cleanup"] = build_release_cleanup_node(["save", 0])
    else:
        workflow.pop("cleanup", None)
    return workflow


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
    """Map ordered attachments into one registered conditioning Strategy."""
    strategy = STRATEGIES.get(strategy_name)
    if strategy is None:
        raise ValueError(f"Remote Conditioning strategy is not registered: {strategy_name!r}")
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    image_count = len(image_refs)
    maximum = strategy.maximum_images
    if image_count < strategy.minimum_images or (maximum is not None and image_count > maximum):
        maximum_text = "unbounded" if maximum is None else str(maximum)
        raise ValueError(
            f"Remote Conditioning {strategy.name} requires {strategy.minimum_images} to {maximum_text} images; got {image_count}"
        )
    if strategy_name == "text":
        return build_workflow(clip, prompt, request_id, peer_memory_policy)
    _validate_clip_selection(clip)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Remote Conditioning prompt must be a non-empty string")
    if not is_valid_request_id(request_id):
        raise ValueError("Remote Conditioning request_id must be 1-64 URL-safe characters")
    if not isinstance(grounding_px, int) or isinstance(grounding_px, bool) or not 0 <= grounding_px <= 4096:
        raise ValueError("Remote Conditioning grounding_px must be 0 to 4096")
    if not isinstance(system_prompt, str):
        raise ValueError("Remote Conditioning system_prompt must be a string")
    if any(not isinstance(ref, str) or not ref.endswith(" [temp]") for ref in image_refs):
        raise ValueError("Remote Conditioning image references must use validated temp uploads")

    workflow: dict[str, Any] = {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip.clip_name,
                "type": clip.clip_type,
                "device": clip.device,
            },
        },
        "encode": {
            "class_type": strategy.remote_node,
            "inputs": {
                "clip": ["clip", 0],
                "prompt": prompt,
                "image": ["image_1", 0],
                "grounding_px": grounding_px,
                "system_prompt": system_prompt,
            },
        },
        "save": {
            "class_type": "SaveConditioning",
            "inputs": {
                "conditioning": ["encode", 0],
                "filename_prefix": f"remote-conditioning/{request_id}/cond",
            },
        },
    }
    for index, image_ref in enumerate(image_refs, start=1):
        workflow[f"image_{index}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_ref},
        }
    if image_count == 2:
        workflow["encode"]["inputs"]["image_b"] = ["image_2", 0]
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        workflow["cleanup"] = build_release_cleanup_node(["save", 0])
    return workflow


def validate_strategy_capability(
    strategy: ConditioningStrategySpec, node_info: dict[str, Any], image_count: int
) -> None:
    """Fail before upload when the peer cannot satisfy the selected Strategy contract."""
    try:
        inputs = node_info["input"]
        required = inputs["required"]
        optional = inputs.get("optional", {})
        outputs = node_info["output"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Remote Conditioning {strategy.remote_node} capability is malformed") from exc
    expected_required = {"clip", "prompt"}
    expected_optional = {"image"} | ({"image_b"} if image_count > 1 else set())
    if (
        not expected_required.issubset(required)
        or not expected_optional.issubset(optional)
        or "CONDITIONING" not in outputs
    ):
        raise RuntimeError(f"Remote Conditioning peer {strategy.remote_node} is incompatible with {strategy.name}")


def extract_artifact_path(history: dict[str, Any], prompt_id: str, request_id: str) -> str:
    """Return the sole request-scoped `.cond` result from a completed history entry."""
    return extract_artifact_paths(
        history,
        prompt_id,
        request_id,
        (CONDITIONING_ARTIFACT,),
    )["conditioning"]
