"""Conditioning strategy service, cache and ComfyUI node adapter."""

from __future__ import annotations

# Rust runtime callback exports.
import uuid as uuid
from ..core.artifacts import CONDITIONING_ARTIFACT as CONDITIONING_ARTIFACT
from ..core.execution import (
    RequestInputs as RequestInputs,
    completed_artifacts as completed_artifacts,
)
from ..core.media import ordered_image_frames as ordered_image_frames
from ..core.protocol import validate_cleanup_capability as validate_cleanup_capability
from ..core.transport import HttpComfyApi as HttpComfyApi
from .workflow import (
    build_strategy_workflow as build_strategy_workflow,
    build_workflow as build_workflow,
    validate_strategy_capability as validate_strategy_capability,
)

from ..preparation.capabilities import Preparation

import hashlib
from typing import Any

from ..core.media import (
    _MAX_IMAGE_ATTACHMENTS,
    DynamicImageAndPolicyInputs,
    encode_image_frame_png,
)
from ..core.protocol import (
    CACHE_POLICIES as CACHE_POLICIES,
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    ClipSelection,
    _parse_env_positive_int,
    _validate_clip_selection,
    _validate_policy,
    _validate_server_url,
)
from ..core.result_cache import ByteResultCache
from ..defaults import DEFAULT_SERVER_URL, default_value
from .workflow import (
    STRATEGIES,
)

_CONDITIONING_BYTES_CACHE = ByteResultCache(
    max_entries=8,
    max_bytes=_parse_env_positive_int(
        "REMOTE_CONDITIONING_CACHE_BYTES", 256 * 1024 * 1024
    ),
)


def _frame_fingerprint(frame: Any) -> str:
    return hashlib.sha256(encode_image_frame_png(frame)).hexdigest()


def _conditioning_cache_key(
    *,
    strategy_name: str,
    clip: ClipSelection,
    prompt: str,
    frames: list[Any],
    grounding_px: int,
    system_prompt: str,
    server_url: str,
    peer_memory_policy: str,
) -> tuple[Any, ...]:
    from .._native import cache_identity_native

    _validate_clip_selection(clip)
    _validate_server_url(server_url)
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    return cache_identity_native(
        "conditioning",
        dict(
            strategy_name=strategy_name,
            clip=clip,
            prompt=prompt,
            frames=tuple(_frame_fingerprint(frame) for frame in frames),
            grounding_px=grounding_px,
            system_prompt=system_prompt,
            server_url=server_url,
        ),
    )


def _get_cached_conditioning_bytes(cache_key: tuple[Any, ...]) -> bytes | None:
    parts = _CONDITIONING_BYTES_CACHE.get(cache_key)
    return None if parts is None else parts[0]


def _put_cached_conditioning_bytes(cache_key: tuple[Any, ...], data: bytes) -> None:
    _CONDITIONING_BYTES_CACHE.put(cache_key, (data,))


def clear_conditioning_cache() -> None:
    _CONDITIONING_BYTES_CACHE.clear()


def execute(
    *,
    clip: ClipSelection,
    prompt: str,
    request_id: str,
    api: Any,
    deserialize: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> Any:
    return deserialize(
        execute_bytes(
            clip=clip,
            prompt=prompt,
            request_id=request_id,
            api=api,
            peer_memory_policy=peer_memory_policy,
        )
    )


def execute_bytes(
    *,
    clip: ClipSelection,
    prompt: str,
    request_id: str,
    api: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> bytes:
    from .._native import encoder_execute_native
    import sys

    return encoder_execute_native("conditioning", sys.modules[__name__], locals())


def execute_strategy(
    *,
    strategy_name: str,
    clip: ClipSelection,
    prompt: str,
    request_id: str,
    frames: list[Any],
    grounding_px: int,
    system_prompt: str,
    api: Any,
    deserialize: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> Any:
    return deserialize(
        execute_strategy_bytes(
            strategy_name=strategy_name,
            clip=clip,
            prompt=prompt,
            request_id=request_id,
            frames=frames,
            grounding_px=grounding_px,
            system_prompt=system_prompt,
            api=api,
            peer_memory_policy=peer_memory_policy,
        )
    )


def execute_strategy_bytes(
    *,
    strategy_name: str,
    clip: ClipSelection,
    prompt: str,
    request_id: str,
    frames: list[Any],
    grounding_px: int,
    system_prompt: str,
    api: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> bytes:
    from .._native import encoder_execute_native
    import sys

    return encoder_execute_native(
        "conditioning-strategy", sys.modules[__name__], locals()
    )


class RemoteConditioning:
    """ComfyUI master node: prompt plus remote CLIPLoader selection to CONDITIONING."""

    REMOTE_PREPARATION = Preparation("encoder")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "conditioning_strategy": (list(STRATEGIES),),
                "remote_text_encoder": (
                    [ZIT_DEFAULT_CLIP.clip_name],
                    {"default": ZIT_DEFAULT_CLIP.clip_name},
                ),
                "clip_type": (
                    [ZIT_DEFAULT_CLIP.clip_type],
                    {"default": ZIT_DEFAULT_CLIP.clip_type},
                ),
                "device": (
                    [ZIT_DEFAULT_CLIP.device],
                    {"default": ZIT_DEFAULT_CLIP.device},
                ),
                "image_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": _MAX_IMAGE_ATTACHMENTS},
                ),
                "grounding_px": (
                    "INT",
                    {"default": 768, "min": 0, "max": 4096, "step": 64},
                ),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 120, "min": 5, "max": 1800}),
            },
            "optional": DynamicImageAndPolicyInputs(),
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        remote_text_encoder: Any,
        clip_type: Any,
        device: Any,
        peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
        cache_policy: str = DEFAULT_CACHE_POLICY,
    ) -> bool | str:
        from .._native import encoder_local_inputs_native

        return encoder_local_inputs_native("conditioning", locals())

    @classmethod
    def IS_CHANGED(
        cls, cache_policy: str = DEFAULT_CACHE_POLICY, **_kwargs: Any
    ) -> float:
        from .._native import cache_fingerprint_native

        return cache_fingerprint_native(cache_policy, "Remote Conditioning")

    def run(
        self,
        prompt: str,
        conditioning_strategy: str,
        remote_text_encoder: str,
        clip_type: str,
        device: str,
        image_count: int,
        grounding_px: int,
        system_prompt: str,
        server_url: str,
        timeout_seconds: int = 120,
        peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
        cache_policy: str = DEFAULT_CACHE_POLICY,
        unique_id=None,
        **images: Any,
    ) -> tuple[Any]:
        from .._native import encoder_run_native
        import sys

        return encoder_run_native("conditioning", sys.modules[__name__], locals())


NODE_CLASS_MAPPINGS = {"RemoteConditioning": RemoteConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteConditioning": "Remote Conditioning"}

ZIT_DEFAULT_CLIP = ClipSelection(
    default_value("conditioning_clip", ""),
    default_value("conditioning_clip_type", "lumina2"),
    "default",
)
