"""Conditioning strategy service, cache and ComfyUI node adapter."""

from __future__ import annotations

import hashlib
import io
import uuid
from typing import Any

from ..core.artifacts import CONDITIONING_ARTIFACT
from ..core.execution import RequestInputs, completed_artifacts
from ..core.media import (
    _MAX_IMAGE_ATTACHMENTS,
    DynamicImageAndPolicyInputs,
    encode_image_frame_png,
    ordered_image_frames,
)
from ..core.protocol import (
    CACHE_POLICIES,
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    ClipSelection,
    _parse_env_positive_int,
    _validate_clip_selection,
    _validate_policy,
    _validate_server_url,
    validate_cleanup_capability,
)
from ..core.result_cache import ByteResultCache
from ..core.transport import HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL, default_value
from .workflow import (
    STRATEGIES,
    build_strategy_workflow,
    build_workflow,
    validate_strategy_capability,
)

_CONDITIONING_BYTES_CACHE = ByteResultCache(
    max_entries=8,
    max_bytes=_parse_env_positive_int("REMOTE_CONDITIONING_CACHE_BYTES", 256 * 1024 * 1024),
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
    _validate_clip_selection(clip)
    _validate_server_url(server_url)
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    return (
        strategy_name,
        clip,
        prompt,
        tuple(_frame_fingerprint(frame) for frame in frames),
        grounding_px,
        system_prompt,
        server_url.rstrip("/"),
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
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        validate_cleanup_capability(api.require_node("VRAMCleanup"))
    workflow = build_workflow(clip, prompt, request_id, peer_memory_policy)
    with completed_artifacts(api, workflow, request_id, (CONDITIONING_ARTIFACT,)) as blobs:
        return blobs["conditioning"]


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
    """Execute text or image-grounded conditioning through one narrow Strategy boundary."""
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    if strategy_name == "text":
        if frames:
            raise ValueError("Remote Conditioning text strategy requires 0 images")
        return execute_bytes(
            clip=clip,
            prompt=prompt,
            request_id=request_id,
            api=api,
            peer_memory_policy=peer_memory_policy,
        )
    strategy = STRATEGIES.get(strategy_name)
    if strategy is None:
        raise ValueError(f"Remote Conditioning strategy is not registered: {strategy_name!r}")
    maximum = strategy.maximum_images
    if len(frames) < strategy.minimum_images or (maximum is not None and len(frames) > maximum):
        maximum_text = "unbounded" if maximum is None else str(maximum)
        raise ValueError(
            f"Remote Conditioning {strategy.name} requires {strategy.minimum_images} to {maximum_text} images; got {len(frames)}"
        )
    validate_strategy_capability(strategy, api.require_node(strategy.remote_node), len(frames))
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        validate_cleanup_capability(api.require_node("VRAMCleanup"))
    encoded = [encode_image_frame_png(frame) for frame in frames]
    refs: list[str] = []
    with RequestInputs(api, request_id) as request:
        for index, png in enumerate(encoded, start=1):
            refs.append(request.upload_image(index=index, png=png))
        workflow = build_strategy_workflow(
            strategy_name=strategy_name,
            clip=clip,
            prompt=prompt,
            request_id=request_id,
            image_refs=refs,
            grounding_px=grounding_px,
            system_prompt=system_prompt,
            peer_memory_policy=peer_memory_policy,
        )
        with completed_artifacts(api, workflow, request_id, (CONDITIONING_ARTIFACT,)) as blobs:
            return blobs["conditioning"]


class RemoteConditioning:
    """ComfyUI master node: prompt plus remote CLIPLoader selection to CONDITIONING."""

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
        for field_name, value in (
            ("remote_text_encoder", remote_text_encoder),
            ("clip_type", clip_type),
            ("device", device),
        ):
            if not isinstance(value, str) or not value.strip():
                return f"Remote Conditioning {field_name} must be a non-empty string"
        try:
            _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
            _validate_policy(cache_policy, CACHE_POLICIES, "cache_policy")
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, cache_policy: str = DEFAULT_CACHE_POLICY, **_kwargs: Any) -> float:
        _validate_policy(cache_policy, CACHE_POLICIES, "cache_policy")
        return float("nan") if cache_policy == "always_refresh" else 0

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
        **images: Any,
    ) -> tuple[Any]:
        try:
            import torch

            frames = [] if conditioning_strategy == "text" else ordered_image_frames(image_count, **images)
            clip = ClipSelection(remote_text_encoder, clip_type, device)
            cache_key = None
            if cache_policy == DEFAULT_CACHE_POLICY:
                cache_key = _conditioning_cache_key(
                    strategy_name=conditioning_strategy,
                    clip=clip,
                    prompt=prompt,
                    frames=frames,
                    grounding_px=grounding_px,
                    system_prompt=system_prompt,
                    server_url=server_url,
                    peer_memory_policy=peer_memory_policy,
                )
                cached = _get_cached_conditioning_bytes(cache_key)
                if cached is not None:
                    print("[RemoteConditioning] cache hit")
                    return (torch.load(io.BytesIO(cached), map_location="cpu", weights_only=False),)
            else:
                _validate_policy(cache_policy, CACHE_POLICIES, "cache_policy")
            data = execute_strategy_bytes(
                strategy_name=conditioning_strategy,
                clip=clip,
                prompt=prompt,
                request_id=uuid.uuid4().hex,
                frames=frames,
                grounding_px=grounding_px,
                system_prompt=system_prompt,
                api=HttpComfyApi(server_url, timeout_seconds),
                peer_memory_policy=peer_memory_policy,
            )
            if cache_key is not None:
                _put_cached_conditioning_bytes(cache_key, data)
            result = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Remote Conditioning failed: {exc}") from exc
        return (result,)


NODE_CLASS_MAPPINGS = {"RemoteConditioning": RemoteConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteConditioning": "Remote Conditioning"}

ZIT_DEFAULT_CLIP = ClipSelection(default_value("conditioning_clip", ""), default_value("conditioning_clip_type", "lumina2"), "default")
