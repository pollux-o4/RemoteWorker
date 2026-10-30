"""Remote peer text-generation adapter returning an caller-ready STRING prompt."""

from __future__ import annotations

# Rust runtime callback exports.
import uuid as uuid
from ..core.artifacts import PROMPT_ARTIFACT as PROMPT_ARTIFACT
from ..core.execution import (
    RequestInputs as RequestInputs,
    completed_artifacts as completed_artifacts,
)
from ..core.media import encode_image_frame_png as encode_image_frame_png
from ..prompt.transport import PromptComfyApi as PromptComfyApi

import re
from typing import Any

from ..defaults import DEFAULT_SERVER_URL
from .profiles import DEFAULT_PROFILE, PromptEnhancerProfile

_MAX_PROMPT_BYTES = 128 * 1024
_DYNAMIC_MEDIA_NAME = re.compile(r"^(image|video)_([0-9]+)$")
_MAX_IMAGE_REFERENCES = 8
# Match MiniMax H3 Ref2VA's official ref_video_0..2 contract.  The tokenizer
# itself preserves each batch as an independent <Video N> reference.
_MAX_VIDEO_REFERENCES = 3
_MAX_VIDEO_FRAMES = 32


def build_generation_prompt(
    prompt: str, system_prompt: str = DEFAULT_PROFILE.system_prompt
) -> str:
    from .._native import generated_prompt

    return generated_prompt(prompt, system_prompt)


def build_workflow(
    profile: PromptEnhancerProfile,
    prompt: str,
    request_id: str,
    *,
    image_refs: list[str] | None = None,
    video_refs: list[list[str]] | None = None,
) -> dict[str, Any]:
    from .._native import workflow_build
    from dataclasses import asdict

    if not isinstance(profile, PromptEnhancerProfile):
        raise ValueError("Remote Prompt Enhancer profile is invalid")
    return workflow_build(
        "prompt",
        dict(
            profile=asdict(profile),
            prompt=prompt,
            request_id=request_id,
            image_refs=image_refs,
            video_refs=video_refs,
        ),
    )


def decode_generated_prompt(data: bytes) -> str:
    from .._native import decode_generated_prompt_native

    return decode_generated_prompt_native(data)


def _ordered_batches(kind: str, media: dict[str, Any], maximum: int) -> list[Any]:
    from .._native import ordered_images_native

    return ordered_images_native(kind, media, True, maximum, True)


def _ordered_media(kind: str, media: dict[str, Any], maximum: int) -> list[Any]:
    return [
        frame for batch in _ordered_batches(kind, media, maximum) for frame in batch
    ]


def execute(
    *,
    profile_id: str,
    prompt: str,
    request_id: str,
    api: Any,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> str:
    from .._native import encoder_execute_native
    import sys

    return encoder_execute_native("prompt", sys.modules[__name__], locals())


class DynamicPromptMedia(dict[str, tuple[str]]):
    """Declare autogrow image/video sockets; profile policy validates their use."""

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and _DYNAMIC_MEDIA_NAME.fullmatch(key) is not None

    def __getitem__(self, key: str) -> tuple[str]:
        if key not in self:
            raise KeyError(key)
        return ("IMAGE",)


class RemotePromptEnhancer:
    """Execute an peer-owned enhancement profile and emit plain text for any graph."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "profile_id": ([DEFAULT_PROFILE.id],),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 1800}),
            },
            "optional": DynamicPromptMedia(),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote"

    @classmethod
    def VALIDATE_INPUTS(cls, profile_id: Any, **_kwargs: Any) -> bool | str:
        from .._native import encoder_local_inputs_native

        return encoder_local_inputs_native("prompt", locals())

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        return float("nan")

    def run(
        self,
        prompt: str,
        profile_id: str,
        server_url: str = DEFAULT_SERVER_URL,
        timeout_seconds: int = 180,
        **media: Any,
    ) -> tuple[str]:
        from .._native import encoder_run_native
        import sys

        return encoder_run_native("prompt", sys.modules[__name__], locals())


class RemotePromptGenerate:
    """peer-only worker: turn a system prompt, raw request, and visual refs into text."""
    DEV_ONLY = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
                "system_prompt": ("STRING", {"multiline": True}),
                "media_strategy": (["qwen-image", "minimax-ref2va"],),
                "max_length": ("INT", {"default": 256, "min": 16, "max": 2048}),
            },
            "optional": {
                "image": ("IMAGE",),
                **{
                    f"video_{index}": ("IMAGE",)
                    for index in range(_MAX_VIDEO_REFERENCES)
                },
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote/internal"

    def run(
        self,
        clip: Any,
        prompt: str,
        system_prompt: str,
        media_strategy: str,
        max_length: int,
        image: Any = None,
        **media: Any,
    ) -> tuple[str]:
        from .._native import prompt_generate_native
        import sys

        return prompt_generate_native(
            sys.modules[__name__],
            clip,
            prompt,
            system_prompt,
            media_strategy,
            max_length,
            image,
            media,
        )


NODE_CLASS_MAPPINGS = {
    "RemotePromptEnhancer": RemotePromptEnhancer,
    "RemotePromptGenerate": RemotePromptGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotePromptEnhancer": "Remote Prompt Enhancer",
    "RemotePromptGenerate": "Remote Prompt Generate (internal)",
}
