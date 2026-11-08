"""Remote MiniMax H3 ref2va adapter: peer encodes; caller samples."""

from __future__ import annotations

import uuid as uuid
from typing import Any

# Rust runtime callback exports.
from ..core.artifacts import REMOTE_CONDITIONING_ARTIFACT
from ..core.execution import (
    RequestInputs as RequestInputs,
)
from ..core.execution import (
    completed_artifacts as completed_artifacts,
)
from ..core.media import encode_image_frame_png as encode_image_frame_png
from ..core.protocol import DEFAULT_CACHE_POLICY, ClipSelection
from ..core.transport import HttpComfyApi as HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from ..preparation.capabilities import Preparation
from .common import (
    DEFAULT_PEER_MEMORY_POLICY,
    MINIMAX_AUDIO_VAE_CHOICES,
    MINIMAX_CLIP_CHOICES,
    MINIMAX_VIDEO_VAE_CHOICES,
    validate_minimax_loader_inputs,
    validate_peer_memory_policy,
)
from .common import LATENT_ARTIFACT as LATENT_ARTIFACT
from .common import (
    decode_remote_outputs as decode_remote_outputs,
)
from .common import (
    get_cached_minimax_outputs as get_cached_minimax_outputs,
)
from .common import (
    put_cached_minimax_outputs as put_cached_minimax_outputs,
)
from .common import (
    resolve_vae_name as resolve_vae_name,
)
from .common import (
    validate_cache_policy as validate_cache_policy,
)
from .common import (
    validate_peer_inputs as validate_peer_inputs,
)
from .common import (
    validate_release_cleanup_capability as validate_release_cleanup_capability,
)
from .media import (
    DynamicReferenceAndVaeInputs,
    _audio_fingerprint,
    _image_batch_fingerprint,
)
from .media import (
    _ordered_audio as _ordered_audio,
)
from .media import (
    _ordered_images as _ordered_images,
)
from .media import (
    encode_audio_wav as encode_audio_wav,
)
from .reference_workflow import build_workflow as build_workflow

CONDITIONING_ARTIFACT = REMOTE_CONDITIONING_ARTIFACT


def _reference_cache_key(
    *,
    clip: ClipSelection,
    vae_name: str,
    audio_vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    images: list[Any],
    videos: list[Any],
    video_audios: list[Any],
    audios: list[Any],
    server_url: str,
    peer_memory_policy: str,
) -> tuple[Any, ...]:
    from .._native import cache_identity_native

    validate_peer_memory_policy(peer_memory_policy)
    return cache_identity_native(
        "minimax-reference",
        dict(
            clip=clip,
            vae_name=vae_name,
            audio_vae_name=audio_vae_name,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            images=tuple(_image_batch_fingerprint(image) for image in images),
            videos=tuple(_image_batch_fingerprint(video) for video in videos),
            video_audios=tuple(_audio_fingerprint(audio) for audio in video_audios),
            audios=tuple(_audio_fingerprint(audio) for audio in audios),
            server_url=server_url,
        ),
    )


def validate_ref2va_capability(capability: dict[str, Any]) -> None:
    from .._native import workflow_validate

    workflow_validate("minimax-reference", capability, "", 0)


def execute(
    *,
    clip: ClipSelection,
    vae_name: str,
    audio_vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    images: list[Any],
    videos: list[Any],
    video_audios: list[Any],
    audios: list[Any],
    request_id: str,
    api: Any,
    deserialize: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> tuple[Any, Any]:
    import sys

    from .._native import encoder_execute_native

    return encoder_execute_native("minimax-reference", sys.modules[__name__], locals())


class RemoteMiniMaxH3ReferenceEncode:
    """Mirror MiniMaxH3ReferenceToVideo inputs and return remote outputs locally."""

    REMOTE_PREPARATION = Preparation("encoder")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "clip_name": (MINIMAX_CLIP_CHOICES,),
                "clip_type": (["minimax"],),
                "device": (["default"],),
                "vae_name": (
                    MINIMAX_VIDEO_VAE_CHOICES,
                    {"default": MINIMAX_VIDEO_VAE_CHOICES[0]},
                ),
                "audio_vae_name": (
                    MINIMAX_AUDIO_VAE_CHOICES,
                    {"default": MINIMAX_AUDIO_VAE_CHOICES[0]},
                ),
                "width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 16384, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384, "step": 32},
                ),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"],),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 600, "min": 5, "max": 1800}),
            },
            "optional": DynamicReferenceAndVaeInputs(),
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "conditioning/remote"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        clip_name: Any,
        clip_type: Any,
        device: Any,
        vae_name: Any,
        audio_vae_name: Any,
        peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
        cache_policy: str = DEFAULT_CACHE_POLICY,
    ) -> bool | str:
        return validate_minimax_loader_inputs(
            clip_name=clip_name,
            clip_type=clip_type,
            device=device,
            peer_memory_policy=peer_memory_policy,
            cache_policy=cache_policy,
            vae_names=(vae_name, audio_vae_name),
        )

    @classmethod
    def IS_CHANGED(cls, cache_policy: str = DEFAULT_CACHE_POLICY, **_kwargs: Any) -> float:
        from .._native import cache_fingerprint_native

        return cache_fingerprint_native(cache_policy, "Remote MiniMax H3")

    def run(
        self,
        prompt,
        clip_name,
        clip_type,
        device,
        vae_name,
        audio_vae_name,
        width,
        height,
        length,
        ref_image_size,
        server_url=DEFAULT_SERVER_URL,
        timeout_seconds=600,
        peer_memory_policy=DEFAULT_PEER_MEMORY_POLICY,
        cache_policy=DEFAULT_CACHE_POLICY,
        unique_id=None,
        extra_pnginfo=None,
        **media,
    ):
        import sys

        from .._native import encoder_run_native

        return encoder_run_native("minimax-reference", sys.modules[__name__], locals())


NODE_CLASS_MAPPINGS = {"RemoteMiniMaxH3ReferenceEncode": RemoteMiniMaxH3ReferenceEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteMiniMaxH3ReferenceEncode": "Remote MiniMax H3 Reference Encode"}
