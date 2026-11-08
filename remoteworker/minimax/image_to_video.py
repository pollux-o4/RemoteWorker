"""Remote adapter for the official MiniMax H3 image/text-to-video encoder."""

from __future__ import annotations

import hashlib
import uuid as uuid
from typing import Any

from .._native import workflow_build

# Rust runtime callback exports.
from ..core.artifacts import REMOTE_CONDITIONING_ARTIFACT
from ..core.execution import (
    RequestInputs as RequestInputs,
)
from ..core.execution import (
    completed_artifacts as completed_artifacts,
)
from ..core.media import encode_image_frame_png
from ..core.protocol import CACHE_POLICIES, DEFAULT_CACHE_POLICY, ClipSelection
from ..core.transport import HttpComfyApi as HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from ..preparation.capabilities import Preparation
from .common import (
    DEFAULT_PEER_MEMORY_POLICY,
    MINIMAX_CLIP_CHOICES,
    MINIMAX_VIDEO_VAE_CHOICES,
    PEER_MEMORY_POLICIES,
    VAE_NAME_OVERRIDE_SOCKET,
    validate_minimax_loader_inputs,
    validate_peer_memory_policy,
)
from .common import (
    LATENT_ARTIFACT as LATENT_ARTIFACT,
)
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

CONDITIONING_ARTIFACT = REMOTE_CONDITIONING_ARTIFACT


def _single_frame(image: Any, name: str) -> Any:
    from .._native import single_frame_native

    return single_frame_native(image, name)


def _frame_fingerprint(image: Any | None, name: str) -> str | None:
    if image is None:
        return None
    return hashlib.sha256(encode_image_frame_png(_single_frame(image, name))).hexdigest()


def _i2v_cache_key(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame: Any | None,
    last_frame: Any | None,
    server_url: str,
    peer_memory_policy: str,
) -> tuple[Any, ...]:
    from .._native import cache_identity_native

    validate_peer_memory_policy(peer_memory_policy)
    return cache_identity_native(
        "minimax-i2v",
        dict(
            clip=clip,
            vae_name=vae_name,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            first_frame=_frame_fingerprint(first_frame, "first_frame"),
            last_frame=_frame_fingerprint(last_frame, "last_frame"),
            server_url=server_url,
        ),
    )


def build_workflow(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame: str | None,
    last_frame: str | None,
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    return workflow_build("minimax-i2v", locals())


def execute(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame: Any | None,
    last_frame: Any | None,
    request_id: str,
    api: Any,
    deserialize: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> tuple[Any, Any]:
    import sys

    from .._native import encoder_execute_native

    return encoder_execute_native("minimax-i2v", sys.modules[__name__], locals())


class RemoteMiniMaxH3ImageToVideo:
    """Run peer's MiniMaxH3ImageToVideo encoder and return caller-ready outputs."""

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
                "width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 16384, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384, "step": 32},
                ),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 600, "min": 5, "max": 1800}),
            },
            "optional": {
                "vae_name_override": (VAE_NAME_OVERRIDE_SOCKET, {"forceInput": True}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "peer_memory_policy": (
                    list(PEER_MEMORY_POLICIES),
                    {"default": DEFAULT_PEER_MEMORY_POLICY},
                ),
                "cache_policy": (
                    list(CACHE_POLICIES),
                    {"default": DEFAULT_CACHE_POLICY},
                ),
            },
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
        peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
        cache_policy: str = DEFAULT_CACHE_POLICY,
    ) -> bool | str:
        return validate_minimax_loader_inputs(
            clip_name=clip_name,
            clip_type=clip_type,
            device=device,
            peer_memory_policy=peer_memory_policy,
            cache_policy=cache_policy,
            vae_names=(vae_name,),
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
        width,
        height,
        length,
        server_url=DEFAULT_SERVER_URL,
        timeout_seconds=600,
        peer_memory_policy=DEFAULT_PEER_MEMORY_POLICY,
        cache_policy=DEFAULT_CACHE_POLICY,
        first_frame=None,
        last_frame=None,
        vae_name_override=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        import sys

        from .._native import encoder_run_native

        return encoder_run_native("minimax-i2v", sys.modules[__name__], locals())


NODE_CLASS_MAPPINGS = {"RemoteMiniMaxH3ImageToVideo": RemoteMiniMaxH3ImageToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteMiniMaxH3ImageToVideo": "Remote MiniMax H3 Image to Video"}
