"""Remote VAE name selector for line-based MiniMax UX.

This node deliberately returns only a model-name string.  It does not resolve a
local path and does not instantiate a local VAE object on the caller runtime.
"""
from __future__ import annotations

from typing import Any

from ..defaults import DEFAULT_SERVER_URL
from ..minimax.common import (
    MINIMAX_AUDIO_VAE_CHOICES,
    MINIMAX_VIDEO_VAE_CHOICES,
    resolve_vae_name,
)


def _default_vae_choices() -> list[str]:
    return list(dict.fromkeys([*MINIMAX_VIDEO_VAE_CHOICES, *MINIMAX_AUDIO_VAE_CHOICES]))


class RemoteVAESelector:
    """Pick one peer VAELoader model name and pass it through a wire."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 30, "min": 5, "max": 1800}),
                "vae_name": (_default_vae_choices(), {"default": _default_vae_choices()[0]}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("vae_name",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote"

    def run(self, server_url: str, timeout_seconds: int, vae_name: str) -> tuple[str]:
        return (resolve_vae_name(vae_name),)


NODE_CLASS_MAPPINGS = {"RemoteVAESelector": RemoteVAESelector}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteVAESelector": "Remote VAE Name Selector"}
