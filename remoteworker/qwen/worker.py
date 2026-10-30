"""Peer-side Qwen Image 2.1 encoder used by the narrow remote adapter.

The stock node always runs reference images through both positive and negative
branches.  This worker preserves the stock output contract while making that
second vision pass an explicit opt-in for workloads that do not use a negative
prompt.
"""

from __future__ import annotations

from typing import Any


class RemoteQwenImage21Encode:
    """peer worker: emit Qwen 2.1 positive, negative, and matching latent."""
    DEV_ONLY = True

    CATEGORY = "conditioning/remote/internal"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 4096, "step": 32},
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "negative_vision": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "The negative prompt is always encoded. Enable only when the negative branch also needs the reference image's visual context.",
                    },
                ),
                **{f"image_{index}": ("IMAGE",) for index in range(1, 17)},
            },
        }

    def encode(
        self,
        clip: Any,
        prompt: str,
        negative_prompt: str,
        resolution: int = 1024,
        vae: Any = None,
        negative_vision: bool = False,
        **images: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        # Keep module registration importable outside a live ComfyUI process;
        # the real runtime dependencies are only needed when this peer worker
        # actually executes.
        from .._native import qwen_encode_native

        return qwen_encode_native(
            clip, prompt, negative_prompt, resolution, vae, negative_vision, images
        )


NODE_CLASS_MAPPINGS = {"RemoteQwenImage21Encode": RemoteQwenImage21Encode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteQwenImage21Encode": "Remote Qwen Image 2.1 Encode (Worker)"
}
