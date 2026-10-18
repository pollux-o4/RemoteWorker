"""Peer-side Qwen Image 2.1 encoder used by the narrow remote adapter.

The stock node always runs reference images through both positive and negative
branches.  This worker preserves the stock output contract while making that
second vision pass an explicit opt-in for workloads that do not use a negative
prompt.
"""
from __future__ import annotations

import math
from typing import Any


class RemoteQwenImage21Encode:
    """peer worker: emit Qwen 2.1 positive, negative, and matching latent."""

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
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32}),
            },
            "optional": {
                "vae": ("VAE",),
                "negative_vision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "The negative prompt is always encoded. Enable only when the negative branch also needs the reference image's visual context.",
                }),
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
        import comfy.model_management
        import comfy.utils
        import node_helpers
        import torch

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Remote Qwen Image 2.1 requires a non-empty prompt")
        if not isinstance(negative_prompt, str):
            raise ValueError("Remote Qwen Image 2.1 negative_prompt must be a string")
        if not isinstance(resolution, int) or isinstance(resolution, bool) or not 0 <= resolution <= 4096:
            raise ValueError("Remote Qwen Image 2.1 resolution must be 0 to 4096")

        ref_latents: list[Any] = []
        images_vl: list[Any] = []
        latent_width = latent_height = resolution or 1024
        for name in sorted(images, key=lambda value: int(value.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                width = round(samples.shape[3] / 32) * 32
                height = round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            resized = image[:1] if (width, height) == (samples.shape[3], samples.shape[2]) else (
                comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            )
            if not images_vl:
                latent_width, latent_height = width, height
            rgb = resized[:, :, :, :3]
            if resized.shape[-1] > 3:
                rgb = rgb * resized[:, :, :, 3:] + (1.0 - resized[:, :, :, 3:])
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(resized))

        keep_vision = not ref_latents
        positive_tokens = clip.tokenize(prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
        negative_tokens = clip.tokenize(
            negative_prompt,
            images=images_vl if negative_vision else [],
            keep_vision=keep_vision,
            prevent_empty_text=True,
        )
        positive = clip.encode_from_tokens_scheduled(positive_tokens)
        negative = clip.encode_from_tokens_scheduled(negative_tokens)
        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros(
            [1, 64, latent_height // 16, latent_width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        return positive, negative, {"samples": latent}


NODE_CLASS_MAPPINGS = {"RemoteQwenImage21Encode": RemoteQwenImage21Encode}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteQwenImage21Encode": "Remote Qwen Image 2.1 Encode (Worker)"}
