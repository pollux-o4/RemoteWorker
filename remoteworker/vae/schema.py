"""Public VAE node inputs and semantic connection contracts."""

from __future__ import annotations

from .. import _native
from .._native import (
    vae_validate_model_choice as validate_model_choice,
)
from .._native import (
    vae_video_fields as video_fields,
)
from .._native import (
    vae_video_options as video_options,
)
from ..defaults import DEFAULT_SERVER_URL

WARMUP_TYPE = "REMOTE_VAE_WARMUP"
LATENT_TYPE = "REMOTE_VAE_LATENT"
VIDEO_FPS, VIDEO_CRF = _native.VAE_VIDEO_FPS, _native.VAE_VIDEO_CRF
DECODE_DEFAULTS = _native.vae_decode_defaults()


def connection_inputs():
    return {
        "vae_name": (
            [""],
            {
                "default": "",
                "tooltip": "Choose a VAE from the peer model list. No VAE is loaded on this server.",
            },
        ),
        "server_url": (
            "STRING",
            {
                "default": DEFAULT_SERVER_URL,
                "tooltip": "ComfyUI that will load and run the VAE. Use a different server from this workflow.",
            },
        ),
        "timeout_seconds": (
            "INT",
            {
                "default": 600,
                "min": 5,
                "max": 1800,
                "advanced": True,
                "tooltip": "Maximum wait for each peer request, in seconds.",
            },
        ),
    }


def memory_input():
    return (
        ["release_after_result", "keep"],
        {
            "advanced": True,
            "tooltip": "Release peer models after saving the result, or retain them for the next decode.",
        },
    )


def decode_fields():
    return {
        "tiled": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Enable native tiled decoding. H3 video VAE already tiles internally.",
            },
        ),
        "tile_size": (
            "INT",
            {
                "default": DECODE_DEFAULTS["tile_size"],
                "min": 64,
                "max": 4096,
                "step": 32,
                "advanced": True,
            },
        ),
        "overlap": (
            "INT",
            {
                "default": DECODE_DEFAULTS["overlap"],
                "min": 0,
                "max": 4096,
                "advanced": True,
            },
        ),
        "temporal_size": (
            "INT",
            {
                "default": DECODE_DEFAULTS["temporal_size"],
                "min": 8,
                "max": 4096,
                "advanced": True,
            },
        ),
        "temporal_overlap": (
            "INT",
            {
                "default": DECODE_DEFAULTS["temporal_overlap"],
                "min": 4,
                "max": 4096,
                "advanced": True,
            },
        ),
    }


def decode_inputs(kind):
    required = {
        "samples": (
            "LATENT",
            {
                "lazy": True,
                "tooltip": "Sampler output. H3 combined video/audio latents are supported.",
            },
        ),
        **connection_inputs(),
        "memory_policy": memory_input(),
    }
    if kind == "image":
        required.update(decode_fields())
    return {
        "required": required,
        "optional": {
            "warmup": (
                WARMUP_TYPE,
                {"forceInput": True, "tooltip": "Optional manual warmup ticket."},
            ),
            "output_mode": (
                ["shared", "tensor"],
                {
                    "default": "shared",
                    "tooltip": "Both modes return IMAGE/AUDIO. Shared: transfer via temporary shared tensor files. Tensor: HTTP transfer (512 MiB limit).",
                },
            ),
            **({name: value for name, value in video_fields().items() if name != "tiled"} if kind == "image" else {}),
            "request_id": (
                "STRING",
                {
                    "default": "",
                    "advanced": True,
                    "tooltip": "Retry a retained shared job without rerunning its sampler.",
                },
            ),
        },
    }


def warmup_inputs():
    # Preserve widget order for existing saved workflows.
    fields = connection_inputs()
    return {
        "required": {
            "vae_name": fields["vae_name"],
            "kind": (
                ["image", "audio"],
                {"tooltip": "Image/video or audio VAE to prepare."},
            ),
            "server_url": fields["server_url"],
            "timeout_seconds": fields["timeout_seconds"],
            "background": (
                "BOOLEAN",
                {
                    "default": True,
                    "advanced": True,
                    "tooltip": "Queue warmup and continue sampling immediately. Decode waits for the readiness signal.",
                },
            ),
        },
        "optional": {
            "after": (
                "*",
                {"tooltip": "Connect conditioning after encoding and cleanup. Pass through to the sampler to start warmup before sampling."},
            )
        },
    }


def video_inputs():
    return {
        "required": {
            **connection_inputs(),
            **video_fields(),
            "memory_policy": memory_input(),
        },
        "optional": {
            "latent": (LATENT_TYPE,),
            "request_id": (
                "STRING",
                {
                    "default": "",
                    "tooltip": "재시도 작업 ID. 완료된 결과도 재조회할 수 있습니다.",
                },
            ),
            "warmup": (WARMUP_TYPE,),
        },
    }


__all__ = [
    "WARMUP_TYPE",
    "LATENT_TYPE",
    "VIDEO_FPS",
    "VIDEO_CRF",
    "DECODE_DEFAULTS",
    "connection_inputs",
    "memory_input",
    "decode_fields",
    "decode_inputs",
    "warmup_inputs",
    "validate_model_choice",
    "video_fields",
    "video_inputs",
    "video_options",
]
