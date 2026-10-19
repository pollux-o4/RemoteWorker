"""Public VAE node inputs and semantic connection contracts."""
from __future__ import annotations

from ..defaults import DEFAULT_SERVER_URL

WARMUP_TYPE = "REMOTE_VAE_WARMUP"


def connection_inputs():
    return {
        "vae_name": ([""], {"default": "", "tooltip": "Choose a VAE from the peer model list. No VAE is loaded on this server."}),
        "server_url": ("STRING", {"default": DEFAULT_SERVER_URL, "tooltip": "ComfyUI that will load and run the VAE. Use a different server from this workflow."}),
        "timeout_seconds": ("INT", {"default": 600, "min": 5, "max": 1800, "advanced": True, "tooltip": "Maximum wait for each peer request, in seconds."}),
    }


def decode_inputs(kind):
    required = {"samples": ("LATENT", {"tooltip": "Sampler output. H3 combined video/audio latents are supported."}), **connection_inputs(),
        "memory_policy": (["release_after_result", "keep"], {"advanced": True,
            "tooltip": "Release the peer's models after saving the decoded result, or retain them for the next decode."})}
    if kind == "image":
        required.update({
            "tiled": ("BOOLEAN", {"default": False, "tooltip": "Enable native tiled decoding. H3 video VAE already tiles internally."}),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "advanced": True}),
            "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "advanced": True}),
            "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "advanced": True}),
        })
    return {"required": required, "optional": {
        "warmup": (WARMUP_TYPE, {"forceInput": True, "tooltip": "Optional manual override. VAE preparation is automatic during sampling; connect only to explicitly order a manual warmup using the same peer, VAE and output type."})}}


def warmup_inputs():
    # Preserve widget order for existing saved workflows.
    fields = connection_inputs()
    return {"required": {
        "vae_name": fields["vae_name"],
        "kind": (["image", "audio"], {"tooltip": "Image/video or audio VAE to prepare."}),
        "server_url": fields["server_url"], "timeout_seconds": fields["timeout_seconds"],
        "background": ("BOOLEAN", {"default": True, "advanced": True,
            "tooltip": "Queue warmup and continue sampling immediately. Decode waits for the readiness signal."}),
    }, "optional": {"after": ("*", {"tooltip": "Connect conditioning after encoding and cleanup. Pass through to the sampler to start warmup before sampling."})}}


def validate_model_choice(vae_name):
    return True if isinstance(vae_name, str) and vae_name.strip() and "\x00" not in vae_name else "원격 VAE를 선택하세요. 노드의 모델 목록 새로고침으로 목록 조회 가능."
