"""Remote adapter for the official MiniMax H3 image/text-to-video encoder."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from ..core.artifacts import CONDITIONING_ARTIFACT
from ..core.execution import RequestInputs, completed_artifacts
from ..core.media import encode_image_frame_png
from ..core.protocol import CACHE_POLICIES, DEFAULT_CACHE_POLICY, ClipSelection
from ..core.transport import HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from .common import (
    DEFAULT_PEER_MEMORY_POLICY,
    LATENT_ARTIFACT,
    MINIMAX_CLIP_CHOICES,
    MINIMAX_VIDEO_VAE_CHOICES,
    PEER_MEMORY_POLICIES,
    VAE_NAME_OVERRIDE_SOCKET,
    add_release_cleanup,
    decode_remote_outputs,
    get_cached_minimax_outputs,
    put_cached_minimax_outputs,
    resolve_vae_name,
    validate_cache_policy,
    validate_minimax_loader_inputs,
    validate_peer_inputs,
    validate_peer_memory_policy,
    validate_release_cleanup_capability,
)


def _single_frame(image: Any, name: str) -> Any:
    """Match MiniMax's first-frame behavior without silently accepting bad tensors."""
    shape = getattr(image, "shape", None)
    if not isinstance(shape, (tuple, list)) or len(shape) != 4:
        raise ValueError(f"Remote MiniMax H3 {name} must be a BHWC IMAGE tensor")
    frames, height, width, channels = shape
    if not all(isinstance(value, int) and value > 0 for value in (frames, height, width)) or channels not in {3, 4}:
        raise ValueError(f"Remote MiniMax H3 {name} has an invalid IMAGE shape: {shape!r}")
    return image[0]


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
    validate_peer_memory_policy(peer_memory_policy)
    return (
        "minimax-h3-i2v",
        clip,
        vae_name,
        prompt,
        int(width),
        int(height),
        int(length),
        _frame_fingerprint(first_frame, "first_frame"),
        _frame_fingerprint(last_frame, "last_frame"),
        str(server_url).rstrip("/"),
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
    if not all(isinstance(value, str) and value.strip() for value in (vae_name, prompt)):
        raise ValueError("Remote MiniMax H3 Image to Video requires a non-empty prompt and VAE name")
    if width < 32 or height < 32 or length < 5 or width % 32 or height % 32:
        raise ValueError("Remote MiniMax H3 Image to Video dimensions are invalid")
    workflow: dict[str, Any] = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip.clip_name, "type": clip.clip_type, "device": clip.device}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "encode": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": width, "height": height, "length": length}},
        "save": {"class_type": "SaveConditioning", "inputs": {"conditioning": ["encode", 0], "filename_prefix": f"remote-conditioning/{request_id}/cond"}},
        "save_latent": {"class_type": "SaveRemoteLatent", "inputs": {"samples": ["encode", 1], "filename_prefix": f"remote-conditioning/{request_id}/latent"}},
    }
    for input_name, path in (("first_frame", first_frame), ("last_frame", last_frame)):
        if path is not None:
            loader = input_name + "_loader"
            workflow[loader] = {"class_type": "LoadImage", "inputs": {"image": path}}
            workflow["encode"]["inputs"][input_name] = [loader, 0]
    add_release_cleanup(workflow, peer_memory_policy)
    return workflow


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
    capability = api.require_node("MiniMaxH3ImageToVideo")
    if not {"CONDITIONING", "LATENT"}.issubset(capability.get("output", [])):
        raise RuntimeError("Remote MiniMax H3 Image to Video peer is incompatible")
    validate_release_cleanup_capability(api, peer_memory_policy)
    validate_peer_inputs(api, clip, vae_name)
    with RequestInputs(api, request_id) as request:
        paths: dict[str, str | None] = {"first_frame": None, "last_frame": None}
        for name, image in (("first_frame", first_frame), ("last_frame", last_frame)):
            if image is not None:
                paths[name] = request.upload_file(
                    filename=f"{name}.png",
                    data=encode_image_frame_png(_single_frame(image, name)),
                    content_type="image/png",
                )
        workflow = build_workflow(
            clip=clip,
            vae_name=vae_name,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            first_frame=paths["first_frame"],
            last_frame=paths["last_frame"],
            request_id=request_id,
            peer_memory_policy=peer_memory_policy,
        )
        with completed_artifacts(api, workflow, request_id, (CONDITIONING_ARTIFACT, LATENT_ARTIFACT)) as blobs:
            return deserialize(blobs["conditioning"], blobs["latent"])


class RemoteMiniMaxH3ImageToVideo:
    """Run peer's MiniMaxH3ImageToVideo encoder and return caller-ready outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "clip_name": (MINIMAX_CLIP_CHOICES,),
                "clip_type": (["minimax"],),
                "device": (["default"],),
                "vae_name": (MINIMAX_VIDEO_VAE_CHOICES, {"default": MINIMAX_VIDEO_VAE_CHOICES[0]}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 600, "min": 5, "max": 1800}),
            },
            "optional": {
                "vae_name_override": (VAE_NAME_OVERRIDE_SOCKET, {"forceInput": True}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "peer_memory_policy": (list(PEER_MEMORY_POLICIES), {"default": DEFAULT_PEER_MEMORY_POLICY}),
                "cache_policy": (list(CACHE_POLICIES), {"default": DEFAULT_CACHE_POLICY}),
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
        validate_cache_policy(cache_policy)
        return float("nan") if cache_policy == "always_refresh" else 0

    def run(self, prompt, clip_name, clip_type, device, vae_name, width, height, length, server_url=DEFAULT_SERVER_URL, timeout_seconds=600, peer_memory_policy=DEFAULT_PEER_MEMORY_POLICY, cache_policy=DEFAULT_CACHE_POLICY, first_frame=None, last_frame=None, vae_name_override=None, unique_id=None, extra_pnginfo=None):
        try:
            validate_cache_policy(cache_policy)
            input_name = "vae_name_override" if vae_name_override is not None else "vae_name"
            vae_name = resolve_vae_name(
                vae_name,
                vae_name_override,
                input_name=input_name,
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
            )
            clip = ClipSelection(clip_name, clip_type, device)
            cache_key = None
            if cache_policy == DEFAULT_CACHE_POLICY:
                cache_key = _i2v_cache_key(
                    clip=clip,
                    vae_name=vae_name,
                    prompt=prompt,
                    width=width,
                    height=height,
                    length=length,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    server_url=server_url,
                    peer_memory_policy=peer_memory_policy,
                )
                cached = get_cached_minimax_outputs(cache_key)
                if cached is not None:
                    print("[RemoteMiniMaxH3ImageToVideo] cache hit")
                    return decode_remote_outputs(cached[0], cached[1])
            captured: dict[str, tuple[bytes, bytes]] = {}

            def deserialize_and_capture(conditioning: bytes, latent: bytes) -> tuple[Any, dict[str, Any]]:
                captured["outputs"] = (bytes(conditioning), bytes(latent))
                return decode_remote_outputs(conditioning, latent)

            result = execute(
                clip=clip,
                vae_name=vae_name,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                first_frame=first_frame,
                last_frame=last_frame,
                request_id=uuid.uuid4().hex,
                api=HttpComfyApi(server_url, timeout_seconds),
                deserialize=deserialize_and_capture,
                peer_memory_policy=peer_memory_policy,
            )
            if cache_key is not None and "outputs" in captured:
                put_cached_minimax_outputs(cache_key, captured["outputs"][0], captured["outputs"][1])
            return result
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Remote MiniMax H3 Image to Video failed: {exc}") from exc


NODE_CLASS_MAPPINGS = {"RemoteMiniMaxH3ImageToVideo": RemoteMiniMaxH3ImageToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteMiniMaxH3ImageToVideo": "Remote MiniMax H3 Image to Video"}
