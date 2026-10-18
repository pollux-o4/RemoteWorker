"""Remote MiniMax H3 ref2va adapter: peer encodes; caller samples."""

from __future__ import annotations

import uuid
from typing import Any

from ..core.artifacts import CONDITIONING_ARTIFACT
from ..core.execution import RequestInputs, completed_artifacts
from ..core.media import encode_image_frame_png
from ..core.protocol import DEFAULT_CACHE_POLICY, ClipSelection
from ..core.transport import HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from .common import (
    DEFAULT_PEER_MEMORY_POLICY,
    LATENT_ARTIFACT,
    MINIMAX_AUDIO_VAE_CHOICES,
    MINIMAX_CLIP_CHOICES,
    MINIMAX_VIDEO_VAE_CHOICES,
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
from .media import (
    _MAX_REF_AUDIOS,
    _MAX_REF_IMAGES,
    _MAX_REF_VIDEOS,
    DynamicReferenceAndVaeInputs,
    _audio_fingerprint,
    _image_batch_fingerprint,
    _ordered_audio,
    _ordered_images,
    encode_audio_wav,
)
from .reference_workflow import build_workflow


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
    validate_peer_memory_policy(peer_memory_policy)
    return (
        "minimax-h3-ref2va",
        clip,
        vae_name,
        audio_vae_name,
        prompt,
        int(width),
        int(height),
        int(length),
        ref_image_size,
        tuple(_image_batch_fingerprint(image) for image in images),
        tuple(_image_batch_fingerprint(video) for video in videos),
        tuple(_audio_fingerprint(audio) for audio in video_audios),
        tuple(_audio_fingerprint(audio) for audio in audios),
        str(server_url).rstrip("/"),
    )


def validate_ref2va_capability(capability: dict[str, Any]) -> None:
    """Require the current V3 nested Autogrow payload contract before upload."""
    if not {"CONDITIONING", "LATENT"}.issubset(capability.get("output", [])):
        raise RuntimeError("Remote MiniMax H3 peer is incompatible")
    try:
        optional = capability["input"]["optional"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Remote MiniMax H3 peer ref2va capability is malformed") from exc
    required_groups = {"ref_images", "ref_videos", "ref_video_audios", "ref_audios"}
    if not required_groups.issubset(optional):
        raise RuntimeError("Remote MiniMax H3 peer does not support the nested reference input contract")


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
    capability = api.require_node("MiniMaxH3ReferenceToVideo")
    validate_ref2va_capability(capability)
    validate_release_cleanup_capability(api, peer_memory_policy)
    validate_peer_inputs(api, clip, vae_name, audio_vae_name)
    with RequestInputs(api, request_id) as request:
        image_refs = []
        for index, image in enumerate(images):
            image_refs.append(request.upload_image(index=index + 1, png=encode_image_frame_png(image[0])))
        video_refs: list[list[str]] = []
        for video_index, video in enumerate(videos):
            paths: list[str] = []
            for frame_index in range(video.shape[0]):
                paths.append(
                    request.upload_file(
                        filename=f"video-{video_index:02d}-frame-{frame_index:04d}.png",
                        data=encode_image_frame_png(video[frame_index]),
                        content_type="image/png",
                    )
                )
            video_refs.append(paths)
        video_audio_refs = []
        for index, audio in enumerate(video_audios):
            video_audio_refs.append(
                request.upload_file(
                    filename=f"video-audio-{index:02d}.wav",
                    data=encode_audio_wav(audio),
                    content_type="audio/wav",
                )
            )
        audio_refs = []
        for index, audio in enumerate(audios):
            audio_refs.append(
                request.upload_file(
                    filename=f"audio-{index:02d}.wav",
                    data=encode_audio_wav(audio),
                    content_type="audio/wav",
                )
            )
        workflow = build_workflow(
            clip=clip,
            vae_name=vae_name,
            audio_vae_name=audio_vae_name,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            image_refs=image_refs,
            video_refs=video_refs,
            video_audio_refs=video_audio_refs,
            audio_refs=audio_refs,
            request_id=request_id,
            peer_memory_policy=peer_memory_policy,
        )
        with completed_artifacts(api, workflow, request_id, (CONDITIONING_ARTIFACT, LATENT_ARTIFACT)) as blobs:
            return deserialize(blobs["conditioning"], blobs["latent"])


class RemoteMiniMaxH3ReferenceEncode:
    """Mirror MiniMaxH3ReferenceToVideo inputs and return remote outputs locally."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "clip_name": (MINIMAX_CLIP_CHOICES,),
                "clip_type": (["minimax"],),
                "device": (["default"],),
                "vae_name": (MINIMAX_VIDEO_VAE_CHOICES, {"default": MINIMAX_VIDEO_VAE_CHOICES[0]}),
                "audio_vae_name": (MINIMAX_AUDIO_VAE_CHOICES, {"default": MINIMAX_AUDIO_VAE_CHOICES[0]}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
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
        validate_cache_policy(cache_policy)
        return float("nan") if cache_policy == "always_refresh" else 0

    def run(self, prompt, clip_name, clip_type, device, vae_name, audio_vae_name, width, height, length, ref_image_size, server_url=DEFAULT_SERVER_URL, timeout_seconds=600, peer_memory_policy=DEFAULT_PEER_MEMORY_POLICY, cache_policy=DEFAULT_CACHE_POLICY, unique_id=None, extra_pnginfo=None, **media):
        try:
            validate_cache_policy(cache_policy)
            vae_name_override = media.pop("vae_name_override", None)
            audio_vae_name_override = media.pop("audio_vae_name_override", None)
            vae_name = resolve_vae_name(
                vae_name,
                vae_name_override,
                input_name="vae_name_override" if vae_name_override is not None else "vae_name",
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
            )
            audio_vae_name = resolve_vae_name(
                audio_vae_name,
                audio_vae_name_override,
                input_name="audio_vae_name_override" if audio_vae_name_override is not None else "audio_vae_name",
                unique_id=unique_id,
                extra_pnginfo=extra_pnginfo,
            )
            images = _ordered_images("ref_image", media, batches=False, maximum=_MAX_REF_IMAGES)
            videos = _ordered_images("ref_video", media, batches=True, maximum=_MAX_REF_VIDEOS)
            video_audios = _ordered_audio("ref_video_audio", media, _MAX_REF_AUDIOS)
            audios = _ordered_audio("ref_audio", media, _MAX_REF_AUDIOS)
            if len(video_audios) > len(videos):
                raise ValueError("Remote MiniMax H3 video audio requires a same-index reference video")
            clip = ClipSelection(clip_name, clip_type, device)
            cache_key = None
            if cache_policy == DEFAULT_CACHE_POLICY:
                cache_key = _reference_cache_key(
                    clip=clip,
                    vae_name=vae_name,
                    audio_vae_name=audio_vae_name,
                    prompt=prompt,
                    width=width,
                    height=height,
                    length=length,
                    ref_image_size=ref_image_size,
                    images=images,
                    videos=videos,
                    video_audios=video_audios,
                    audios=audios,
                    server_url=server_url,
                    peer_memory_policy=peer_memory_policy,
                )
                cached = get_cached_minimax_outputs(cache_key)
                if cached is not None:
                    print("[RemoteMiniMaxH3] cache hit")
                    return decode_remote_outputs(cached[0], cached[1])
            captured: dict[str, tuple[bytes, bytes]] = {}

            def deserialize_and_capture(conditioning: bytes, latent: bytes) -> tuple[Any, dict[str, Any]]:
                captured["outputs"] = (bytes(conditioning), bytes(latent))
                return decode_remote_outputs(conditioning, latent)

            result = execute(
                clip=clip,
                vae_name=vae_name,
                audio_vae_name=audio_vae_name,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                images=images,
                videos=videos,
                video_audios=video_audios,
                audios=audios,
                request_id=uuid.uuid4().hex,
                api=HttpComfyApi(server_url, timeout_seconds),
                deserialize=deserialize_and_capture,
                peer_memory_policy=peer_memory_policy,
            )
            if cache_key is not None and "outputs" in captured:
                put_cached_minimax_outputs(cache_key, captured["outputs"][0], captured["outputs"][1])
            return result
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Remote MiniMax H3 failed: {exc}") from exc


NODE_CLASS_MAPPINGS = {"RemoteMiniMaxH3ReferenceEncode": RemoteMiniMaxH3ReferenceEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteMiniMaxH3ReferenceEncode": "Remote MiniMax H3 Reference Encode"}
