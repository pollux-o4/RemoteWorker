"""MiniMax reference workflow construction."""
from __future__ import annotations

from typing import Any

from ..core.protocol import ClipSelection
from .common import DEFAULT_PEER_MEMORY_POLICY, add_release_cleanup
from .media import _MAX_REF_AUDIOS, _MAX_REF_IMAGES, _MAX_REF_VIDEOS


def _image_loader(workflow: dict[str, Any], name: str, image: str) -> list[Any]:
    workflow[name] = {"class_type": "LoadImage", "inputs": {"image": image}}
    return [name, 0]


def _video_loader(workflow: dict[str, Any], index: int, frames: list[str]) -> list[Any]:
    if not frames:
        raise ValueError("Remote MiniMax H3 reference video has no frames")
    previous = _image_loader(workflow, f"ref_video_{index}_frame_0", frames[0])
    for frame_index, frame in enumerate(frames[1:], start=1):
        current = _image_loader(workflow, f"ref_video_{index}_frame_{frame_index}", frame)
        batch = f"ref_video_{index}_batch_{frame_index}"
        workflow[batch] = {"class_type": "ImageBatch", "inputs": {"image1": previous, "image2": current}}
        previous = [batch, 0]
    return previous


def build_workflow(
    *,
    clip: ClipSelection,
    vae_name: str,
    audio_vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_image_size: str,
    image_refs: list[str],
    video_refs: list[list[str]],
    video_audio_refs: list[str],
    audio_refs: list[str],
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    if not 0 <= len(image_refs) <= _MAX_REF_IMAGES or not 0 <= len(video_refs) <= _MAX_REF_VIDEOS:
        raise ValueError("Remote MiniMax H3 reference count is invalid")
    if not 0 <= len(video_audio_refs) <= _MAX_REF_AUDIOS or not 0 <= len(audio_refs) <= _MAX_REF_AUDIOS:
        raise ValueError("Remote MiniMax H3 audio reference count is invalid")
    if len(video_audio_refs) > len(video_refs):
        raise ValueError("Remote MiniMax H3 video audio cannot exceed reference videos")
    if not all(isinstance(value, str) and value.strip() for value in (vae_name, audio_vae_name, prompt)):
        raise ValueError("Remote MiniMax H3 requires non-empty VAE names and prompt")
    if width < 32 or height < 32 or length < 5 or ref_image_size not in {"match", "max"}:
        raise ValueError("Remote MiniMax H3 dimensions or ref_image_size are invalid")
    workflow: dict[str, Any] = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip.clip_name, "type": clip.clip_type, "device": clip.device}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        "encode": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["audio_vae", 0], "prompt": prompt, "width": width, "height": height, "length": length, "ref_image_size": ref_image_size}},
        "save": {"class_type": "SaveConditioning", "inputs": {"conditioning": ["encode", 0], "filename_prefix": f"remote-conditioning/{request_id}/cond"}},
        "save_latent": {"class_type": "SaveRemoteLatent", "inputs": {"samples": ["encode", 1], "filename_prefix": f"remote-conditioning/{request_id}/latent"}},
    }
    for index, ref in enumerate(image_refs):
        name = f"ref_image_{index}"
        workflow["encode"]["inputs"][f"ref_images.{name}"] = _image_loader(workflow, name, ref)
    for index, frames in enumerate(video_refs):
        name = f"ref_video_{index}"
        workflow["encode"]["inputs"][f"ref_videos.{name}"] = _video_loader(workflow, index, frames)
    for index, ref in enumerate(video_audio_refs):
        name = f"ref_video_audio_{index}"
        workflow[name] = {"class_type": "LoadAudio", "inputs": {"audio": ref}}
        workflow["encode"]["inputs"][f"ref_video_audios.{name}"] = [name, 0]
    for index, ref in enumerate(audio_refs):
        name = f"ref_audio_{index}"
        workflow[name] = {"class_type": "LoadAudio", "inputs": {"audio": ref}}
        workflow["encode"]["inputs"][f"ref_audios.{name}"] = [name, 0]
    add_release_cleanup(workflow, peer_memory_policy)
    return workflow


