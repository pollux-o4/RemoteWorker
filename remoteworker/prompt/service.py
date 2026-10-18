"""Remote peer text-generation adapter returning an caller-ready STRING prompt."""

from __future__ import annotations

import re
import uuid
from typing import Any

from ..core.artifacts import PROMPT_ARTIFACT
from ..core.execution import RequestInputs, completed_artifacts
from ..core.media import encode_image_frame_png
from ..core.protocol import is_valid_request_id
from ..defaults import DEFAULT_SERVER_URL
from ..prompt.transport import PromptComfyApi
from .profiles import DEFAULT_PROFILE, PromptEnhancerProfile

_MAX_PROMPT_BYTES = 128 * 1024
_DYNAMIC_MEDIA_NAME = re.compile(r"^(image|video)_([0-9]+)$")
_MAX_IMAGE_REFERENCES = 8
# Match MiniMax H3 Ref2VA's official ref_video_0..2 contract.  The tokenizer
# itself preserves each batch as an independent <Video N> reference.
_MAX_VIDEO_REFERENCES = 3
_MAX_VIDEO_FRAMES = 32


def build_generation_prompt(prompt: str, system_prompt: str = DEFAULT_PROFILE.system_prompt) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Remote Prompt Enhancer prompt must be a non-empty string")
    if "\x00" in prompt or len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("Remote Prompt Enhancer prompt is invalid or too large")
    if not isinstance(system_prompt, str) or not system_prompt.strip() or "\x00" in system_prompt:
        raise ValueError("Remote Prompt Enhancer system_prompt is invalid")
    return f"{system_prompt.strip()}\n\nUser request:\n{prompt.strip()}"


def _image_batch(workflow: dict[str, Any], prefix: str, references: list[str]) -> list[Any]:
    if not references:
        raise ValueError("Remote Prompt Enhancer image batch cannot be empty")
    previous: list[Any] | None = None
    for index, reference in enumerate(references):
        load_name = f"{prefix}_{index}"
        workflow[load_name] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        current: list[Any] = [load_name, 0]
        if previous is not None:
            batch_name = f"{prefix}_batch_{index}"
            workflow[batch_name] = {"class_type": "ImageBatch", "inputs": {"image1": previous, "image2": current}}
            current = [batch_name, 0]
        previous = current
    return previous


def build_workflow(
    profile: PromptEnhancerProfile,
    prompt: str,
    request_id: str,
    *,
    image_refs: list[str] | None = None,
    video_refs: list[list[str]] | None = None,
) -> dict[str, Any]:
    if not isinstance(profile, PromptEnhancerProfile):
        raise ValueError("Remote Prompt Enhancer profile is invalid")
    if not is_valid_request_id(request_id):
        raise ValueError("Remote Prompt Enhancer request_id must be 1-64 URL-safe characters")
    image_refs = image_refs or []
    video_refs = video_refs or []
    if image_refs and "image" not in profile.accepted_media:
        raise ValueError(f"Remote Prompt Enhancer profile {profile.id} does not accept images")
    if video_refs and "video" not in profile.accepted_media:
        raise ValueError(f"Remote Prompt Enhancer profile {profile.id} does not accept video")
    workflow: dict[str, Any] = {
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": profile.clip_name,
                "type": profile.clip_type,
                "device": profile.device,
            },
        },
        "generate": {
            "class_type": "RemotePromptGenerate",
            "inputs": {
                "clip": ["clip", 0],
                "prompt": prompt,
                "system_prompt": profile.system_prompt,
                "media_strategy": "minimax-ref2va" if profile.clip_type == "minimax" else "qwen-image",
                "max_length": profile.max_length,
            },
        },
        "save": {
            "class_type": "SaveRemoteText",
            "inputs": {
                "text": ["generate", 0],
                "filename_prefix": f"remote-conditioning/{request_id}/prompt",
            },
        },
    }
    if image_refs:
        workflow["generate"]["inputs"]["image"] = _image_batch(workflow, "image", image_refs)
    for index, frames in enumerate(video_refs):
        workflow["generate"]["inputs"][f"video_{index}"] = _image_batch(workflow, f"video_{index}", frames)
    return workflow


def decode_generated_prompt(data: bytes) -> str:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_PROMPT_BYTES:
        raise RuntimeError("Remote Prompt Enhancer returned an invalid prompt artifact")
    try:
        result = data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Remote Prompt Enhancer returned a non-UTF-8 prompt artifact") from exc
    if not result or "\x00" in result:
        raise RuntimeError("Remote Prompt Enhancer returned an empty prompt")
    return result


def _ordered_batches(kind: str, media: dict[str, Any], maximum: int) -> list[Any]:
    entries: list[tuple[int, Any]] = []
    for name, value in media.items():
        match = _DYNAMIC_MEDIA_NAME.fullmatch(name)
        if match and match.group(1) == kind:
            entries.append((int(match.group(2)), value))
    entries.sort(key=lambda entry: entry[0])
    if len(entries) > maximum or [index for index, _ in entries] != list(range(len(entries))):
        raise ValueError(f"Remote Prompt Enhancer {kind} sockets must be contiguous and contain at most {maximum} references")
    for _index, batch in entries:
        shape = getattr(batch, "shape", None)
        if not isinstance(shape, (tuple, list)) or len(shape) != 4 or shape[0] < 1 or shape[-1] not in {3, 4}:
            raise ValueError(f"Remote Prompt Enhancer {kind} must be a BHWC IMAGE tensor")
    return [batch for _index, batch in entries]


def _ordered_media(kind: str, media: dict[str, Any], maximum: int) -> list[Any]:
    return [frame for batch in _ordered_batches(kind, media, maximum) for frame in batch]


def execute(
    *,
    profile_id: str,
    prompt: str,
    request_id: str,
    api: Any,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> str:
    profile = api.get_prompt_enhancer_profile(profile_id)
    if not isinstance(profile, PromptEnhancerProfile):
        raise RuntimeError("Remote Prompt Enhancer profile response is invalid")
    api.require_node("RemotePromptGenerate")
    api.require_node("SaveRemoteText")
    images, videos = (images or [], videos or [])
    if len(images) > _MAX_IMAGE_REFERENCES or len(videos) > _MAX_VIDEO_REFERENCES:
        raise ValueError("Remote Prompt Enhancer media exceeds the request limit")
    if any((getattr(video, "shape", (0,))[0] > _MAX_VIDEO_FRAMES for video in videos)):
        raise ValueError("Remote Prompt Enhancer video exceeds the frame limit")
    if images and "image" not in profile.accepted_media:
        raise ValueError(f"Remote Prompt Enhancer profile {profile.id} does not accept images")
    if videos and "video" not in profile.accepted_media:
        raise ValueError(f"Remote Prompt Enhancer profile {profile.id} does not accept video")
    with RequestInputs(api, request_id) as request:
        image_refs = []
        for index, image in enumerate(images, start=1):
            reference = request.upload_image(index=index, png=encode_image_frame_png(image))
            image_refs.append(reference)
        video_refs: list[list[str]] = []
        for video_index, video in enumerate(videos):
            frames = []
            for frame_index in range(video.shape[0]):
                reference = request.upload_file(
                    filename=f"prompt-video-{video_index:02d}-frame-{frame_index:03d}.png",
                    data=encode_image_frame_png(video[frame_index]),
                    content_type="image/png",
                )
                frames.append(reference)
            video_refs.append(frames)
        workflow = build_workflow(profile, prompt, request_id, image_refs=image_refs, video_refs=video_refs)
        with completed_artifacts(api, workflow, request_id, (PROMPT_ARTIFACT,)) as blobs:
            return decode_generated_prompt(blobs["prompt"])


class DynamicPromptMedia(dict[str, tuple[str]]):
    """Declare autogrow image/video sockets; profile policy validates their use."""

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and _DYNAMIC_MEDIA_NAME.fullmatch(key) is not None

    def __getitem__(self, key: str) -> tuple[str]:
        if key not in self:
            raise KeyError(key)
        return ("IMAGE",)


class RemotePromptEnhancer:
    """Execute an peer-owned enhancement profile and emit plain text for any graph."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "profile_id": ([DEFAULT_PROFILE.id],),
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 180, "min": 5, "max": 1800}),
            },
            "optional": DynamicPromptMedia(),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote"

    @classmethod
    def VALIDATE_INPUTS(cls, profile_id: Any, **_kwargs: Any) -> bool | str:
        if not isinstance(profile_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", profile_id):
            return "Remote Prompt Enhancer profile_id is invalid"
        return True

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        return float("nan")

    def run(
        self,
        prompt: str,
        profile_id: str,
        server_url: str = DEFAULT_SERVER_URL,
        timeout_seconds: int = 180,
        **media: Any,
    ) -> tuple[str]:
        try:
            api = PromptComfyApi(server_url, timeout_seconds)
            result = execute(
                profile_id=profile_id,
                prompt=prompt,
                request_id=uuid.uuid4().hex,
                api=api,
                images=_ordered_media("image", media, _MAX_IMAGE_REFERENCES),
                videos=_ordered_batches("video", media, _MAX_VIDEO_REFERENCES),
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Remote Prompt Enhancer failed: {exc}") from exc
        return (result,)


class RemotePromptGenerate:
    """peer-only worker: turn a system prompt, raw request, and visual refs into text."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True}),
                "system_prompt": ("STRING", {"multiline": True}),
                "media_strategy": (["qwen-image", "minimax-ref2va"],),
                "max_length": ("INT", {"default": 256, "min": 16, "max": 2048}),
            },
            "optional": {
                "image": ("IMAGE",),
                **{f"video_{index}": ("IMAGE",) for index in range(_MAX_VIDEO_REFERENCES)},
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "conditioning/remote/internal"

    def run(self, clip: Any, prompt: str, system_prompt: str, media_strategy: str, max_length: int, image: Any = None, **media: Any) -> tuple[str]:
        text = build_generation_prompt(prompt, system_prompt)
        videos = [media[f"video_{index}"] for index in range(_MAX_VIDEO_REFERENCES) if media.get(f"video_{index}") is not None]
        if media_strategy == "qwen-image":
            if videos:
                raise ValueError("Remote Prompt Generate qwen-image does not accept video")
            tokens = clip.tokenize(text, image=image, skip_template=False, min_length=1, thinking=False)
        elif media_strategy == "minimax-ref2va":
            refs = []
            if image is not None:
                refs.extend({"type": "image", "data": image[index:index + 1]} for index in range(image.shape[0]))
            refs.extend({"type": "video", "data": video} for video in videos)
            tokens = clip.tokenize(text, minimax_ref_items=refs)
        else:
            raise ValueError("Remote Prompt Generate media strategy is invalid")
        generated_ids = clip.generate(tokens, do_sample=False, max_length=max_length)
        return (clip.decode(generated_ids),)


NODE_CLASS_MAPPINGS = {
    "RemotePromptEnhancer": RemotePromptEnhancer,
    "RemotePromptGenerate": RemotePromptGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotePromptEnhancer": "Remote Prompt Enhancer",
    "RemotePromptGenerate": "Remote Prompt Generate (internal)",
}
