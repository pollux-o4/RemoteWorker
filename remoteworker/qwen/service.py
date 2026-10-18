"""Remote Qwen Image 2.1 conditioning: peer encodes, caller samples."""

from __future__ import annotations

import hashlib
import io
import uuid
from typing import Any

from ..core.execution import RequestInputs, completed_artifacts
from ..defaults import DEFAULT_SERVER_URL, default_value

try:
    from comfy_api.latest import io as comfy_io
except ImportError:  # Unit tests exercise the transport service without ComfyUI.
    comfy_io = None

from ..core.artifacts import RemoteArtifactSpec
from ..core.media import encode_image_frame_png, ordered_image_frames
from ..core.protocol import (
    CACHE_POLICIES,
    DEFAULT_CACHE_POLICY,
    DEFAULT_PEER_MEMORY_POLICY,
    PEER_MEMORY_POLICIES,
    ClipSelection,
    _validate_policy,
    validate_cleanup_capability,
)
from ..core.result_cache import ByteResultCache
from ..core.transport import HttpComfyApi

QWEN_WORKER = "RemoteQwenImage21Encode"
SAVE_REMOTE_CONDITIONING = "SaveRemoteConditioning"
POSITIVE_ARTIFACT = RemoteArtifactSpec("positive", "save_positive", "conditioning", ".cond")
NEGATIVE_ARTIFACT = RemoteArtifactSpec("negative", "save_negative", "conditioning", ".cond")
LATENT_ARTIFACT = RemoteArtifactSpec("latent", "save_latent", "latents", ".latent")
DEFAULT_QWEN_CLIP = default_value("qwen_clip", "")
DEFAULT_QWEN_VAE = default_value("qwen_vae", "")
_MAX_IMAGES = 16
_CACHE = ByteResultCache(max_entries=4, max_bytes=512 * 1024 * 1024)


def _validate_choice(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"Remote Qwen Image 2.1 {name} must be a non-empty string")
    return value


def _validate_connection(server_url: Any, timeout_seconds: Any, remote_text_encoder: Any, vae_name: Any) -> tuple[str, int, str, str]:
    server_url = _validate_choice("server_url", server_url)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 5 <= timeout_seconds <= 1800:
        raise ValueError("Remote Qwen Image 2.1 timeout_seconds must be 5 to 1800")
    return (
        server_url,
        timeout_seconds,
        _validate_choice("remote_text_encoder", remote_text_encoder),
        _validate_choice("vae_name", vae_name),
    )


def _validate_dynamic_peer_models(remote_text_encoder: Any, vae_name: Any) -> bool | str:
    """Allow values refreshed from a peer while retaining execution-time capability checks."""
    try:
        _validate_choice("remote_text_encoder", remote_text_encoder)
        _validate_choice("vae_name", vae_name)
    except ValueError as exc:
        return str(exc)
    return True


def _ordered_qwen_image_frames(**images: Any) -> list[Any]:
    indices = sorted(
        int(name[6:]) for name in images
        if name.startswith("image_") and name[6:].isdigit()
    )
    if len(indices) > _MAX_IMAGES or indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"Remote Qwen Image 2.1 accepts up to {_MAX_IMAGES} consecutive image sockets")
    frames = ordered_image_frames(len(indices), **images)
    if len(frames) > _MAX_IMAGES:
        raise ValueError(f"Remote Qwen Image 2.1 accepts at most {_MAX_IMAGES} image attachments")
    return frames


def _validate_qwen_capability(info: dict[str, Any]) -> None:
    try:
        required = info["input"]["required"]
        optional = info["input"].get("optional", {})
        outputs = info["output"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Remote Qwen Image 2.1 peer capability is malformed") from exc
    if not {"clip", "prompt", "negative_prompt", "resolution"}.issubset(required) or not {"vae", "negative_vision", "image_1"}.issubset(optional) or not {"CONDITIONING", "LATENT"}.issubset(outputs):
        raise RuntimeError("Remote Qwen Image 2.1 peer is incompatible; update custom-nodes on the peer")


def _validate_peer_loaders(api: Any, clip: ClipSelection, vae_name: str) -> None:
    options = api.clip_loader_options()
    for field, value in zip(("clip_name", "clip_type", "device"), clip):
        if value not in options.get(field, []):
            raise ValueError(f"Remote Qwen Image 2.1 peer does not provide {field}: {value!r}")
    try:
        vaes = api.require_node("VAELoader")["input"]["required"]["vae_name"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("Remote Qwen Image 2.1 peer VAELoader capability is malformed") from exc
    if not isinstance(vaes, list) or vae_name not in vaes:
        raise ValueError(f"Remote Qwen Image 2.1 peer does not provide vae_name: {vae_name!r}")


def build_workflow(*, clip: ClipSelection, vae_name: str, prompt: str, negative_prompt: str, resolution: int, negative_vision: bool, image_refs: list[str], request_id: str, peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY) -> dict[str, Any]:
    _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
    if not 0 <= len(image_refs) <= _MAX_IMAGES:
        raise ValueError(f"Remote Qwen Image 2.1 accepts at most {_MAX_IMAGES} images")
    if not isinstance(resolution, int) or isinstance(resolution, bool) or not 0 <= resolution <= 4096:
        raise ValueError("Remote Qwen Image 2.1 resolution must be 0 to 4096")
    for name, value in (("prompt", prompt), ("negative_prompt", negative_prompt), ("vae_name", vae_name), ("request_id", request_id)):
        _validate_choice(name, value) if name != "negative_prompt" or value else None
    workflow: dict[str, Any] = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip.clip_name, "type": clip.clip_type, "device": clip.device}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "encode": {"class_type": QWEN_WORKER, "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "negative_prompt": negative_prompt, "resolution": resolution, "negative_vision": bool(negative_vision)}},
        "save_positive": {"class_type": SAVE_REMOTE_CONDITIONING, "inputs": {"conditioning": ["encode", 0], "filename_prefix": f"remote-conditioning/{request_id}/positive"}},
        "save_negative": {"class_type": SAVE_REMOTE_CONDITIONING, "inputs": {"conditioning": ["encode", 1], "filename_prefix": f"remote-conditioning/{request_id}/negative", "dependency": ["save_positive", 0]}},
        "save_latent": {"class_type": "SaveRemoteLatent", "inputs": {"samples": ["encode", 2], "filename_prefix": f"remote-conditioning/{request_id}/latent", "dependency": ["save_negative", 0]}},
    }
    for index, ref in enumerate(image_refs, start=1):
        loader = f"image_{index}_loader"
        workflow[loader] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        workflow["encode"]["inputs"][f"image_{index}"] = [loader, 0]
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        workflow["cleanup"] = {"class_type": "VRAMCleanup", "inputs": {"anything": ["save_latent", 0], "offload_model": True, "offload_cache": True, "reset_execution_cache": False}}
    return workflow


def _cache_key(*, clip: ClipSelection, vae_name: str, prompt: str, negative_prompt: str, resolution: int, negative_vision: bool, frames: list[Any], server_url: str) -> tuple[Any, ...]:
    return ("qwen-image-2.1", clip, vae_name, prompt, negative_prompt, resolution, negative_vision, tuple(hashlib.sha256(encode_image_frame_png(frame)).hexdigest() for frame in frames), server_url.rstrip("/"))


def _decode(positive: bytes, negative: bytes, latent: bytes) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    latent_value = torch.load(io.BytesIO(latent), map_location="cpu", weights_only=False)
    if not isinstance(latent_value, dict) or "samples" not in latent_value:
        raise RuntimeError("Remote Qwen Image 2.1 latent artifact is invalid")
    return (torch.load(io.BytesIO(positive), map_location="cpu", weights_only=False), torch.load(io.BytesIO(negative), map_location="cpu", weights_only=False), latent_value)


def execute(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    resolution: int,
    negative_vision: bool,
    frames: list[Any],
    request_id: str,
    api: Any,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> tuple[bytes, bytes, bytes]:
    _validate_qwen_capability(api.require_node(QWEN_WORKER))
    _validate_peer_loaders(api, clip, vae_name)
    if peer_memory_policy == DEFAULT_PEER_MEMORY_POLICY:
        validate_cleanup_capability(api.require_node("VRAMCleanup"))
    refs: list[str] = []
    with RequestInputs(api, request_id) as request:
        for index, frame in enumerate(frames, start=1):
            refs.append(request.upload_image(index=index, png=encode_image_frame_png(frame)))
        workflow = build_workflow(
            clip=clip,
            vae_name=vae_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            negative_vision=negative_vision,
            image_refs=refs,
            request_id=request_id,
            peer_memory_policy=peer_memory_policy,
        )
        with completed_artifacts(
            api,
            workflow,
            request_id,
            (POSITIVE_ARTIFACT, NEGATIVE_ARTIFACT, LATENT_ARTIFACT),
        ) as blobs:
            return tuple((blobs[name] for name in ("positive", "negative", "latent")))


class RemoteQwenImage21Service:
    """Transport service shared by the native Comfy schema and unit tests."""

    def run(self, prompt: str, negative_prompt: str, server_url: str, timeout_seconds: int, remote_text_encoder: str, vae_name: str, resolution: int, negative_vision: bool, peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY, cache_policy: str = DEFAULT_CACHE_POLICY, **images: Any) -> tuple[Any, Any, dict[str, Any]]:
        _validate_policy(peer_memory_policy, PEER_MEMORY_POLICIES, "peer_memory_policy")
        _validate_policy(cache_policy, CACHE_POLICIES, "cache_policy")
        server_url, timeout_seconds, remote_text_encoder, vae_name = _validate_connection(
            server_url, timeout_seconds, remote_text_encoder, vae_name
        )
        frames = _ordered_qwen_image_frames(**images)
        clip = ClipSelection(remote_text_encoder, "qwen_image", "default")
        _validate_choice("prompt", prompt)
        if not isinstance(negative_prompt, str):
            raise ValueError("Remote Qwen Image 2.1 negative_prompt must be a string")
        api = HttpComfyApi(server_url, timeout_seconds)
        key = _cache_key(clip=clip, vae_name=vae_name, prompt=prompt, negative_prompt=negative_prompt, resolution=resolution, negative_vision=negative_vision, frames=frames, server_url=server_url)
        cached = _CACHE.get(key) if cache_policy == DEFAULT_CACHE_POLICY else None
        if cached is not None:
            print("[RemoteQwenImage21] cache hit")
            return _decode(*cached)
        result = execute(clip=clip, vae_name=vae_name, prompt=prompt, negative_prompt=negative_prompt, resolution=resolution, negative_vision=negative_vision, frames=frames, request_id=uuid.uuid4().hex, api=api, peer_memory_policy=peer_memory_policy)
        if cache_policy == DEFAULT_CACHE_POLICY:
            _CACHE.put(key, result)
        return _decode(*result)


if comfy_io is not None:

    class RemoteQwenImage21Node(comfy_io.ComfyNode):
        """Native V3 presentation of the remote Qwen encoder contract."""

        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="RemoteQwenImage21",
                display_name="Qwen 2.1 원격 인코더",
                category="conditioning/remote",
                description=(
                    "이 노드는 선택한 원격 ComfyUI에서 Qwen 텍스트·참조 이미지·VAE 인코딩만 실행합니다. "
                    "원격 주소를 입력한 뒤 각 모델 목록을 새로고침하고, 참조 이미지를 연결하세요."
                ),
                inputs=[
                    comfy_io.String.Input(
                        "server_url",
                        display_name="1. 원격 ComfyUI 주소",
                        default=DEFAULT_SERVER_URL,
                        placeholder="http://127.0.0.1:8188",
                        tooltip="Qwen 인코딩을 실행할 ComfyUI 주소입니다. 주소를 입력한 뒤 아래 모델 목록 새로고침을 누르세요.",
                    ),
                    comfy_io.Int.Input(
                        "timeout_seconds",
                        display_name="연결 제한 시간(초)",
                        default=600,
                        min=5,
                        max=1800,
                        advanced=True,
                    ),
                    comfy_io.Combo.Input(
                        "remote_text_encoder",
                        display_name="2. 원격 텍스트 인코더",
                        options=[DEFAULT_QWEN_CLIP],
                        default=DEFAULT_QWEN_CLIP,
                        tooltip="위 원격 주소에 실제 설치된 Qwen 텍스트 인코더입니다. 목록 새로고침 후 선택하세요.",
                    ),
                    comfy_io.Combo.Input(
                        "vae_name",
                        display_name="3. 참조 이미지 VAE",
                        options=[DEFAULT_QWEN_VAE],
                        default=DEFAULT_QWEN_VAE,
                        tooltip="참조 이미지를 latent로 바꿀 때 원격에서 사용할 VAE입니다. 목록 새로고침 후 선택하세요.",
                    ),
                    comfy_io.String.Input(
                        "prompt",
                        display_name="편집 지시 (Positive)",
                        multiline=True,
                        dynamic_prompts=True,
                        default="",
                        placeholder="무엇을 만들거나 바꿀지 적으세요",
                    ),
                    comfy_io.String.Input(
                        "negative_prompt",
                        display_name="제외 지시 (Negative)",
                        multiline=True,
                        dynamic_prompts=True,
                        default="",
                        placeholder="피하고 싶은 요소를 적으세요 (비워도 됨)",
                    ),
                    comfy_io.Int.Input(
                        "resolution",
                        display_name="참조 이미지 처리 해상도",
                        default=1024,
                        min=0,
                        max=4096,
                        step=32,
                        tooltip="각 참조 이미지를 이 크기 근처로 맞춰 텍스트 인코더와 VAE에 보냅니다. 0은 원본 비율·크기를 유지합니다.",
                    ),
                    comfy_io.Boolean.Input(
                        "negative_vision",
                        display_name="Negative에도 참조 이미지 사용",
                        default=False,
                        label_on="이미지 사용",
                        label_off="텍스트만",
                        tooltip="Negative 텍스트는 항상 인코딩됩니다. 참조 이미지를 negative branch에도 넣어야 할 때만 켜세요.",
                    ),
                    comfy_io.Combo.Input(
                        "peer_memory_policy",
                        display_name="원격 메모리 정리 방식",
                        options=list(PEER_MEMORY_POLICIES),
                        default=DEFAULT_PEER_MEMORY_POLICY,
                        advanced=True,
                        tooltip="기본값은 인코딩 뒤 원격 GPU 메모리를 정리합니다. 같은 원격 모델을 연속 실행할 때만 유지로 바꾸세요.",
                    ),
                    comfy_io.Combo.Input(
                        "cache_policy",
                        display_name="동일 입력 결과 캐시",
                        options=list(CACHE_POLICIES),
                        default=DEFAULT_CACHE_POLICY,
                        advanced=True,
                        tooltip="기본값은 같은 입력의 원격 인코딩 결과를 재사용합니다. 매번 다시 인코딩해야 할 때만 새로고침을 선택하세요.",
                    ),
                    comfy_io.Autogrow.Input(
                        "images",
                        display_name="참조 이미지",
                        template=comfy_io.Autogrow.TemplateNames(
                            comfy_io.Image.Input("image", display_name="참조 이미지"),
                            names=[f"image_{index}" for index in range(1, _MAX_IMAGES + 1)],
                            min=0,
                        ),
                        tooltip="편집에 반영할 이미지입니다. image_1을 연결하면 image_2가 나타나며, 최대 16개까지 추가됩니다.",
                    ),
                ],
                outputs=[
                    comfy_io.Conditioning.Output(display_name="Positive 컨디셔닝"),
                    comfy_io.Conditioning.Output(display_name="Negative 컨디셔닝"),
                    comfy_io.Latent.Output(display_name="샘플러용 Latent"),
                ],
            )

        @classmethod
        def fingerprint_inputs(cls, cache_policy: str = DEFAULT_CACHE_POLICY, **_kwargs: Any) -> float:
            _validate_policy(cache_policy, CACHE_POLICIES, "cache_policy")
            return float("nan") if cache_policy == "always_refresh" else 0

        @classmethod
        def validate_inputs(
            cls,
            remote_text_encoder: str,
            vae_name: str,
            images: dict[str, Any] | None = None,
        ) -> bool | str:
            # The peer-specific lists are refreshed in the frontend.  Their
            # initial schema options are only safe defaults, not an allowlist.
            # execute() verifies that the selected values exist on this peer.
            return _validate_dynamic_peer_models(remote_text_encoder, vae_name)

        @classmethod
        def execute(
            cls,
            server_url: str,
            timeout_seconds: int,
            remote_text_encoder: str,
            vae_name: str,
            prompt: str,
            negative_prompt: str,
            resolution: int,
            negative_vision: bool,
            peer_memory_policy: str,
            cache_policy: str,
            images: dict[str, Any] | None = None,
        ):
            values = RemoteQwenImage21Service().run(
                prompt=prompt,
                negative_prompt=negative_prompt,
                server_url=server_url,
                timeout_seconds=timeout_seconds,
                remote_text_encoder=remote_text_encoder,
                vae_name=vae_name,
                resolution=resolution,
                negative_vision=negative_vision,
                peer_memory_policy=peer_memory_policy,
                cache_policy=cache_policy,
                **(images or {}),
            )
            return comfy_io.NodeOutput(*values)

    NODE_CLASS_MAPPINGS = {
        "RemoteQwenImage21": RemoteQwenImage21Node,
    }
else:
    # Import-only tests do not provide Comfy's V3 API.  They exercise the
    # transport services directly, while the live runtime always takes the
    # V3 branch above.
    NODE_CLASS_MAPPINGS = {
        "RemoteQwenImage21": RemoteQwenImage21Service,
    }


NODE_DISPLAY_NAME_MAPPINGS = {"RemoteQwenImage21": "Qwen 2.1 원격 인코더"}
