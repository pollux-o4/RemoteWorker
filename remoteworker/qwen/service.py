"""Remote Qwen Image 2.1 conditioning: peer encodes, caller samples."""

from __future__ import annotations

# Rust runtime callback exports.
import uuid as uuid
from ..core.execution import (
    RequestInputs as RequestInputs,
    completed_artifacts as completed_artifacts,
)
from ..core.protocol import (
    resolve_peer_model_name as resolve_peer_model_name,
    validate_cleanup_capability as validate_cleanup_capability,
)
from ..core.transport import HttpComfyApi as HttpComfyApi

from .._native import workflow_build

from ..preparation.capabilities import Preparation

import hashlib
import io
from typing import Any

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
    _validate_policy as _validate_policy,
)
from ..core.result_cache import ByteResultCache

QWEN_WORKER = "RemoteQwenImage21Encode"
SAVE_REMOTE_CONDITIONING = "SaveRemoteConditioning"
POSITIVE_ARTIFACT = RemoteArtifactSpec(
    "positive", "save_positive", "conditioning", ".cond"
)
NEGATIVE_ARTIFACT = RemoteArtifactSpec(
    "negative", "save_negative", "conditioning", ".cond"
)
LATENT_ARTIFACT = RemoteArtifactSpec("latent", "save_latent", "latents", ".latent")
DEFAULT_QWEN_CLIP = default_value("qwen_clip", "")
DEFAULT_QWEN_VAE = default_value("qwen_vae", "")
_MAX_IMAGES = 16
_CACHE = ByteResultCache(max_entries=4, max_bytes=512 * 1024 * 1024)


def _validate_choice(name: str, value: Any) -> str:
    from .._native import clean_model_name_native

    return clean_model_name_native(value, "Remote Qwen Image 2.1", name, False)


def _validate_connection(
    server_url: Any, timeout_seconds: Any, remote_text_encoder: Any, vae_name: Any
) -> tuple[str, int, str, str]:
    from .._native import qwen_connection_native

    return qwen_connection_native(
        server_url, timeout_seconds, remote_text_encoder, vae_name
    )


def _validate_dynamic_peer_models(
    remote_text_encoder: Any, vae_name: Any
) -> bool | str:
    from .._native import encoder_local_inputs_native

    return encoder_local_inputs_native("qwen", locals())


def _ordered_qwen_image_frames(**images: Any) -> list[Any]:
    from .._native import qwen_socket_count_native, validate_qwen_frame_count_native

    count = qwen_socket_count_native(images)
    frames = ordered_image_frames(count, **images)
    validate_qwen_frame_count_native(len(frames))
    return frames


def _validate_qwen_capability(info: dict[str, Any]) -> None:
    from .._native import workflow_validate

    workflow_validate("qwen", info, "", 0)


def _validate_peer_loaders(
    api: Any, clip: ClipSelection, vae_name: str
) -> tuple[ClipSelection, str]:
    from .._native import peer_loaders_native
    import sys

    return peer_loaders_native("qwen", sys.modules[__name__], api, clip, (vae_name,))


def build_workflow(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    resolution: int,
    negative_vision: bool,
    image_refs: list[str],
    request_id: str,
    peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
) -> dict[str, Any]:
    return workflow_build("qwen", locals())


def _cache_key(
    *,
    clip: ClipSelection,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    resolution: int,
    negative_vision: bool,
    frames: list[Any],
    server_url: str,
) -> tuple[Any, ...]:
    from .._native import cache_identity_native

    return cache_identity_native(
        "qwen",
        dict(
            clip=clip,
            vae_name=vae_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            negative_vision=negative_vision,
            frames=tuple(
                hashlib.sha256(encode_image_frame_png(frame)).hexdigest()
                for frame in frames
            ),
            server_url=server_url,
        ),
    )


def _decode(
    positive: bytes, negative: bytes, latent: bytes
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    latent_value = torch.load(
        io.BytesIO(latent), map_location="cpu", weights_only=False
    )
    from .._native import validate_latent_artifact_native

    validate_latent_artifact_native(latent_value, "Remote Qwen Image 2.1")
    return (
        torch.load(io.BytesIO(positive), map_location="cpu", weights_only=False),
        torch.load(io.BytesIO(negative), map_location="cpu", weights_only=False),
        latent_value,
    )


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
    from .._native import encoder_execute_native
    import sys

    return encoder_execute_native("qwen", sys.modules[__name__], locals())


def preparation_inputs(values):
    from .._native import qwen_preparation_inputs_native

    return qwen_preparation_inputs_native(values)


class RemoteQwenImage21Service:
    """Transport service shared by the native Comfy schema and unit tests."""

    REMOTE_PREPARATION = Preparation("encoder", inputs=preparation_inputs)

    def run(
        self,
        prompt: str,
        negative_prompt: str,
        server_url: str,
        timeout_seconds: int,
        remote_text_encoder: str,
        vae_name: str,
        resolution: int,
        negative_vision: bool,
        peer_memory_policy: str = DEFAULT_PEER_MEMORY_POLICY,
        cache_policy: str = DEFAULT_CACHE_POLICY,
        unique_id=None,
        **images: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        from .._native import encoder_run_native
        import sys

        return encoder_run_native("qwen", sys.modules[__name__], locals())


if comfy_io is not None:

    class RemoteQwenImage21Node(comfy_io.ComfyNode):
        """Native V3 presentation of the remote Qwen encoder contract."""

        REMOTE_PREPARATION = Preparation(
            "encoder", executor=RemoteQwenImage21Service, inputs=preparation_inputs
        )

        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="RemoteQwenImage21",
                hidden=[comfy_io.Hidden.unique_id],
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
                        display_name="2. 원격 텍스트 인코더 파일명",
                        options=[DEFAULT_QWEN_CLIP],
                        default=DEFAULT_QWEN_CLIP,
                        tooltip="원격 서버에서 로드할 모델 파일명 선택. 로컬 CLIP 출력 연결 불가. 주소 입력 후 목록 새로고침.",
                    ),
                    comfy_io.Combo.Input(
                        "vae_name",
                        display_name="3. 원격 VAE 파일명",
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
                            names=[
                                f"image_{index}" for index in range(1, _MAX_IMAGES + 1)
                            ],
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
        def fingerprint_inputs(
            cls, cache_policy: str = DEFAULT_CACHE_POLICY, **_kwargs: Any
        ) -> float:
            from .._native import cache_fingerprint_native

            return cache_fingerprint_native(cache_policy, "Remote Conditioning")

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
                unique_id=cls.hidden.unique_id if cls.hidden else None,
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
