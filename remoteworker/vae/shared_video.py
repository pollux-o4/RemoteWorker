"""Public shared-latent nodes and peer execution facade."""

from __future__ import annotations

from ..preparation.capabilities import Preparation
from .. import _native

from ..defaults import DEFAULT_SERVER_URL
from . import shared_store as store
from .schema import (
    LATENT_TYPE,
    VIDEO_FPS,
    VIDEO_CRF,
    video_inputs,
    video_options,
    validate_model_choice,
)
from .service import peer_api


class SaveSharedVAELatent:
    DESCRIPTION = "공유 폴더에 latent를 원자적으로 저장합니다. 영상 저장·경로 확인 성공 후에만 임시 latent를 삭제합니다. 실패한 작업 ID로 재시도할 수 있습니다."
    CATEGORY = "RemoteWorker/VAE"
    RETURN_TYPES = (LATENT_TYPE, "STRING")
    RETURN_NAMES = ("latent", "request_id")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}, "optional": {"audio": ("AUDIO",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, samples, audio=None):
        request_id = store.save_latent(samples, audio)
        return {"ui": {"text": [request_id]}, "result": (request_id, request_id)}


class LoadSharedVAELatent:
    DESCRIPTION = "실패 후 보존된 작업 ID를 입력해 공유 latent를 다시 사용합니다. 완료 후 삭제된 latent는 읽을 수 없습니다."
    CATEGORY = "RemoteWorker/VAE"
    RETURN_TYPES = (LATENT_TYPE, "LATENT")
    RETURN_NAMES = ("latent", "samples")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request_id": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, request_id):
        # LATENT output deliberately requires the input file; completed jobs can
        # be retried directly with RemoteVAEVideo's request_id override instead.
        return request_id, store.load_latent(request_id)


class RemoteVAEVideo:
    REMOTE_PREPARATION = Preparation("decoder", kind="image")

    DESCRIPTION = "공유 latent를 처리 서버에서 디코딩하고 MP4(H.264)로 저장합니다. 영상 텐서를 전송하지 않고 경로만 반환합니다. 결과 확인 후 latent 삭제; 실패 시 보존합니다."
    CATEGORY = "RemoteWorker/VAE"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return video_inputs()

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name, audio_vae_name=""):
        return validate_model_choice(vae_name)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        vae_name,
        fps=VIDEO_FPS,
        audio_vae_name="",
        crf=VIDEO_CRF,
        tiled=False,
        server_url=DEFAULT_SERVER_URL,
        timeout_seconds=600,
        memory_policy="release_after_result",
        latent=None,
        request_id="",
        warmup=None,
        kind="image",
    ):
        request_id = _native.vae_validate_retry(latent, request_id)
        store.request_path("temp", request_id, "latent.pt")
        options = video_options(vae_name, audio_vae_name, fps, crf, tiled, kind)
        path = execute_shared(
            peer_api(server_url, timeout_seconds),
            request_id,
            options,
            server_url=server_url,
            memory_policy=memory_policy,
            warmup=warmup,
        )
        return {"ui": {"text": [path]}, "result": (path,)}


def execute_shared(api, request_id, options, *, server_url, memory_policy, warmup=None):
    return _native.vae_execute_shared(
        api, request_id, options, server_url, memory_policy, warmup
    )


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (SaveSharedVAELatent, LoadSharedVAELatent, RemoteVAEVideo)
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSharedVAELatent": "Save Shared VAE Latent",
    "LoadSharedVAELatent": "Load Shared VAE Latent",
    "RemoteVAEVideo": "Remote VAE Video (Shared File)",
}
