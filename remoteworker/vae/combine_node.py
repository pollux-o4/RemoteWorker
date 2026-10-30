"""File-based counterpart to CreateVideo; standard VIDEO output stays composable."""

from .mux import combine
from .. import _native


class RemoteVAECombineMedia:
    CATEGORY = "RemoteWorker/VAE"
    DESCRIPTION = "공유 파일 경로 결합 · 검증 후 중간 파일 정리"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "file_path")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video_path": ("STRING", {"forceInput": True})},
            "optional": {
                "audio_path": ("STRING", {"forceInput": True}),
                "cleanup_sources": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "완성 파일 검증 후 해당 중간 MP4·WAV 삭제. 다른 노드에서도 사용하는 경우 자동 보존.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    def run(
        self,
        video_path,
        audio_path="",
        cleanup_sources=True,
        prompt=None,
        unique_id=None,
    ):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy_api.latest import InputImpl

        exclusive = _native.vae_exclusive_sources(prompt, unique_id, cleanup_sources)
        receipt = combine(
            video_path,
            audio_path,
            cleanup=exclusive,
            interrupt=throw_exception_if_processing_interrupted,
        )
        media = {
            key: receipt[key] for key in ("request_id", "path", "sources", "final")
        }
        media["url"] = "/remote-vae/media/" + receipt["request_id"]
        return {
            "ui": {
                "shared_media": [media],
                "text": [receipt["path"]],
                "source_cleanup": [exclusive],
            },
            "result": (InputImpl.VideoFromFile(receipt["path"]), receipt["path"]),
        }


NODE_CLASS_MAPPINGS = {"RemoteVAECombineMedia": RemoteVAECombineMedia}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAECombineMedia": "Remote VAE · 영상·음성 결합 (파일)"
}
