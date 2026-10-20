"""Public VAE node inputs and semantic connection contracts."""
from __future__ import annotations

import math

from ..defaults import DEFAULT_SERVER_URL

WARMUP_TYPE = "REMOTE_VAE_WARMUP"
LATENT_TYPE = "REMOTE_VAE_LATENT"
VIDEO_FPS, VIDEO_CRF = 24.0, 18
DECODE_DEFAULTS = dict(tiled=False, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)


def connection_inputs():
    return {
        "vae_name": ([""], {"default": "", "tooltip": "Choose a VAE from the peer model list. No VAE is loaded on this server."}),
        "server_url": ("STRING", {"default": DEFAULT_SERVER_URL, "tooltip": "ComfyUI that will load and run the VAE. Use a different server from this workflow."}),
        "timeout_seconds": ("INT", {"default": 600, "min": 5, "max": 1800, "advanced": True, "tooltip": "Maximum wait for each peer request, in seconds."}),
    }


def memory_input():
    return (["release_after_result", "keep"], {"advanced": True,
        "tooltip": "Release peer models after saving the result, or retain them for the next decode."})


def decode_fields():
    return {
        "tiled": ("BOOLEAN", {"default": False, "tooltip": "Enable native tiled decoding. H3 video VAE already tiles internally."}),
        "tile_size": ("INT", {"default": DECODE_DEFAULTS["tile_size"], "min": 64, "max": 4096, "step": 32, "advanced": True}),
        "overlap": ("INT", {"default": DECODE_DEFAULTS["overlap"], "min": 0, "max": 4096, "advanced": True}),
        "temporal_size": ("INT", {"default": DECODE_DEFAULTS["temporal_size"], "min": 8, "max": 4096, "advanced": True}),
        "temporal_overlap": ("INT", {"default": DECODE_DEFAULTS["temporal_overlap"], "min": 4, "max": 4096, "advanced": True}),
    }

def decode_inputs(kind):
    required = {"samples": ("LATENT", {"lazy": True, "tooltip": "Sampler output. H3 combined video/audio latents are supported."}), **connection_inputs(),
        "memory_policy": memory_input()}
    if kind == "image":
        required.update(decode_fields())
    return {"required": required, "optional": {
        "warmup": (WARMUP_TYPE, {"forceInput": True, "tooltip": "Optional manual warmup ticket."}),
        "output_mode": (["shared", "tensor"], {"default": "shared", "tooltip": "Both modes return IMAGE/AUDIO. Shared: transfer via temporary shared tensor files. Tensor: HTTP transfer (512 MiB limit)."}),
        **({name: value for name, value in video_fields().items() if name != "tiled"} if kind == "image" else {}),
        "request_id": ("STRING", {"default": "", "advanced": True, "tooltip": "Retry a retained shared job without rerunning its sampler."}),
    }}


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


def video_fields():
    """Shared caller/worker video settings; ranges also drive validation."""
    return {
        'fps': ('FLOAT', {'default': VIDEO_FPS, 'min': 1.0, 'max': 120.0}),
        'audio_vae_name': ([''], {'default': '', 'tooltip': 'H3 오디오 VAE 모델명. 비우면 저장된 AUDIO를 사용하거나 무음으로 저장합니다.'}),
        'crf': ('INT', {'default': VIDEO_CRF, 'min': 0, 'max': 51, 'advanced': True}),
        'tiled': ('BOOLEAN', {'default': False, 'advanced': True}),
    }


def video_inputs():
    return {'required': {**connection_inputs(), **video_fields(),
        'memory_policy': memory_input()}, 'optional': {
        'latent': (LATENT_TYPE,),
        'request_id': ('STRING', {'default': '', 'tooltip': '재시도 작업 ID. 완료된 결과도 재조회할 수 있습니다.'}),
        'warmup': (WARMUP_TYPE,),
    }}


def video_options(vae_name, audio_vae_name, fps, crf, tiled, kind="image"):
    if kind not in ("image", "audio"):
        raise ValueError("Unknown shared media kind")
    if validate_model_choice(vae_name) is not True:
        raise ValueError('영상 VAE를 선택하세요.')
    if not isinstance(audio_vae_name, str):
        raise ValueError('Invalid audio VAE name')
    fields = video_fields()
    for name, value, types in (('fps', fps, (int, float)), ('crf', crf, (int,))):
        limits = fields[name][1]
        if type(value) not in types or not math.isfinite(value) or not limits['min'] <= value <= limits['max']:
            raise ValueError(f'Video {name} must be between {limits["min"]} and {limits["max"]}')
    result = dict(vae_name=vae_name, audio_vae_name=audio_vae_name, fps=fps, crf=crf, tiled=bool(tiled))
    if kind == "audio":
        result["kind"] = kind
    return result
