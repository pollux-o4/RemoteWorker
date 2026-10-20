"""Caller facade for peer VAE decode and non-persistent dummy warmup."""
from __future__ import annotations

import uuid

from ..core.artifacts import RemoteArtifactSpec
from ..core.execution import RequestInputs, completed_artifacts
from ..core.protocol import build_release_cleanup_node, validate_cleanup_capability
from ..core.transport import HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from .codec import dumps, loads, validate_latent, validate_result
from . import warmup as warmup_tasks
from .schema import DECODE_DEFAULTS, VIDEO_FPS, VIDEO_CRF, WARMUP_TYPE, decode_inputs, warmup_inputs, validate_model_choice

RESULT = RemoteArtifactSpec("result", "save", "vae_output", ".pt")


def peer_api(server_url, timeout_seconds):
    # A ComfyUI execution thread cannot wait for another prompt on its own
    # serial queue. Reject this wiring before submitting anything.
    from urllib.parse import urlsplit
    import sys
    module = sys.modules.get("server")
    instance = getattr(getattr(module, "PromptServer", None), "instance", None)
    port = getattr(instance, "port", None)
    url = urlsplit(server_url)
    own_hosts = {"127.0.0.1", "localhost", "::1"}
    if port and url.port == port:
        import socket
        own_hosts.add(socket.gethostname())
        try:
            import psutil
            own_hosts.update(address.address.split("%")[0]
                for addresses in psutil.net_if_addrs().values() for address in addresses)
        except ImportError:
            pass
        if url.hostname in own_hosts:
            raise ValueError("처리 서버가 현재 서버와 같습니다. 다른 ComfyUI 포트를 지정하세요.")
    return HttpComfyApi(server_url, timeout_seconds)


def preflight(api, vae_name, worker, memory_policy="keep"):
    if not isinstance(vae_name, str) or not vae_name.strip() or "\x00" in vae_name:
        raise ValueError("Remote VAE requires a model name")
    if memory_policy not in ("keep", "release_after_result"):
        raise ValueError("Unknown Remote VAE memory policy")
    api.require_node(worker)
    info = api.require_node("VAELoader")
    if vae_name not in info.get("input", {}).get("required", {}).get("vae_name", [[]])[0]:
        raise ValueError(f"처리 서버에 선택한 VAE가 없습니다: {vae_name}")
    if memory_policy == "release_after_result":
        validate_cleanup_capability(api.require_node("VRAMCleanup"))


def build_workflow(vae_name, filename, kind, options, request_id, memory_policy):
    graph = {
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "decode": {"class_type": "RemoteVAEDecodeWorker", "inputs": {
            "vae": ["vae", 0], "filename": filename, "kind": kind, **options}},
        "save": {"class_type": "SaveRemoteVAEOutput", "inputs": {
            "value": ["decode", 0], "filename_prefix": f"remote-conditioning/{request_id}/decoded"}},
    }
    if memory_policy == "release_after_result":
        graph["cleanup"] = build_release_cleanup_node(["save", 0])
    return graph


def execute_decode(api, *, samples, vae_name, kind, options, memory_policy):
    if kind not in ("image", "audio"):
        raise ValueError("Unknown Remote VAE output kind")
    preflight(api, vae_name, "RemoteVAEDecodeWorker", memory_policy)
    api.require_node("SaveRemoteVAEOutput")
    data = dumps(validate_latent(samples))
    request_id = uuid.uuid4().hex
    with RequestInputs(api, request_id) as request:
        filename = request.upload_file(filename="samples.pt", data=data, content_type="application/octet-stream")
        workflow = build_workflow(vae_name, filename, kind, options, request_id, memory_policy)
        try:
            with completed_artifacts(api, workflow, request_id, (RESULT,)) as blobs:
                return validate_result(loads(blobs["result"]), kind)
        finally:
            try:
                api.discard_vae_outputs(request_id)
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                import logging
                logging.getLogger(__name__).warning("Could not remove VAE transfer output: %s", exc)


def execute_warmup(api, *, vae_name, kind, server_url, background=False):
    if kind not in ("image", "audio"):
        raise ValueError("Unknown Remote VAE warmup kind")
    preflight(api, vae_name, "RemoteVAEWarmupWorker")
    api.require_node("SaveRemoteText")
    ticket = warmup_tasks.schedule(api, vae_name=vae_name, kind=kind, server_url=server_url)
    if background:
        return ticket
    warmup_tasks.finish(api, ticket, vae_name=vae_name, kind=kind, server_url=server_url)
    return ticket


class RemoteVAEDecode:
    KIND = "image"
    DESCRIPTION = "Decode on a peer. Shared folder saves a video and returns its path; tensor mode returns IMAGE (512 MiB limit)."
    SEARCH_ALIASES = ["remote decode", "원격 VAE", "영상 디코딩"]
    OUTPUT_TOOLTIPS = ("Active in tensor mode.", "Active in shared folder mode.")

    @classmethod
    def INPUT_TYPES(cls):
        return decode_inputs(cls.KIND)

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name, output_mode="tensor", server_url=DEFAULT_SERVER_URL, audio_vae_name=""):
        valid = validate_model_choice(vae_name)
        if valid is not True or output_mode == "tensor":
            return valid
        if output_mode != "shared":
            return "알 수 없는 결과 방식입니다."
        from .delivery import check_shared
        try:
            api = peer_api(server_url, 5)
            preflight(api, vae_name, "SharedVAEVideoWorker")
            if cls.KIND == "image" and audio_vae_name:
                preflight(api, audio_vae_name, "SharedVAEVideoWorker")
            check_shared(api)
        except (OSError, ValueError, RuntimeError) as exc:
            return str(exc)
        return True

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_path")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "RemoteWorker/VAE"

    def check_lazy_status(self, samples=None, request_id="", output_mode="tensor", **kwargs):
        return [] if output_mode == "shared" and request_id else (["samples"] if samples is None else [])

    def run(self, samples, vae_name, server_url=DEFAULT_SERVER_URL, timeout_seconds=600,
            memory_policy="release_after_result", tiled=False, tile_size=DECODE_DEFAULTS["tile_size"],
            overlap=DECODE_DEFAULTS["overlap"], temporal_size=DECODE_DEFAULTS["temporal_size"],
            temporal_overlap=DECODE_DEFAULTS["temporal_overlap"], warmup=None, output_mode="tensor",
            fps=VIDEO_FPS, audio_vae_name="", crf=VIDEO_CRF, request_id=""):
        api = peer_api(server_url, timeout_seconds)
        if output_mode == "shared":
            from .delivery import decode_shared
            return decode_shared(api, samples, vae_name, self.KIND, server_url, timeout_seconds,
                memory_policy, tiled, fps, audio_vae_name, crf, request_id, warmup)
        if output_mode != "tensor":
            raise ValueError("알 수 없는 결과 방식입니다.")
        warmup_tasks.finish(api, warmup, vae_name=vae_name, kind=self.KIND, server_url=server_url)
        result = execute_decode(api, samples=samples,
            vae_name=vae_name, kind=self.KIND, memory_policy=memory_policy, options={
                "tiled": tiled, "tile_size": tile_size, "overlap": overlap,
                "temporal_size": temporal_size, "temporal_overlap": temporal_overlap})
        from comfy_execution.graph_utils import ExecutionBlocker
        return (result, ExecutionBlocker(None))


class RemoteVAEDecodeAudio(RemoteVAEDecode):
    KIND = "audio"
    DESCRIPTION = "Decode audio on another ComfyUI server. Preserves the native decoder's waveform and sample rate."
    SEARCH_ALIASES = ["remote audio", "원격 오디오", "오디오 디코딩"]
    OUTPUT_TOOLTIPS = ("Active in tensor mode.", "Active in shared folder mode (float WAV).")
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")



class RemoteVAEWarmup:
    DESCRIPTION = "Prepare a peer VAE with a tiny dummy decode. Route conditioning through passthrough before the sampler; connect ticket to the remote decoder. No image/audio file generated."
    SEARCH_ALIASES = ["preload VAE", "VAE 웜업", "VAE 미리 준비"]
    OUTPUT_TOOLTIPS = ("Connect to the remote decoder's warmup input.", "Unchanged input; connect to sampler conditioning to order warmup before sampling.")
    @classmethod
    def INPUT_TYPES(cls):
        return warmup_inputs()

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name):
        return validate_model_choice(vae_name)

    RETURN_TYPES = (WARMUP_TYPE, "*")
    RETURN_NAMES = ("ticket", "passthrough")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RemoteWorker/VAE"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, vae_name, kind, server_url=DEFAULT_SERVER_URL, timeout_seconds=600, background=True, after=None):
        ticket = execute_warmup(peer_api(server_url, timeout_seconds), vae_name=vae_name,
            kind=kind, server_url=server_url, background=background)
        return ticket, after


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (RemoteVAEDecode, RemoteVAEDecodeAudio, RemoteVAEWarmup)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAEDecode": "Remote VAE Decode", "RemoteVAEDecodeAudio": "Remote VAE Decode Audio",
    "RemoteVAEWarmup": "Remote VAE Warmup",
}
