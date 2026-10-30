"""Caller facade for peer VAE decode and non-persistent dummy warmup."""

from __future__ import annotations

from ..preparation.capabilities import Preparation
from .. import _native


from ..core.artifacts import RemoteArtifactSpec
from ..core.transport import HttpComfyApi
from ..defaults import DEFAULT_SERVER_URL
from .schema import (
    DECODE_DEFAULTS,
    VIDEO_FPS,
    VIDEO_CRF,
    WARMUP_TYPE,
    decode_inputs,
    warmup_inputs,
    validate_model_choice,
)

RESULT = RemoteArtifactSpec("result", "save", "vae_output", ".pt")


def peer_api(server_url, timeout_seconds):
    # A ComfyUI execution thread cannot wait for another prompt on its own
    # serial queue. Reject this wiring before submitting anything.
    import sys

    module = sys.modules.get("server")
    instance = getattr(getattr(module, "PromptServer", None), "instance", None)
    port = getattr(instance, "port", None)
    own_hosts = {"127.0.0.1", "localhost", "::1"}
    if port:
        import socket

        own_hosts.add(socket.gethostname())
        try:
            import psutil

            own_hosts.update(
                address.address.split("%")[0]
                for addresses in psutil.net_if_addrs().values()
                for address in addresses
            )
        except ImportError:
            pass
    _native.vae_validate_peer_url(server_url, port, list(own_hosts))
    return HttpComfyApi(server_url, timeout_seconds)


def execute_decode(api, *, samples, vae_name, kind, options, memory_policy):
    from .codec import MAX_BYTES

    return _native.vae_execute_decode(
        api,
        samples=samples,
        vae_name=vae_name,
        kind=kind,
        options=options,
        memory_policy=memory_policy,
        max_bytes=MAX_BYTES,
    )


class RemoteVAEDecode:
    REMOTE_PREPARATION = Preparation("decoder", kind="image")

    KIND = "image"
    DESCRIPTION = "Return IMAGE through a peer VAE. Both transfer modes return tensors; HTTP limit 512 MiB."
    SEARCH_ALIASES = ["remote decode", "원격 VAE", "영상 디코딩"]
    OUTPUT_TOOLTIPS = (
        "Decoded tensor for preview or downstream saving in either transfer mode.",
        "Reserved legacy file output; inactive.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return decode_inputs(cls.KIND)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        vae_name,
        output_mode="tensor",
        server_url=DEFAULT_SERVER_URL,
        audio_vae_name="",
    ):
        probe, valid = _native.vae_decode_validation(vae_name, output_mode)
        if not probe:
            return valid
        from .delivery import check_shared

        try:
            api = peer_api(server_url, 5)
            preflight(api, vae_name, "SharedVAETensorWorker")
            check_shared(api)
        except (OSError, ValueError, RuntimeError) as exc:
            return str(exc)
        return True

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_path")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "RemoteWorker/VAE"

    def check_lazy_status(
        self, samples=None, request_id="", output_mode="tensor", **kwargs
    ):
        return _native.vae_lazy_status(samples, request_id, output_mode)

    def run(
        self,
        samples,
        vae_name,
        server_url=DEFAULT_SERVER_URL,
        timeout_seconds=600,
        memory_policy="release_after_result",
        tiled=False,
        tile_size=DECODE_DEFAULTS["tile_size"],
        overlap=DECODE_DEFAULTS["overlap"],
        temporal_size=DECODE_DEFAULTS["temporal_size"],
        temporal_overlap=DECODE_DEFAULTS["temporal_overlap"],
        warmup=None,
        output_mode="tensor",
        fps=VIDEO_FPS,
        audio_vae_name="",
        crf=VIDEO_CRF,
        request_id="",
    ):
        from .codec import MAX_BYTES
        from ..core.cancellation import check_cancelled
        from comfy_execution.graph_utils import ExecutionBlocker

        api = peer_api(server_url, timeout_seconds)
        result = _native.vae_run_decode(
            api,
            samples,
            vae_name,
            self.KIND,
            server_url,
            memory_policy,
            {
                "tiled": tiled,
                "tile_size": tile_size,
                "overlap": overlap,
                "temporal_size": temporal_size,
                "temporal_overlap": temporal_overlap,
            },
            warmup,
            output_mode,
            request_id,
            MAX_BYTES,
            check_cancelled,
        )
        outputs = (result, ExecutionBlocker(None))
        if output_mode == "shared":
            return {"result": outputs, "ui": {"shared_transfer": [self.KIND]}}
        return outputs


class RemoteVAEDecodeAudio(RemoteVAEDecode):
    REMOTE_PREPARATION = Preparation("decoder", kind="audio")
    KIND = "audio"
    DESCRIPTION = "Return AUDIO through a peer VAE; preserve the native sample rate."
    SEARCH_ALIASES = ["remote audio", "원격 오디오", "오디오 디코딩"]
    OUTPUT_TOOLTIPS = (
        "Native waveform and sample rate in either transfer mode.",
        "Reserved legacy file output; inactive.",
    )
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")


class RemoteVAEWarmup:
    DESCRIPTION = "Connect readiness to the remote decoder; pass conditioning through to order preparation."
    SEARCH_ALIASES = ["preload VAE", "VAE 웜업", "VAE 미리 준비"]
    OUTPUT_TOOLTIPS = (
        "Connect to the remote decoder's warmup input.",
        "Unchanged input; connect to sampler conditioning to order warmup before sampling.",
    )

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

    def run(
        self,
        vae_name,
        kind,
        server_url=DEFAULT_SERVER_URL,
        timeout_seconds=600,
        background=True,
        after=None,
    ):
        ticket = execute_warmup(
            peer_api(server_url, timeout_seconds),
            vae_name=vae_name,
            kind=kind,
            server_url=server_url,
            background=background,
        )
        return ticket, after


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (RemoteVAEDecode, RemoteVAEDecodeAudio, RemoteVAEWarmup)
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteVAEDecode": "Remote VAE Decode",
    "RemoteVAEDecodeAudio": "Remote VAE Decode Audio",
    "RemoteVAEWarmup": "Remote VAE Warmup",
}

from .._native import vae_preflight as preflight, vae_build_workflow as build_workflow

from .._native import vae_execute_warmup as execute_warmup

__all__ = [
    "RESULT",
    "peer_api",
    "preflight",
    "build_workflow",
    "execute_decode",
    "execute_warmup",
    "RemoteVAEDecode",
    "RemoteVAEDecodeAudio",
    "RemoteVAEWarmup",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
