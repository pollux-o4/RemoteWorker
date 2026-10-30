"""Shared disk is a tensor transport; final media saving belongs to downstream nodes."""

from __future__ import annotations


from ..core.cancellation import check_cancelled
from .. import _native
from . import shared_store as store
from .schema import decode_fields


def options(vae_name, kind, **decode_options):
    from .._native import vae_tensor_options

    return vae_tensor_options(vae_name, kind, decode_options)


def encode(samples, path, settings):
    from nodes import VAELoader
    from .worker import decode

    return _native.vae_encode_tensor(
        samples, path, settings, lambda name: VAELoader().load_vae(name)[0], decode
    )


def consume(request_id, reported, settings):
    return _native.vae_consume(request_id, reported, settings, check_cancelled)


def execute(
    api,
    samples,
    vae_name,
    kind,
    server_url,
    memory_policy,
    decode_options,
    request_id="",
    warmup=None,
):
    return _native.vae_execute_tensor(
        api,
        samples,
        vae_name,
        kind,
        server_url,
        memory_policy,
        decode_options,
        request_id,
        warmup,
        check_cancelled,
    )


class SharedVAETensorWorker:
    DEV_ONLY = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RemoteWorker/internal"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request_id": ("STRING",),
                "vae_name": ("STRING",),
                "kind": (["image", "audio"],),
                **decode_fields(),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, request_id, vae_name, kind, **decode_options):
        receipt = store.publish(
            request_id, options(vae_name, kind, **decode_options), encode
        )
        return {"ui": {"shared_tensor": [receipt]}, "result": (receipt["path"],)}


NODE_CLASS_MAPPINGS = {"SharedVAETensorWorker": SharedVAETensorWorker}
NODE_DISPLAY_NAME_MAPPINGS = {"SharedVAETensorWorker": "Shared VAE Tensor (Worker)"}
