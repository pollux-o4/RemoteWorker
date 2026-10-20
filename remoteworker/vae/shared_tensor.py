"""Shared disk is a tensor transport; final media saving belongs to downstream nodes."""
from __future__ import annotations

import torch

from ..core.cancellation import check_cancelled
from . import shared_store as store
from .codec import cpu_value, restore, validate_result
from .schema import DECODE_DEFAULTS, decode_fields


def options(vae_name, kind, **decode_options):
    if kind not in ("image", "audio"):
        raise ValueError("Unknown Remote VAE output kind")
    if set(decode_options) - DECODE_DEFAULTS.keys():
        raise ValueError("Unknown shared tensor decode option")
    return dict(delivery="tensor", vae_name=vae_name, kind=kind, **(DECODE_DEFAULTS | decode_options))


def encode(samples, path, settings):
    from nodes import VAELoader
    from .worker import decode

    vae = VAELoader().load_vae(settings["vae_name"])[0]
    result = validate_result(decode(vae, samples, settings["kind"],
        **{key: settings[key] for key in DECODE_DEFAULTS}, transfer=False), settings["kind"])
    torch.save(cpu_value(result), path)
    return {"kind": settings["kind"]}


def consume(request_id, reported, settings):
    # Validate the full CPU value before acknowledging or deleting either file.
    with store.request_lock(request_id):
        receipt = store.read_receipt(request_id, settings)
        if receipt is None or receipt != reported:
            raise ValueError("공유 텐서 완료 정보가 일치하지 않습니다. 임시 파일을 보존합니다.")
        path = store.media_path(request_id, settings)
        check_cancelled()
        result = validate_result(restore(torch.load(path, map_location="cpu", weights_only=True)), settings["kind"])
        check_cancelled()
        store.request_path("temp", request_id, "latent.pt").unlink(missing_ok=True)
        path.unlink()
        store.receipt_file(request_id, settings).unlink()
        return result


def execute(api, samples, vae_name, kind, server_url, memory_policy, decode_options, request_id="", warmup=None):
    from ..core.execution import completed_prompt
    from ..core.artifacts import successful_outputs
    from ..core.protocol import build_release_cleanup_node
    from .service import preflight
    from .delivery import check_shared
    from . import warmup as warmup_tasks

    preflight(api, vae_name, "SharedVAETensorWorker", memory_policy)
    check_shared(api)
    settings = options(vae_name, kind, **decode_options)
    request_id = request_id or store.save_latent(samples)
    try:
        warmup_tasks.finish(api, warmup, vae_name=vae_name, kind=kind, server_url=server_url)
        graph = {"save": {"class_type": "SharedVAETensorWorker", "inputs": {
            "request_id": request_id, **{key: value for key, value in settings.items() if key != "delivery"}}}}
        if memory_policy == "release_after_result":
            graph["cleanup"] = build_release_cleanup_node(["save", 0])
        with completed_prompt(api, graph) as (prompt_id, history):
            receipts = successful_outputs(history, prompt_id).get("save", {}).get("shared_tensor")
            if not isinstance(receipts, list) or len(receipts) != 1:
                raise ValueError("Shared VAE tensor receipt is missing")
            return consume(request_id, receipts[0], settings)
    except Exception as exc:
        raise RuntimeError(f"공유 텐서 전달 실패 · 재시도 ID: {request_id} · 임시 파일 보존: {exc}") from exc


class SharedVAETensorWorker:
    DEV_ONLY = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "RemoteWorker/internal"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request_id": ("STRING",), "vae_name": ("STRING",),
            "kind": (["image", "audio"],), **decode_fields()}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, request_id, vae_name, kind, **decode_options):
        receipt = store.publish(request_id, options(vae_name, kind, **decode_options), encode)
        return {"ui": {"shared_tensor": [receipt]}, "result": (receipt["path"],)}


NODE_CLASS_MAPPINGS = {"SharedVAETensorWorker": SharedVAETensorWorker}
NODE_DISPLAY_NAME_MAPPINGS = {"SharedVAETensorWorker": "Shared VAE Tensor (Worker)"}
