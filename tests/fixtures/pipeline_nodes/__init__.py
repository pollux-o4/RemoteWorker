"""Opt-in CPU nodes for real Comfy queue/pipeline integration tests."""

import json
import os
import sys
import threading
import time
from functools import lru_cache

if os.environ.get("REMOTEWORKER_PIPELINE_FIXTURE") != "1":
    raise RuntimeError("Pipeline fixtures require REMOTEWORKER_PIPELINE_FIXTURE=1")

import torch
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context

_log_lock = threading.Lock()


def record(event):
    context = get_executing_context()
    if context is None:
        raise RuntimeError("Fixture node executed without native context")
    if event.startswith(("sample", "preview", "model")):
        assert get_progress_state().prompt_id == context.prompt_id
    with _log_lock:
        with open(os.environ["REMOTEWORKER_PIPELINE_TEST_LOG"], "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": event,
                        "at": time.monotonic(),
                        "prompt": context.prompt_id,
                    }
                )
                + "\n"
            )


def _package_module(suffix):
    matches = [module for name, module in sys.modules.copy().items() if name.endswith(".remoteworker." + suffix) or name == "remoteworker." + suffix]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one loaded RemoteWorker {suffix}, got {len(matches)}")
    return matches[0]


class PipelineSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag": ("STRING",),
                "delay": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {"model": ("MODEL",)},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "test/pipeline"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, tag, delay, model=None):
        import comfy.model_management as models

        record("sample-start-" + tag)
        until = time.monotonic() + delay
        while time.monotonic() < until:
            models.throw_exception_if_processing_interrupted()
            time.sleep(0.025)
        record("sample-end-" + tag)
        return ({"samples": torch.zeros(1, 4, 2, 2)},)


def _decode(self, samples, server_url, delay):
    cancellation = _package_module("core.cancellation")
    record("decode-start")
    until = time.monotonic() + delay
    while time.monotonic() < until:
        time.sleep(0.025)
        cancellation.check_cancelled()
    record("decode-end")
    return (torch.ones(1, 16, 16, 3),)


@lru_cache(maxsize=1)
def _managed_decode():
    return _package_module("preparation.integration").managed(_decode, "decoder")


class PipelineRemoteDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "server_url": (
                    "STRING",
                    {"default": "http://pipeline-fixture.invalid"},
                ),
                "delay": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "test/pipeline"

    def run(self, samples, server_url, delay):
        return _managed_decode()(None, samples=samples, server_url=server_url, delay=delay)


class PipelineSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "tag": ("STRING",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "test/pipeline"

    def run(self, image, tag):
        record("preview-" + tag)
        return {"ui": {"tag": [tag], "shape": [list(image.shape)]}, "result": ()}


class FakeModel:
    pass


class PipelineModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"strength": ("FLOAT", {"default": 1.0})},
            "optional": {"token": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "test/pipeline"

    def run(self, strength, token=""):
        record("model-load-" + str(strength))
        return (FakeModel(),)


NODE_CLASS_MAPPINGS = {node.__name__: node for node in (PipelineSampler, PipelineRemoteDecode, PipelineSink, PipelineModel)}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
