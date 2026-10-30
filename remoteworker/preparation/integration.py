"""ComfyUI adapter; ordinary nodes keep owning conditioning caches and decoding."""

from functools import wraps
from contextlib import contextmanager
import sys
from .. import _native, pipeline
from ..core.transport import _raise_if_caller_interrupted
from .plan import head
import inspect
import os
import time
from .coordinator import Coordinator
from .invocation import is_speculative
from .gates import gate
from .capabilities import discover

coordinator = None


class PreparedResult(Exception):
    def __init__(self, result):
        self.result = result



def current_prompt():
    from comfy_execution.progress import get_progress_state

    return get_progress_state().prompt_id


def foreground_gate(url, decode=False, check_wait=None):
    return _native.prep_foreground_lease(sys.modules[__name__], url, decode,
                                        check_wait or _raise_if_caller_interrupted)


def managed(function, kind=None):
    if getattr(function, "_remote_preparation", False):
        return function
    call_signature = inspect.signature(function)

    @wraps(function)
    def run(*args, **kwargs):
        values = call_signature.bind(*args, **kwargs)
        values.apply_defaults()
        return _native.prep_managed_call(sys.modules[__name__], function, args, kwargs,
                                         values.arguments, kind)

    run._remote_preparation = True
    return run


def register(classes):
    global coordinator
    try:
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or coordinator is not None:
        return
    if os.environ.get("REMOTEWORKER_AUTOPREPARE", "1") == "0":
        return
    capabilities = discover(classes)
    encoders = {}
    decoders = {name for name, (_, spec) in capabilities.items() if spec.role == "decoder"}
    for name, (cls, spec) in capabilities.items():
        target = spec.executor or cls
        target.run = managed(target.run, spec.role)
        if spec.role == "encoder":
            encoders[name] = target

    def warmup(url, model, kind):
        from ..vae.service import execute_warmup, peer_api

        api = peer_api(url, 120)
        # Never insert speculative work ahead of somebody else's peer queue.
        queue = api._json("/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("peer_busy")
        execute_warmup(api, vae_name=model, kind=kind, server_url=url, background=False)

    def available(url):
        from ..core.transport import HttpComfyApi

        queue = HttpComfyApi(url, 2)._json("/queue")
        return not (queue.get("queue_running") or queue.get("queue_pending"))

    from .execution_state import sampling_now
    from .memory import PressureRelief
    from ..minimax.common import clear_minimax_result_cache
    from ..conditioning.service import clear_conditioning_cache

    def clear_disposable():
        clear_minimax_result_cache()
        clear_conditioning_cache()

    relief = PressureRelief(clear_disposable)
    from .scheduler import Scheduler
    from .events import EventBridge
    from .inputs import Resolver
    from .handoff import Handoffs
    import nodes

    coordinator = Coordinator(
        server,
        encoders,
        warmup,
        available,
        sampling_now,
        resolve=lambda graph, key: Resolver(graph, nodes.NODE_CLASS_MAPPINGS).inputs(
            key,
            exclude=("samples", "warmup", "latent")
            if graph[key]["class_type"] in decoders
            else (),
        ),
        handoffs=Handoffs(),
        relief=relief,
        scheduler=Scheduler(),
    )
    coordinator.event_bridge = EventBridge(server, coordinator)

    @server.routes.get("/remoteworker/preparation")
    async def status(request):
        from aiohttp import web

        return web.json_response(coordinator.status())

    coordinator.start()
