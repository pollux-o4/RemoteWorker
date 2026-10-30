"""ComfyUI adapter; ordinary nodes keep owning conditioning caches and decoding."""

from functools import wraps
import sys
from .. import _native, pipeline
from ..core.transport import _raise_if_caller_interrupted
from .plan import head
import inspect
import os
import time
from .coordinator import Coordinator
from .invocation import is_speculative
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
    encoders = _native.prep_register_targets(capabilities, managed)
    decoders = _native.preparation_decoders(capabilities)

    def warmup(url, model, kind):
        from ..vae.service import execute_warmup, peer_api

        api = peer_api(url, 120)
        _native.prep_require_idle_peer(api._json("/queue"))
        execute_warmup(api, vae_name=model, kind=kind, server_url=url, background=False)

    def available(url):
        from ..core.transport import HttpComfyApi

        queue = HttpComfyApi(url, 2)._json("/queue")
        return _native.prep_peer_available(queue)

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
    from ..core.cancellation import check_cancelled
    from .handoff import Handoffs
    import nodes

    coordinator = Coordinator(
        server,
        encoders,
        warmup,
        available,
        sampling_now,
        resolve=lambda graph, key: _native.prep_resolve_inputs(
            graph, nodes.NODE_CLASS_MAPPINGS, key, decoders, check_cancelled),
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

# Native callback namespace and public compatibility exports.
__all__ = ['head', 'is_speculative', 'pipeline', 'time']
