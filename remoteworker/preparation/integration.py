"""ComfyUI adapter; ordinary nodes keep owning conditioning caches and decoding."""

from functools import wraps
from contextlib import contextmanager
import inspect
import os
import time
from .coordinator import Coordinator
from .invocation import call_values, is_speculative
from .gates import gate
from .plan import ENCODERS, DECODERS

coordinator = None


def current_prompt():
    from comfy_execution.progress import get_progress_state

    return get_progress_state().prompt_id


@contextmanager
def foreground_gate(url, decode=False):
    from ..core.transport import _raise_if_caller_interrupted

    while True:
        _raise_if_caller_interrupted()
        if decode and coordinator:
            coordinator.ensure_next_before_decode(url)
        with gate(url).enter(check_wait=_raise_if_caller_interrupted):
            if (
                decode
                and coordinator
                and coordinator.next_clip_missing(current_prompt(), url)
            ):
                continue
            yield
            return


def managed(function, kind=None):
    if getattr(function, "_remote_preparation", False):
        return function
    call_signature = inspect.signature(function)

    @wraps(function)
    def run(*args, **kwargs):
        values = call_signature.bind(*args, **kwargs)
        values.apply_defaults()
        url = values.arguments.get("server_url")
        if not url or is_speculative():
            return function(*args, **kwargs)
        with foreground_gate(url, kind in DECODERS):
            if (
                coordinator
                and coordinator.handoffs
                and (
                    kind in ENCODERS
                    or function.__qualname__.split(".")[0]
                    in {c.__name__ for c in coordinator.encoders.values()}
                )
            ):
                from .handoff import signature, unpack

                key = (current_prompt(), str(values.arguments.get("unique_id")))
                blob = coordinator.handoffs.take(key, signature(values.arguments))
                if blob is not None:
                    coordinator.record(
                        "conditioning_consumed",
                        prompt_id=key[0],
                        node_id=key[1],
                        peer=url,
                    )
                    return unpack(blob)
            started = time.monotonic()
            scheduler = coordinator.scheduler if coordinator else None
            result = function(*args, **kwargs)
            if scheduler:
                if kind in ENCODERS:
                    scheduler.observe_encoder(time.monotonic() - started)
            return result

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
    for name in ENCODERS | DECODERS.keys():
        if name in classes:
            classes[name].run = managed(classes[name].run, name)

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
        {k: v for k, v in classes.items() if k in ENCODERS},
        warmup,
        available,
        sampling_now,
        resolve=lambda graph, key: Resolver(graph, nodes.NODE_CLASS_MAPPINGS).inputs(
            key,
            exclude=("samples", "warmup", "latent")
            if graph[key]["class_type"] in DECODERS
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
