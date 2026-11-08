"""Comfy worker adapter: native histories/caches, serialized GPU, isolated remote waits."""

import atexit
import gc
import logging
import sys
import threading
import time

from .. import _native
from ..preparation.memory import snapshot
from .reuse import ModelReuse
from .slots import Slots


class ComfyHost:
    def __init__(self, entry):
        self.entry = entry
        self.slots = Slots(self.capture, self.restore)
        self.reuse = ModelReuse()
        self.server = None
        self.executors = []
        self.policy = _native.HostPolicy()
        self.previews = None
        self._threads = []
        self._thread_guard = threading.Lock()
        self._shutdown_guard = threading.Lock()

    @property
    def admission(self):
        return self.policy.admission

    def status(self):
        return _native.host_status(self)

    def share_models(self, executor):
        from comfy_execution.cache_provider import _contains_self_unequal

        return _native.host_share_models(self, executor, _contains_self_unequal)

    def capture(self, job):
        self.share_models(job.executor)
        import comfy.memory_management as memory
        import comfy_execution.progress as progress
        from comfy_execution.graph_utils import GraphBuilder

        saved = {
            "server": {
                key: getattr(self.server, key, None)
                for key in (
                    "client_id",
                    "last_node_id",
                    "last_prompt_id",
                    "workflow_metadata",
                )
            },
            "progress": progress.global_progress_registry,
            "ram": (memory.extra_ram_release_callback, memory.RAM_CACHE_HEADROOM),
            "preview": self.entry.args.preview_method,
            "prefix": tuple(
                getattr(GraphBuilder, key)
                for key in (
                    "_default_prefix_root",
                    "_default_prefix_call_index",
                    "_default_prefix_graph_index",
                )
            ),
            "models": dict(job.executor.prompt_model_tracker.models),
        }
        job.executor.prompt_model_tracker.end()
        memory.set_ram_cache_release_state(None, 0)
        return saved

    def restore(self, job, saved):
        import comfy.memory_management as memory
        import comfy.model_management as models
        import comfy_execution.progress as progress
        from comfy_execution.graph_utils import GraphBuilder

        for key, value in saved["server"].items():
            setattr(self.server, key, value)
        progress.global_progress_registry = saved["progress"]
        memory.set_ram_cache_release_state(*saved["ram"])
        self.entry.args.preview_method = saved["preview"]
        GraphBuilder.set_default_prefix(*saved["prefix"])
        job.executor.prompt_model_tracker.models = saved["models"]
        for model in saved["models"].values():
            model.set_in_use_by_current_prompt(True)
        models.interrupt_current_processing(job.cancel.is_set())

    def remote(self, operation):
        import comfy.model_management as models
        from comfy_execution.utils import get_executing_context

        from ..core.cancellation import RequestCancelled, cancellation_scope

        return _native.host_remote(
            self,
            operation,
            models,
            cancellation_scope,
            RequestCancelled,
            get_executing_context,
            sys.modules[__name__],
        )

    def executor(self):
        import comfy.model_management as models
        from execution import CacheType, PromptExecutor

        return _native.host_executor(self, PromptExecutor, CacheType, models.total_ram)

    def cancel(self, prompt_id):
        with self.queue.mutex, self.slots.guard:
            return _native.host_cancel(self, prompt_id)

    def install_cancel(self):
        from aiohttp import web
        from comfy_execution.cache_provider import register_cache_provider

        register_cache_provider(self.reuse)
        from comfy_execution.utils import get_executing_context

        from .previews import Previews

        self.previews = Previews(
            self.entry.nodes.PreviewImage,
            self.entry.nodes.NODE_CLASS_MAPPINGS,
            self.server.send_sync,
            lambda: getattr(self.slots.thread, "job", None),
            get_executing_context,
        )
        self.previews.install()
        self.queue.interrupt_if_running = self.cancel

        @web.middleware
        async def interrupt(request, handler):
            if request.method == "POST" and request.path == "/interrupt":
                try:
                    data = await request.json()
                except ValueError:
                    data = {}
                try:
                    _native.host_interrupt(self, data)
                except ValueError as error:
                    return web.json_response({"error": str(error)}, status=400)
                return web.Response(status=200)
            return await handler(request)

        # Installation precedes PromptServer.setup(), before aiohttp freezes middleware.
        self.server.app.middlewares.append(interrupt)

    def housekeeping(self):
        return _native.host_housekeeping(self, sys.modules[__name__])

    def worker(self):
        from execution import PromptQueue

        executor = None
        while not self.policy.stopping:
            executor = _native.host_worker_tick(self, executor, sys.modules[__name__], PromptQueue)

    # shortcut: 진행 중 urllib 호출은 설정 timeout까지 협력 취소 대기, 더 짧은 종료 SLA 도입 시 중단 가능한 transport 검토.
    def shutdown(self):
        with self._shutdown_guard:
            from ..preparation.integration import coordinator

            if coordinator is not None:
                coordinator.shutdown(wait=False)
            with self.slots.guard:
                _native.host_begin_shutdown(self)
            with self._thread_guard:
                workers = tuple(self._threads)
            current = threading.current_thread()
            if current in workers:
                return
            for worker in workers:
                worker.join()
            if coordinator is not None:
                coordinator.shutdown()

    def run(self, queue, server, asset_manager):
        self.queue, self.server, self.assets = queue, server, asset_manager
        with self._thread_guard:
            atexit.register(self.shutdown)
            self._threads.append(threading.current_thread())
            continuation = threading.Thread(target=self.worker, daemon=True, name="pipeline-continuation")
            continuation.start()
            self._threads.append(continuation)
        self.worker()


# Native callback namespace and public compatibility exports.
__all__ = ["gc", "logging", "snapshot", "time"]
