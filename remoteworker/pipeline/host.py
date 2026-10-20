"""Comfy worker adapter: native histories/caches, serialized GPU, isolated remote waits."""
import gc
import logging
import threading
import time

from .slots import Slots
from .reuse import ModelReuse


class ComfyHost:
    def __init__(self, entry):
        self.entry = entry
        self.slots = Slots(self.capture, self.restore)
        self.reuse = ModelReuse()
        self.server = None
        self.executors = []
        self.last_gc = 0
        self.pending_cancel = set()
        self.admission = None

    def status(self):
        return {**self.slots.status(), "admission": self.admission, "model_reuse_hits": self.reuse.hits}

    def share_models(self, executor):
        # Native on_store callbacks are deferred until the executor's event loop yields.
        # Publish weak references before releasing the slot from a synchronous node.
        from comfy_execution.cache_provider import _contains_self_unequal
        cache = executor.caches.outputs
        if not getattr(cache, "initialized", False):
            return
        for node_id in cache.cache_key_set.all_node_ids():
            key = cache.cache_key_set.get_data_key(node_id)
            if key is None or _contains_self_unequal(key):
                continue
            context = cache._build_context(node_id, key)
            if context is None or not self.reuse.should_cache(context):
                continue
            value = cache.get_local(node_id)
            if value is not None and self.reuse.should_cache(context, value):
                self.reuse.remember((context.class_type, context.cache_key_hash), value.outputs)

    def capture(self, job):
        self.share_models(job.executor)
        import comfy.memory_management as memory
        import comfy_execution.progress as progress
        from comfy_execution.graph_utils import GraphBuilder
        saved = {
            "server": {key: getattr(self.server, key, None) for key in
                       ("client_id", "last_node_id", "last_prompt_id", "workflow_metadata")},
            "progress": progress.global_progress_registry,
            "ram": (memory.extra_ram_release_callback, memory.RAM_CACHE_HEADROOM),
            "preview": self.entry.args.preview_method,
            "prefix": tuple(getattr(GraphBuilder, key) for key in
                            ("_default_prefix_root", "_default_prefix_call_index", "_default_prefix_graph_index")),
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
        from ..core.cancellation import cancellation_scope, RequestCancelled
        import comfy.model_management as models
        job = getattr(self.slots.thread, "job", None)
        if job is None:
            return operation()
        models.throw_exception_if_processing_interrupted()
        def call():
            with cancellation_scope(job.cancel.is_set, caller_interrupt=False):
                return operation()
        try:
            return self.slots.remote(call)
        except RequestCancelled as exc:
            raise models.InterruptProcessingException() from exc

    def executor(self):
        from execution import PromptExecutor, CacheType
        import comfy.model_management as models
        args = self.entry.args
        kind = CacheType.CLASSIC if args.cache_classic else CacheType.LRU if args.cache_lru > 0 else CacheType.NONE if args.cache_none else CacheType.RAM_PRESSURE
        ram = min(10.0, max(2.0, models.total_ram * 0.10 / 1024.0))
        inactive = min(128.0, models.total_ram / 1024.0)
        if args.cache_ram:
            ram = args.cache_ram[0]
            if len(args.cache_ram) > 1:
                inactive = args.cache_ram[1]
        executor = PromptExecutor(self.server, cache_type=kind,
            cache_args={"lru": args.cache_lru, "ram": ram, "ram_inactive": inactive}, asset_manager=self.assets)
        self.executors.append(executor)
        return executor

    def cancel(self, prompt_id):
        with self.queue.mutex, self.slots.guard:
            if not any(item[1] == prompt_id for item in self.queue.currently_running.values()):
                return False
            job = self.slots.jobs.get(prompt_id)
            if job is None:
                if any(item[1] == prompt_id for item in self.queue.currently_running.values()):
                    self.pending_cancel.add(prompt_id)
                    return True
                return False
            job.cancel.set()
            if self.slots.active == prompt_id:
                self.entry.nodes.interrupt_processing()
            return True

    def install_cancel(self):
        from aiohttp import web
        from comfy_execution.cache_provider import register_cache_provider
        register_cache_provider(self.reuse)
        self.queue.interrupt_if_running = self.cancel
        @web.middleware
        async def interrupt(request, handler):
            if request.method == "POST" and request.path == "/interrupt":
                try:
                    data = await request.json()
                except ValueError:
                    data = {}
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    self.cancel(prompt_id)
                else:
                    for job in self.status()["jobs"]:
                        self.cancel(job["prompt_id"])
                return web.Response(status=200)
            return await handler(request)
        # Installation precedes PromptServer.setup(), before aiohttp freezes middleware.
        self.server.app.middlewares.append(interrupt)

    def housekeeping(self):
        if self.queue.get_tasks_remaining():
            return
        flags = self.queue.get_flags()
        if flags.get("unload_models", flags.get("free_memory", False)):
            self.entry.comfy.model_management.unload_all_models()
        if flags.get("free_memory"):
            self.reuse.clear()
            for executor in self.executors:
                executor.reset()
        if time.monotonic() - self.last_gc >= 10 or flags:
            gc.collect()
            self.entry.comfy.model_management.soft_empty_cache()
            self.entry.hook_breaker_ac10a0.restore_functions()
            self.assets.queue_output_scan()
            self.assets.resume_background_scan()
            self.last_gc = time.monotonic()

    def worker(self):
        from execution import PromptQueue
        executor = None
        while True:
            if self.admission == "ram_pressure":
                time.sleep(0.1)
            if self.slots.ready:
                time.sleep(0.01)
                continue
            with self.slots.lock:
                if self.slots.ready:
                    continue
                if executor is None:
                    executor = self.executor()
                from ..preparation.memory import snapshot
                if self.slots.jobs and snapshot().get("available", 2**64) < 2 * 1024**3:
                    self.admission = "ram_pressure"
                    continue
                self.admission = None
                # Keep the next prompt authoritative/pending until the GPU slot is free.
                item = self.queue.get(timeout=0.1)
                if item is None:
                    self.housekeeping()
                    continue
                prompt, item_id = item
                prompt_id = prompt[1]
                with self.slots.job(prompt_id, executor) as job:
                    started = time.perf_counter()
                    self.assets.pause_background_scan()
                    self.server.last_prompt_id = prompt_id
                    self.server.workflow_metadata = prompt[3].get("workflow_metadata", {})
                    extra = dict(prompt[3], **prompt[5])
                    try:
                        with self.queue.mutex:
                            cancelled = prompt_id in self.pending_cancel or job.cancel.is_set()
                            self.pending_cancel.discard(prompt_id)
                        if cancelled:
                            executor.status_messages = []
                            executor.add_message("execution_interrupted", {"prompt_id": prompt_id, "node_id": None, "executed": []}, broadcast=True)
                            executor.success = False
                            executor.history_result = {"outputs": {}}
                        else:
                            executor.execute(prompt[2], prompt_id, extra, prompt[4])
                    except Exception:
                        logging.getLogger(__name__).exception("Pipeline execution failed")
                        executor.success = False
                        executor.history_result = {"outputs": {}}
                    finally:
                        self.queue.task_done(item_id, executor.history_result,
                            status=PromptQueue.ExecutionStatus(status_str="success" if executor.success else "error",
                                completed=executor.success, messages=executor.status_messages),
                            process_item=lambda value: value[:5] + value[6:])
                        self.server.send_sync("executing", {"node": None, "prompt_id": prompt_id}, self.server.client_id)
                        self.server.workflow_metadata = {}
                        logging.info("Prompt executed in %.2f seconds", time.perf_counter() - started)
                self.housekeeping()

    def run(self, queue, server, asset_manager):
        self.queue, self.server, self.assets = queue, server, asset_manager
        # Middleware must be installed during node registration, not after server startup.
        threading.Thread(target=self.worker, daemon=True, name="pipeline-continuation").start()
        self.worker()
