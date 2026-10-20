"""Bounded prompt overlap; only remote IO runs outside the local execution slot."""

runtime = None


def status():
    return runtime.status() if runtime else {"enabled": False}


def install():
    import os
    import sys
    if os.environ.get("CLUSTER_RUNTIME") != "amd" or os.environ.get("REMOTEWORKER_PIPELINE", "1") == "0":
        return
    entry = sys.modules.get("__main__")
    if not callable(getattr(entry, "prompt_worker", None)):
        return
    from .host import ComfyHost
    from comfy_execution.caching import BasicCache
    import comfy_execution.progress as progress
    import comfy.memory_management as memory
    if not (hasattr(BasicCache, "_build_context") and hasattr(progress, "global_progress_registry")
            and hasattr(memory, "extra_ram_release_callback")):
        import logging
        logging.getLogger(__name__).warning("Pipeline disabled: unsupported Comfy execution state contract")
        return
    global runtime
    if runtime is None:
        runtime = ComfyHost(entry)
        from server import PromptServer
        runtime.server = PromptServer.instance
        runtime.queue = runtime.server.prompt_queue
        runtime.install_cancel()
        entry.prompt_worker = runtime.run
