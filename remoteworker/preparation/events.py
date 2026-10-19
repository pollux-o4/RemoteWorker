"""Instance-scoped Comfy event adapter; preserve native delivery and client routing."""

from functools import wraps
import logging


class EventBridge:
    def __init__(self, server, coordinator):
        self.server, self.coordinator = server, coordinator
        self.original = server.send_sync
        self.errors = 0

        @wraps(self.original)
        def send(event, data, sid=None):
            result = self.original(event, data, sid)
            try:
                self.observe(event, data)
            except Exception:
                self.errors += 1
                if self.errors == 1:
                    logging.getLogger(__name__).exception(
                        "RemoteWorker progress adapter failed; reconciliation remains active"
                    )
            return result

        server.send_sync = send

    def observe(self, event, data):
        if event == "progress_state" and isinstance(data, dict):
            from comfy_execution.progress import get_progress_state

            registry = get_progress_state()
            job = data.get("prompt_id")
            if registry.prompt_id == job:
                for key, value in data.get("nodes", {}).items():
                    if value.get("state") != "running":
                        continue
                    node = registry.dynprompt.get_node(key)
                    if "Sampler" in node.get("class_type", ""):
                        self.coordinator.scheduler.sampling.update(
                            job, key, value.get("value", 0), value.get("max", 1)
                        )
            self.coordinator.wake.set()
        elif event in {
            "status",
            "executing",
            "execution_start",
            "execution_success",
            "execution_error",
            "execution_interrupted",
        }:
            self.coordinator.wake.set()
