"""Instance-scoped Comfy event adapter; preserve native delivery and client routing."""

import logging
from functools import wraps


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
                    logging.getLogger(__name__).exception("RemoteWorker progress adapter failed; reconciliation remains active")
            return result

        server.send_sync = send

    def observe(self, event, data):
        if not isinstance(event, str):
            return
        from .._native import prep_observe_event

        return prep_observe_event(self.coordinator, event, data, _progress_registry)


def _progress_registry():
    from comfy_execution.progress import get_progress_state

    return get_progress_state()
