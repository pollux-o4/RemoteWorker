"""Release disposable result caches under RAM pressure; preserve queued handoffs."""

import gc
import time


from .._native import memory_snapshot as snapshot


class PressureRelief:
    def __init__(self, clear):
        self.clear = clear
        self.last = 0

    def tick(self):
        state = snapshot()
        from .._native import memory_pressure
        if memory_pressure(state.get("available"), time.monotonic(), self.last):
            self.clear()
            gc.collect()
            self.last = time.monotonic()
        return state
