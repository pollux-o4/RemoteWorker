"""Release disposable result caches under RAM pressure; preserve queued handoffs."""

import gc
import time


def snapshot():
    try:
        with open("/proc/meminfo") as source:
            values = {
                line.split(":")[0]: int(line.split()[1]) * 1024 for line in source
            }
        return {
            "available": values["MemAvailable"],
            "swap_used": values["SwapTotal"] - values["SwapFree"],
        }
    except (OSError, KeyError, ValueError):
        return {}


class PressureRelief:
    def __init__(self, clear):
        self.clear = clear
        self.last = 0

    def tick(self):
        state = snapshot()
        if (
            state.get("available", 2**64) < 2 * 1024**3
            and time.monotonic() - self.last > 30
        ):
            self.clear()
            gc.collect()
            self.last = time.monotonic()
        return state
