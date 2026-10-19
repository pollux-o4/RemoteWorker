"""Native sampler counts; no hardware timing inference in the scheduling path."""

from threading import RLock


class SamplingProgress:
    def __init__(self):
        self.lock = RLock()
        self.job = None
        self.node = None
        self.value = 0
        self.total = 0

    def update(self, job, node, value, total):
        if total <= 0 or value < 0 or value > total or (total == 1 and value == 0):
            return
        with self.lock:
            self.job, self.node, self.value, self.total = job, node, value, total

    def reset(self):
        with self.lock:
            self.job = self.node = None
            self.value = self.total = 0

    def detail(self, job):
        with self.lock:
            return {"step": self.value, "total": self.total} if job == self.job else {}
