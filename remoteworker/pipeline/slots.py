"""Host locks and context capture around Rust-owned pipeline state."""

from contextlib import contextmanager
from threading import Lock, local

from .._native import Job, SlotState


class Slots:
    capacity = 2

    def __init__(self, capture, restore):
        self.capture, self.restore = capture, restore
        self.lock, self.guard, self.thread = Lock(), Lock(), local()
        self.state = SlotState()

    @property
    def jobs(self):
        return self.state.jobs

    @property
    def active(self):
        return self.state.active

    @property
    def ready(self):
        return self.state.ready

    @contextmanager
    def job(self, prompt_id, executor):
        job = Job(prompt_id, executor)
        self.thread.job = job
        with self.guard:
            self.state.start(job)
        try:
            yield job
        finally:
            with self.guard:
                self.state.finish(job)
            self.thread.job = None

    def remote(self, operation, completed=None):
        job = getattr(self.thread, "job", None)
        if job is None:
            return operation()
        saved = self.capture(job)
        with self.guard:
            self.state.suspend(job)
        self.lock.release()
        try:
            result = operation()
            if completed is not None:
                completed(result, saved)
            return result
        finally:
            with self.guard:
                self.state.completed(job)
            self.lock.acquire()
            with self.guard:
                self.state.reenter(job)
            try:
                self.restore(job, saved)
            except BaseException:
                with self.guard:
                    self.state.restore_failed(job)
                raise
            with self.guard:
                self.state.resume(job)

    def status(self):
        with self.guard:
            return self.state.status()
