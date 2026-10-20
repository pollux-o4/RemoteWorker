"""One local compute owner, with scoped suspension during remote IO."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, local


@dataclass
class Job:
    prompt_id: str
    executor: object
    cancel: Event = field(default_factory=Event)
    phase: str = "local"


class Slots:
    capacity = 2

    def __init__(self, capture, restore):
        self.capture, self.restore = capture, restore
        self.lock = Lock()
        self.guard = Lock()
        self.thread = local()
        self.jobs = {}
        self.active = None
        self.ready = 0

    @contextmanager
    def job(self, prompt_id, executor):
        job = Job(prompt_id, executor)
        self.thread.job = job
        with self.guard:
            self.jobs[prompt_id] = job
            self.active = prompt_id
        try:
            yield job
        finally:
            with self.guard:
                self.jobs.pop(prompt_id, None)
                self.active = None
            self.thread.job = None

    def remote(self, operation):
        job = getattr(self.thread, "job", None)
        if job is None:
            return operation()
        saved = self.capture(job)
        with self.guard:
            job.phase = "remote"
            self.active = None
        self.lock.release()
        try:
            return operation()
        finally:
            with self.guard:
                self.ready += 1
                job.phase = "ready"
            self.lock.acquire()
            with self.guard:
                self.ready -= 1
            self.restore(job, saved)
            with self.guard:
                job.phase = "local"
                self.active = job.prompt_id

    def status(self):
        with self.guard:
            return {"enabled": True, "capacity": self.capacity, "local_prompt_id": self.active,
                    "jobs": [{"prompt_id": j.prompt_id, "phase": j.phase} for j in self.jobs.values()]}
