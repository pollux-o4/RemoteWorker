"""Bounded lookahead: next conditioning, then the current job's first VAE."""

from .. import _native
import atexit
import logging
from collections import deque
from threading import Event, Lock, Thread
from . import plan
from .. import pipeline
from .actions import prefetch, prepare_vae
from .memory import snapshot
from ..core.cancellation import RequestCancelled


class Coordinator:
    def __init__(
        self,
        server,
        encoders,
        warmup,
        available=lambda url: True,
        sampling=None,
        resolve=None,
        handoffs=None,
        relief=None,
        scheduler=None,
    ):
        self.server, self.encoders, self.warmup = server, encoders, warmup
        self.available = available
        self.resolve, self.handoffs, self.relief = resolve, handoffs, relief
        self.scheduler = scheduler
        self.wake = Event()
        self.ticks = 0
        self.sampling = sampling or (
            lambda prompt_id, graph: plan.is_sampling(graph, self.server.last_node_id)
        )
        self.stop = Event()
        self.lock = Lock()
        self.events = deque(maxlen=30)
        self.done = deque(maxlen=128)
        self.active = None
        self.thread = None

    def queue(self):
        return _native.prep_queue(self)

    def record(self, state, **fields):
        with self.lock:
            _native.prep_record(self, state, fields)

    def status(self):
        with self.lock:
            return _native.prep_status(self)

    def eligible(self, prompt_id):
        return _native.prep_eligible(self, prompt_id)

    def job_present(self, prompt_id, node_id, expected):
        return _native.prep_job_present(self, prompt_id, node_id, expected)

    def prompt_present(self, prompt_id):
        return _native.prep_prompt_present(self, prompt_id)

    def decoder(self, current):
        return _native.prep_decoder(self, current)

    def tick(self):
        return _native.prep_tick(self)

    def prepare_next(self, current, pending=None, *, allow_after_sampling=False, peer=None):
        return _native.prep_prepare_next(self, current, pending, allow_after_sampling=allow_after_sampling, peer=peer)

    def encoder_started(self, prompt_id, node_id):
        return _native.prep_encoder_started(self, prompt_id, node_id)

    def next_clip_missing(self, current_id, url):
        return _native.prep_next_clip_missing(self, current_id, url)

    def ensure_next_before_decode(self, url, *, current_id=None, following=None):
        return _native.prep_ensure_next_before_decode(self, url, current_id=current_id, following=following)

    def preparation_cancelled(self, current_id, following_id, expected):
        return _native.prep_cancelled(self, current_id, following_id, expected)

    def local_peer(self, url):
        return _native.prep_local_peer(self, url)

    def run(self):
        while not self.stop.is_set():
            self.wake.wait(5)
            self.wake.clear()
            if self.stop.is_set():
                break
            try:
                self.tick()
            except RequestCancelled:
                pass  # The action owns cancellation identity and emits its own event.
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Remote preparation skipped: %s", exc
                )
                self.record("error", reason=str(exc)[:200])

    def shutdown(self, *, wait=True):
        self.stop.set()
        self.wake.set()
        if wait and self.thread is not None:
            from threading import current_thread
            if self.thread is not current_thread():
                self.thread.join()

    def start(self):
        if self.thread is not None:
            return
        self.wake.set()
        self.thread = Thread(target=self.run, daemon=True, name="remote-preparation")
        atexit.register(self.shutdown)
        self.thread.start()

# Native callback namespace and public compatibility exports.
__all__ = ['pipeline', 'prefetch', 'prepare_vae', 'snapshot']
