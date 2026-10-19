"""Per-peer serialization with foreground priority over speculative work."""

from contextlib import contextmanager
from threading import Condition, Lock


class PeerGate:
    def __init__(self):
        self.condition = Condition()
        self.busy = False
        self.foreground = 0

    @contextmanager
    def enter(self, speculative=False, check_wait=None):
        with self.condition:
            if not speculative:
                self.foreground += 1
            try:
                ready = lambda: (
                    not self.busy and (not speculative or not self.foreground)
                )
                if check_wait is None:
                    self.condition.wait_for(ready)
                else:
                    while not ready():
                        check_wait()
                        self.condition.wait(0.25)
                    check_wait()
                self.busy = True
            finally:
                if not speculative:
                    self.foreground -= 1
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()


_lock, _peers = Lock(), {}


def gate(url):
    with _lock:
        return _peers.setdefault(url.rstrip("/"), PeerGate())
