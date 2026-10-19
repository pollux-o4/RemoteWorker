"""Bounded lookahead: next conditioning, then the current job's first VAE."""

import logging
from collections import deque
from threading import Event, Lock, Thread
import time
from . import plan
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

    def queue(self):
        queue = self.server.prompt_queue
        return getattr(queue, "get_current_queue_volatile", queue.get_current_queue)()

    def record(self, state, **fields):
        with self.lock:
            value = {"state": state, **fields}
            if (
                self.active
                and {k: v for k, v in self.active.items() if k != "at"} == value
            ):
                return
            self.active = {**value, "at": time.time()}
            self.events.append(self.active.copy())

    def status(self):
        running, pending = self.queue()
        ids = {item[1] for item in [*running, *pending]}
        with self.lock:
            return {
                "enabled": True,
                "active": self.active
                if self.active and self.active.get("prompt_id") in ids
                else None,
                "events": list(self.events),
                "queue": {"running": len(running), "pending": len(pending)},
                "lookahead": 1,
                "scope": "cpu_ancestors" if self.resolve else "literal_inputs",
                "handoff_bytes": self.handoffs.bytes_used if self.handoffs else 0,
                "vae_policy": "first_decoder",
                "memory": snapshot(),
                "scheduler": {
                    **self.scheduler.status(),
                    "driver": "events",
                    "reconcile_seconds": 5,
                    "ticks": self.ticks,
                }
                if self.scheduler
                else None,
            }

    def eligible(self, prompt_id):
        running, _ = self.queue()
        return bool(
            running
            and running[0][1] == prompt_id
            and self.sampling(prompt_id, running[0][2])
        )

    def job_present(self, prompt_id, node_id, expected):
        running, pending = self.queue()
        for item in [*running, *pending]:
            if item[1] == prompt_id:
                node = item[2].get(node_id)
                return node is not None and plan.fingerprint(node) == expected
        return False

    def prompt_present(self, prompt_id):
        running, pending = self.queue()
        return any(item[1] == prompt_id for item in [*running, *pending])

    def decoder(self, current):
        graph = plan.job_graph(current)
        if not self.resolve:
            return plan.first_decoder(graph)
        adjusted = dict(graph)
        for key, node in graph.items():
            if (
                node.get("class_type") not in plan.DECODERS
                or node.get("inputs", {}).get("warmup") is not None
            ):
                continue
            try:
                values = self.resolve(graph, key)
                if values.get("vae_name_override") is not None:
                    values["vae_name"] = values["vae_name_override"]
                adjusted[key] = {**node, "inputs": values}
            except ValueError:
                adjusted[key] = {**node, "inputs": {}}
        return plan.first_decoder(adjusted)

    def tick(self):
        self.ticks += 1
        if self.relief:
            self.relief.tick()
        running, pending = self.queue()
        ids = {item[1] for item in [*running, *pending]}
        if self.handoffs:
            self.handoffs.retain(ids)
        if self.scheduler:
            self.scheduler.retain(ids)
        if not running:
            return
        current = running[0]
        if not self.eligible(current[1]):
            return
        self.prepare_next(current, pending)
        decoder = self.decoder(current)
        if self.scheduler:
            following = plan.head(pending)[1] if pending else None
            decision = self.scheduler.decide(
                current[1],
                following,
                needs_clip=bool(
                    decoder and self.next_clip_missing(current[1], decoder[1])
                ),
                decoder=decoder,
            )
            if decision.action != "warmup":
                return
        prepare_vae(self, current, decoder)

    def prepare_next(
        self, current, pending=None, *, allow_after_sampling=False, peer=None
    ):
        if pending is None:
            _, pending = self.queue()
        for following, graph, node_id, node, values, token in plan.pending_encoders(
            pending, self.done
        ):
            if self.resolve:
                try:
                    values = self.resolve(graph, node_id)
                except ValueError as exc:
                    self.done.append(token)
                    self.record(
                        "skipped",
                        prompt_id=following[1],
                        node_id=node_id,
                        reason=str(exc),
                    )
                    continue
            if values is None:
                self.done.append(token)
                self.record(
                    "skipped",
                    prompt_id=following[1],
                    node_id=node_id,
                    reason="dynamic_inputs_or_refresh",
                )
                continue
            url = values.get("server_url", "")
            if peer and url.rstrip("/") != peer.rstrip("/"):
                continue
            if not url or self.local_peer(url):
                self.done.append(token)
                self.record("skipped", reason="local_or_missing_peer", node_id=node_id)
                continue
            if not allow_after_sampling and not self.available(url):
                self.record("deferred", prompt_id=following[1], reason="peer_busy")
                return
            try:
                prefetch(
                    self,
                    current,
                    following,
                    graph,
                    node_id,
                    node,
                    values,
                    token,
                    allow_after_sampling=allow_after_sampling,
                )
            except (OSError, RuntimeError, ValueError, TimeoutError):
                self.done.append(
                    token
                )  # Failed/unsupported next encoder falls back to normal execution.
                raise

    def next_clip_missing(self, current_id, url):
        running, pending = self.queue()
        if not running or running[0][1] != current_id or not pending:
            return False
        for following, graph, node_id, node, values, token in plan.pending_encoders(
            pending, self.done
        ):
            peer = node.get("inputs", {}).get("server_url")
            if isinstance(peer, str) and peer.rstrip("/") == url.rstrip("/"):
                return True
        return False

    def ensure_next_before_decode(self, url):
        # A deleted/reordered next job is not an error in the current video's decoder.
        while True:
            running, pending = self.queue()
            if not running or not pending:
                return
            current = running[0]
            try:
                self.prepare_next(current, pending, allow_after_sampling=True, peer=url)
                if self.next_clip_missing(current[1], url):
                    continue
                return
            except RequestCancelled:
                if not self.prompt_present(current[1]):
                    return
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                self.record(
                    "error",
                    prompt_id=(self.active or {}).get("prompt_id"),
                    reason=str(exc)[:200],
                )
                return

    def preparation_cancelled(self, current_id, following_id, expected):
        running, pending = self.queue()
        active = running[0][1] if running else None
        selected = next((p for p in [*running, *pending] if p[1] == following_id), None)
        if selected is None:
            return True
        if plan.job_graph(selected) != expected:
            return True
        if active == following_id:
            return False
        if not self.prompt_present(current_id):
            return True
        return not pending or plan.head(pending)[1] != following_id

    def local_peer(self, url):
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        return (
            parsed.hostname in ("localhost", "127.0.0.1", "::1")
            and parsed.port == self.server.port
        )

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

    def start(self):
        self.wake.set()
        Thread(target=self.run, daemon=True, name="remote-preparation").start()
