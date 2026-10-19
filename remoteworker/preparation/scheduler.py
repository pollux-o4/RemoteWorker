"""Step policy, VAE preparation receipts and observable execution timings."""

from dataclasses import asdict
from threading import RLock
import time
from .progress import SamplingProgress
from .policy import Admission


class Scheduler:
    def __init__(self):
        self.sampling = SamplingProgress()
        self.policy = Admission()
        self.lock = RLock()
        self.reservations = set()
        self.last = {}
        self.counts = {"encoder": 0, "vae": 0}
        self.durations = {}

    def observe_encoder(self, seconds):
        self.observe("encoder", seconds)

    def observe_vae(self, seconds):
        self.observe("vae", seconds)

    def observe(self, kind, seconds):
        with self.lock:
            self.counts[kind] += 1
            self.durations[kind] = seconds

    def retain(self, ids):
        with self.lock:
            self.reservations = {r for r in self.reservations if r[0] in ids}
            if self.sampling.job not in ids:
                self.sampling.reset()
            if self.last.get("prompt_id") not in ids:
                self.last = {}

    def reserve(self, job, decoder):
        _, url, model, _ = decoder
        with self.lock:
            self.reservations.add((job, url.rstrip("/"), model))

    def invalidate_vae(self, job, url):
        with self.lock:
            self.reservations = {
                r for r in self.reservations if r[:2] != (job, url.rstrip("/"))
            }

    def reserved(self, job, decoder):
        if decoder is None:
            return False
        _, url, model, _ = decoder
        with self.lock:
            return (job, url.rstrip("/"), model) in self.reservations

    def decide(self, job, following=None, *, needs_clip=False, decoder=None):
        progress = self.sampling.detail(job)
        steps = (
            max(0, progress["total"] - progress["step"])
            if progress.get("total")
            else None
        )
        result = self.policy.decide(
            remaining_steps=steps,
            needs_clip=needs_clip,
            reserved=self.reserved(job, decoder),
        )
        with self.lock:
            self.last = {
                **asdict(result),
                "prompt_id": job,
                "next_prompt_id": following,
                **progress,
                "at": time.time(),
            }
        return result

    def status(self):
        with self.lock:
            return {
                "policy": "clip_first_two_steps",
                "vae_remaining_steps": self.policy.vae_steps,
                "decision": dict(self.last),
                "measured_samples": dict(self.counts),
                "last_seconds": dict(self.durations),
                "sampling": {
                    "prompt_id": self.sampling.job,
                    **self.sampling.detail(self.sampling.job),
                },
            }
