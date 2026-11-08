"""Missing conditioning first; VAE warmup at two remaining sampling steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    remaining_steps: int | None = None


class Admission:
    vae_steps = 2

    def decide(self, *, remaining_steps, needs_clip, reserved):
        from .._native import admission

        return Decision(*admission(remaining_steps, needs_clip, reserved))
