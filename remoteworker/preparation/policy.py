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
        if needs_clip:
            return Decision("prefetch", "clip_required", remaining_steps)
        if reserved:
            return Decision("hold", "vae_ready", remaining_steps)
        if remaining_steps is None:
            return Decision("wait", "sampling_progress")
        if remaining_steps > self.vae_steps:
            return Decision("wait", "clip_window", remaining_steps)
        return Decision("warmup", "vae_window", remaining_steps)
