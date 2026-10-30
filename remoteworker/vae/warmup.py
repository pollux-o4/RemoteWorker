"""Rust-owned request-scoped warmup tickets."""

from .._native import vae_schedule as schedule, vae_finish as finish

__all__ = ["schedule", "finish"]
