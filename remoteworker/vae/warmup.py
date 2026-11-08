"""Rust-owned request-scoped warmup tickets."""

from .._native import vae_finish as finish
from .._native import vae_schedule as schedule

__all__ = ["schedule", "finish"]
