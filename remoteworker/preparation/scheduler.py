"""Rust-owned admission, reservations and sampling observations."""
from .._native import Scheduler

# Native callback namespace and public compatibility exports.
__all__ = ['Scheduler']
