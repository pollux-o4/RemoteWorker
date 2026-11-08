"""Rust-owned per-job signature, bounded serialization and one-shot transfer."""

from .._native import (
    Handoffs,
)
from .._native import (
    handoff_pack as pack,
)
from .._native import (
    handoff_signature as signature,
)
from .._native import (
    handoff_unpack as unpack,
)

# Native callback namespace and public compatibility exports.
__all__ = ["Handoffs", "pack", "signature", "unpack"]
