"""Rust-owned per-job signature, bounded serialization and one-shot transfer."""
from .._native import Handoffs, handoff_signature as signature, handoff_pack as pack, handoff_unpack as unpack
