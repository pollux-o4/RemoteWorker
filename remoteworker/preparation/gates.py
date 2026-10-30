"""Rust-owned per-peer admission and foreground priority."""
from .._native import PeerGate, gate

# Native callback namespace and public compatibility exports.
__all__ = ['PeerGate', 'gate']
