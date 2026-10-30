"""Rust-owned shared-storage challenge handshake."""

from .._native import (
    vae_check_shared as check_shared,
    vae_receive_probe as receive_probe,
)

__all__ = ["check_shared", "receive_probe"]
