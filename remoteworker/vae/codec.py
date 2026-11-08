"""Torch envelope adapter; validation and traversal owned by Rust."""

from .. import _native
from .._native import (
    vae_cpu_value as cpu_value,
)
from .._native import (
    vae_restore as restore,
)
from .._native import (
    vae_validate_latent as validate_latent,
)
from .._native import (
    vae_validate_result as validate_result,
)

MAX_BYTES = 512 * 1024 * 1024


def dumps(value):
    return _native.vae_dumps(value, MAX_BYTES)


def loads(data):
    return _native.vae_loads(data, MAX_BYTES)


__all__ = [
    "MAX_BYTES",
    "cpu_value",
    "restore",
    "validate_latent",
    "validate_result",
    "dumps",
    "loads",
]
