"""Comfy cancellation adapter for Rust's audited CPU ancestor resolver."""

from .._native import Resolver as _Resolver
from .._native import UnsupportedInput
from ..core.cancellation import check_cancelled


def Resolver(graph, classes):
    return _Resolver(graph, classes, check_cancelled)


# Native callback namespace and public compatibility exports.
__all__ = ["UnsupportedInput"]
