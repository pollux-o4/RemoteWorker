"""Comfy cancellation adapter for Rust's audited CPU ancestor resolver."""
from ..core.cancellation import check_cancelled
from .._native import Resolver as _Resolver, UnsupportedInput


def Resolver(graph, classes):
    return _Resolver(graph, classes, check_cancelled)
