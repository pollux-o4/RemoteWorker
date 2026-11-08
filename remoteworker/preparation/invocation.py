"""Shared invocation contract, independent of ComfyUI registration."""

from contextlib import contextmanager
from contextvars import ContextVar

from .._native import prep_bind_values as call_values

_speculative = ContextVar("remote_preparation_speculative", default=False)


def is_speculative():
    return _speculative.get()


@contextmanager
def speculative_call():
    token = _speculative.set(True)
    try:
        yield
    finally:
        _speculative.reset(token)


# Native callback namespace and public compatibility exports.
__all__ = ["call_values"]
