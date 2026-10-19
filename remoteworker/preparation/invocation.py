"""Shared invocation contract, independent of ComfyUI registration."""

from contextlib import contextmanager
from contextvars import ContextVar
import inspect

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


def call_values(function, values):
    bound = inspect.signature(function).bind(None, **values)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k != "self"}
