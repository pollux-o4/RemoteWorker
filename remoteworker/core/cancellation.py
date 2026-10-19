"""Optional cooperative cancellation for request-scoped background consumers."""

from contextlib import contextmanager
from contextvars import ContextVar

_check = ContextVar("remote_request_cancellation", default=None)
_caller_interrupt = ContextVar("remote_caller_interrupt", default=True)


class RequestCancelled(Exception):
    pass


@contextmanager
def cancellation_scope(check, *, caller_interrupt=True):
    token = _check.set(check)
    caller_token = _caller_interrupt.set(caller_interrupt)
    try:
        yield
    finally:
        _check.reset(token)
        _caller_interrupt.reset(caller_token)


def check_cancelled():
    check = _check.get()
    if check is not None and check():
        raise RequestCancelled("Queued preparation request was cancelled or changed")


def caller_interrupt_enabled():
    return _caller_interrupt.get()
