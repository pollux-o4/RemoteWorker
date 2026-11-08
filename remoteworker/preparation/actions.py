"""One-shot next conditioning and interruptible dummy VAE preparation."""

import sys
import time

from ..core.cancellation import (
    RequestCancelled,
    caller_interrupt_enabled,
    cancellation_scope,
    check_cancelled,
)
from .capabilities import registered
from .gates import gate
from .invocation import call_values, speculative_call
from .plan import graph_contract


def prefetch(
    owner,
    current,
    following,
    graph,
    node_id,
    node,
    values,
    token,
    *,
    allow_after_sampling=False,
):
    from .._native import prep_prefetch

    return prep_prefetch(
        sys.modules[__name__],
        owner,
        current,
        following,
        graph,
        node_id,
        node,
        values,
        token,
        allow_after_sampling=allow_after_sampling,
    )


def prepare_vae(owner, current, decoder):
    from .._native import prep_prepare_vae

    return prep_prepare_vae(sys.modules[__name__], owner, current, decoder)


# Native callback namespace and public compatibility exports.
__all__ = [
    "RequestCancelled",
    "call_values",
    "caller_interrupt_enabled",
    "cancellation_scope",
    "check_cancelled",
    "gate",
    "graph_contract",
    "registered",
    "speculative_call",
    "time",
]
