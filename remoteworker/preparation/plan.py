"""Python registry adapter for the Rust queue planner."""

from .. import _native
from .._native import (
    fingerprint,
    graph_contract,
    head,
    is_sampling,
    job_graph,
    literal_inputs,
    reachable,
)
from .capabilities import decoder_types, encoder_types


def next_encoders(graph):
    return _native.next_encoders(graph, encoder_types())


def first_decoder(graph):
    return _native.first_decoder(graph, decoder_types())


def pending_encoders(pending, completed):
    return iter(_native.pending_encoders(pending, completed, encoder_types()))


# Native callback namespace and public compatibility exports.
__all__ = [
    "fingerprint",
    "graph_contract",
    "head",
    "is_sampling",
    "job_graph",
    "literal_inputs",
    "reachable",
]
