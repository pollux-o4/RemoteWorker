"""Native preparation contracts, discovered from Comfy's registered classes."""
from functools import lru_cache
from .._native import Preparation, discover_preparation as discover, preparation_encoders, preparation_decoders


@lru_cache(maxsize=1)
def registered():
    from ..registry import NODE_CLASS_MAPPINGS

    return discover(NODE_CLASS_MAPPINGS)


def encoder_types():
    return preparation_encoders(registered())


def decoder_types():
    return preparation_decoders(registered())

# Native callback namespace and public compatibility exports.
__all__ = ['Preparation']
