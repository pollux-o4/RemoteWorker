"""Node-owned preparation contracts, discovered from registered node classes."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal


@dataclass(frozen=True)
class Preparation:
    role: Literal["encoder", "decoder"]
    kind: str | None = None
    executor: type | None = None
    inputs: Callable[[dict], dict] = dict

    def __post_init__(self):
        if self.role not in {"encoder", "decoder"}:
            raise ValueError("Unknown preparation role")
        if self.role == "decoder" and self.kind not in {"image", "audio"}:
            raise ValueError("Decoder preparation requires a media kind")


def discover(classes):
    return {
        name: (cls, contract)
        for name, cls in classes.items()
        if isinstance(contract := getattr(cls, "REMOTE_PREPARATION", None), Preparation)
    }


@lru_cache(maxsize=1)
def registered():
    from ..registry import NODE_CLASS_MAPPINGS

    return discover(NODE_CLASS_MAPPINGS)


def encoder_types():
    return {name for name, (_, spec) in registered().items() if spec.role == "encoder"}


def decoder_types():
    return {name: spec.kind for name, (_, spec) in registered().items() if spec.role == "decoder"}
