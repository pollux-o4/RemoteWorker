"""Host measurement and GC adapters for native pressure relief."""

import gc
import sys
import time

from .._native import PressureRelief as _PressureRelief
from .._native import memory_snapshot as snapshot


def PressureRelief(clear):
    return _PressureRelief(clear, sys.modules[__name__])


# Native callback namespace and public compatibility exports.
__all__ = ["gc", "snapshot", "time"]
