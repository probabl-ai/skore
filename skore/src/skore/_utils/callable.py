"""Get a human-readable name for an arbitrary callable."""

from __future__ import annotations

import functools
from collections.abc import Callable


def _callable_name(func: Callable) -> str:
    """Obtain the name of ``func``, with a fallback for callables without ``__name__``.

    ``functools.partial`` objects, for instance, do not have a ``__name__``
    attribute.
    """
    if hasattr(func, "__name__"):
        return func.__name__

    if isinstance(func, functools.partial):
        return _callable_name(func.func)

    return type(func).__name__
