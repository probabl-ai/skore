"""Get a human-readable name for an arbitrary callable."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

import joblib


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


def _callable_hash(func: Callable) -> str:
    """Fingerprint a callable based on its source code."""
    if isinstance(func, functools.partial):
        return _callable_hash(func.func)

    if hasattr(func, "__call__") and inspect.ismethod(func.__call__):  # noqa: B004
        # func is an object with a __call__ method
        return _callable_hash(func.__call__)

    return joblib.hash(inspect.getsource(func))
