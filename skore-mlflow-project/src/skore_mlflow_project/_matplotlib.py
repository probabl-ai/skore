"""Utilities to control matplotlib behavior in non-interactive contexts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def switch_mpl_backend(backend: str = "agg") -> Iterator[None]:
    """
    Temporarily switch to a matplotlib backend.

    Notes
    -----
    The ``agg`` backend is non-interactive and avoids opening UI windows while
    generating plots for logged artifacts.
    """
    import importlib
    import sys

    import matplotlib
    import matplotlib.pyplot

    original_backend = matplotlib.get_backend()
    original_pyplot_module = sys.modules.pop("matplotlib.pyplot")

    try:
        matplotlib.use(backend)
        importlib.import_module("matplotlib.pyplot")
        yield
    finally:
        sys.modules.pop("matplotlib.pyplot")
        matplotlib.use(original_backend)
        sys.modules["matplotlib.pyplot"] = original_pyplot_module
