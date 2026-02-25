"""Utilities to control matplotlib behavior in non-interactive contexts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt


@contextmanager
def switch_mpl_backend(backend: str = "agg") -> Iterator[None]:
    """
    Temporarily switch to a matplotlib backend.

    Notes
    -----
    The ``agg`` backend is non-interactive and avoids opening UI windows while
    generating plots for logged artifacts.
    """
    original_backend = plt.get_backend()
    plt.switch_backend(backend)

    try:
        yield
    finally:
        if plt.get_backend().lower() != original_backend.lower():
            try:
                plt.switch_backend(original_backend)
            except Exception:
                # Keep a safe backend if the original one is unavailable in headless CI.
                plt.switch_backend("agg")
