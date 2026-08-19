from collections.abc import Iterator
from contextlib import contextmanager

from matplotlib import pyplot as plt

__all__ = ["switch_plt_backend"]


@contextmanager
def switch_plt_backend(backend: str = "agg") -> Iterator[None]:
    """
    Context-manager for switching ``matplotlib.pyplot.backend``.

    Notes
    -----
    The ``agg`` backend is a non-interactive backend that can only write to files.
    It is used to generate artifacts where we don't need an X display.

    https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    """
    original = plt.get_backend()

    try:
        plt.switch_backend(backend)
        yield
    finally:
        plt.close("all")
        plt.switch_backend(original)
