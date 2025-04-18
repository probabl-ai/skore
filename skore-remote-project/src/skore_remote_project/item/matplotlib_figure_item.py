"""MatplotlibFigureItem.

This module defines the MatplotlibFigureItem class, used to persist Matplotlib figures.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from io import BytesIO
from typing import TYPE_CHECKING

from joblib import dump, load

from .item import (
    Item,
    ItemTypeError,
    b64_str_to_bytes,
    bytes_to_b64_str,
    lazy_is_instance,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@contextmanager
def mpl_backend(backend="agg"):
    """Context manager for switching matplotlib backend."""
    import matplotlib

    original_backend = matplotlib.get_backend()

    try:
        matplotlib.use(backend)
        yield
    finally:
        matplotlib.use(original_backend)


class MatplotlibFigureItem(Item):
    """A class used to persist a Matplotlib figure."""

    def __init__(self, figure_b64_str: str):
        self.figure_b64_str = figure_b64_str

    @cached_property
    def __raw__(self) -> Figure:
        figure_bytes = b64_str_to_bytes(self.figure_b64_str)

        with BytesIO(figure_bytes) as stream, mpl_backend(backend="agg"):
            return load(stream)

    @property
    def __representation__(self) -> dict:
        with BytesIO() as stream:
            self.__raw__.savefig(stream, format="svg", bbox_inches="tight")

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

        return {
            "representation": {
                "media_type": "image/svg+xml;base64",
                "value": figure_b64_str,
            }
        }

    @classmethod
    def factory(cls, figure: Figure, /, **kwargs) -> MatplotlibFigureItem:
        """
        Create a new MatplotlibFigureItem instance from a Matplotlib figure.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The Matplotlib figure to store.

        Returns
        -------
        MatplotlibFigureItem
            A new MatplotlibFigureItem instance.
        """
        if not lazy_is_instance(figure, "matplotlib.figure.Figure"):
            raise ItemTypeError(f"Type '{figure.__class__}' is not supported.")

        with BytesIO() as stream:
            dump(figure, stream)

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

        instance = cls(figure_b64_str, **kwargs)
        instance.__raw__ = figure

        return instance
