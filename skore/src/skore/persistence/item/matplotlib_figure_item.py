"""MatplotlibFigureItem.

This module defines the MatplotlibFigureItem class, used to persist Matplotlib figures.
"""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import joblib

from skore.persistence.item.item import Item, ItemTypeError
from skore.persistence.item.media_item import lazy_is_instance
from skore.utils import b64_str_to_bytes, bytes_to_b64_str

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@contextmanager
def mpl_backend(backend="agg"):
    """Context manager for switching matplotlib backend."""
    import matplotlib

    original_backend = matplotlib.get_backend()
    matplotlib.use(backend)
    try:
        yield
    finally:
        matplotlib.use(original_backend)


class MatplotlibFigureItem(Item):
    """A class used to persist a Matplotlib figure."""

    def __init__(
        self,
        figure_b64_str: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a MatplotlibFigureItem.

        Parameters
        ----------
        figure_b64_str : str
            The raw bytes of the Matplotlib figure pickled representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.figure_b64_str = figure_b64_str

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
            joblib.dump(figure, stream)

            figure_bytes = stream.getvalue()
            figure_b64_str = bytes_to_b64_str(figure_bytes)

            return cls(figure_b64_str, **kwargs)

    @property
    def figure(self) -> Figure:
        """The figure from the persistence."""
        # switch mpl backend to avoid opening windows in a background thread
        figure_bytes = b64_str_to_bytes(self.figure_b64_str)

        with (
            BytesIO(figure_bytes) as stream,
            mpl_backend(backend="agg"),
        ):
            return joblib.load(stream)
