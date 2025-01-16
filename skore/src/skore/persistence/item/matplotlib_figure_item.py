"""MatplotlibFigureItem.

This module defines the MatplotlibFigureItem class, used to persist Matplotlib figures.
"""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pickle import dumps, loads
from typing import TYPE_CHECKING, Optional

from .item import Item, ItemTypeError
from .media_item import lazy_is_instance

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MatplotlibFigureItem(Item):
    """A class used to persist a Matplotlib figure."""

    def __init__(
        self,
        figure_bytes: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a MatplotlibFigureItem.

        Parameters
        ----------
        figure_bytes : bytes
            The raw bytes of the Matplotlib figure pickled representation.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.figure_bytes = figure_bytes

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

        return cls(dumps(figure), **kwargs)

    @property
    def figure(self) -> Figure:
        """The figure from the persistence."""
        return loads(self.figure_bytes)

    def as_serializable_dict(self) -> dict:
        """Return item as a JSONable dict to export to frontend."""
        with BytesIO() as stream:
            self.figure.savefig(stream, format="svg", bbox_inches="tight")

            figure_bytes = stream.getvalue()
            figure_bytes_b64 = b64encode(figure_bytes).decode()

            return super().as_serializable_dict() | {
                "media_type": "image/svg+xml;base64",
                "value": figure_bytes_b64,
            }
