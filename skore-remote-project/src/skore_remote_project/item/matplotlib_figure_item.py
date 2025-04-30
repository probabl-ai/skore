"""
MatplotlibFigureItem.

This module defines the ``MatplotlibFigureItem`` class used to serialize instances of
``matplotlib.Figure``, using binary protocols.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from .item import ItemTypeError, bytes_to_b64_str, lazy_is_instance, switch_mpl_backend
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MatplotlibFigureItem(PickleItem):
    """Serialize instances of ``matplotlib.Figure``, using binary protocols."""

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``MatplotlibFigureItem`` instance."""
        with switch_mpl_backend(), BytesIO() as stream:
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
    def factory(cls, value: Figure, /) -> MatplotlibFigureItem:
        """
        Create a new ``MatplotlibFigureItem`` from an instance of ``matplotlib.Figure``.

        It uses binary protocols.

        Parameters
        ----------
        value : ``matplotlib.Figure``
            The value to serialize.

        Returns
        -------
        MatplotlibFigureItem
            A new ``MatplotlibFigureItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``matplotlib.Figure``.
        """
        if not lazy_is_instance(value, "matplotlib.figure.Figure"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        return super().factory(value)
