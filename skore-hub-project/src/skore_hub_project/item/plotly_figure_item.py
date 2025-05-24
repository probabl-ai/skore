"""
PlotlyFigureItem.

This module defines the ``PlotlyFigureItem`` class used to serialize instances of
``plotly`` figures, using the ``JSON`` format.
"""

from __future__ import annotations

from functools import cached_property
from json import loads
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    import plotly.basedatatypes


class PlotlyFigureItem(Item):
    """Serialize instances of ``plotly`` figures, using the ``JSON`` format."""

    def __init__(self, figure_json_str: str):
        """
        Initialize a ``PlotlyFigureItem``.

        Parameters
        ----------
        figure_json_str : str
            The ``plotly`` figure serialized in a str in the ``JSON`` format.
        """
        self.figure_json_str = figure_json_str

    @cached_property
    def __raw__(self) -> plotly.basedatatypes.BaseFigure:
        """Get the value from the ``PlotlyFigureItem`` instance."""
        import plotly.io

        return plotly.io.from_json(self.figure_json_str)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PlotlyFigureItem`` instance."""
        return {
            "representation": {
                "media_type": "application/vnd.plotly.v1+json",
                "value": loads(self.figure_json_str),
            }
        }

    @classmethod
    def factory(cls, value: plotly.basedatatypes.BaseFigure, /) -> PlotlyFigureItem:
        """
        Create a new ``PlotlyFigureItem`` from an instance of ``plotly`` figure.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value: ``plotly`` figure.
            The value to serialize.

        Returns
        -------
        PlotlyFigureItem
            A new ``PlotlyFigureItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``plotly`` figure.
        """
        if not lazy_is_instance(value, "plotly.basedatatypes.BaseFigure"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        import plotly.io

        instance = cls(plotly.io.to_json(value, engine="json"))
        instance.__raw__ = value

        return instance
