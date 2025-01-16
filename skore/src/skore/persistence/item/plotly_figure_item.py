"""PlotlyFigureItem.

This module defines the PlotlyFigureItem class, used to persist Plotly figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .item import Item, ItemTypeError
from .media_item import lazy_is_instance

if TYPE_CHECKING:
    import plotly.basedatatypes as plotly


class PlotlyFigureItem(Item):
    """A class used to persist a Plotly figure."""

    def __init__(
        self,
        figure_str: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a PlotlyFigureItem.

        Parameters
        ----------
        figure_str : bytes
            The JSON str of the Plotly figure.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.figure_str = figure_str

    @classmethod
    def factory(cls, figure: plotly.BaseFigure, /, **kwargs) -> PlotlyFigureItem:
        """
        Create a new PlotlyFigureItem instance from a Plotly figure.

        Parameters
        ----------
        figure : plotly.basedatatypes.BaseFigure
            The Plotly figure to store.

        Returns
        -------
        PlotlyFigureItem
            A new PlotlyFigureItem instance.
        """
        if not lazy_is_instance(figure, "plotly.basedatatypes.BaseFigure"):
            raise ItemTypeError(f"Type '{figure.__class__}' is not supported.")

        import plotly.io

        return cls(plotly.io.to_json(figure, engine="json"), **kwargs)

    @property
    def figure(self) -> plotly.BaseFigure:
        """The figure from the persistence."""
        import plotly.io

        return plotly.io.from_json(self.figure_str)

    def as_serializable_dict(self):
        """Return item as a JSONable dict to export to frontend."""
        import base64

        figure_bytes = self.figure_str.encode("utf-8")
        figure_bytes_b64 = base64.b64encode(figure_bytes).decode()

        return super().as_serializable_dict() | {
            "media_type": "application/vnd.plotly.v1+json;base64",
            "value": figure_bytes_b64,
        }