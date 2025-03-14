from __future__ import annotations

from functools import cached_property
from json import loads
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError, Representation, lazy_is_instance

if TYPE_CHECKING:
    import plotly.basedatatypes


class PlotlyFigureItem(Item):
    def __init__(self, figure_json_str: str):
        self.figure_json_str = figure_json_str

    @cached_property
    def __raw__(self) -> plotly.basedatatypes.BaseFigure:
        import plotly.io

        return plotly.io.from_json(self.figure_json_str)

    @property
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/vnd.plotly.v1+json",
            value=loads(self.figure_json_str),
        )

    @classmethod
    def factory(cls, figure: plotly.basedatatypes.BaseFigure, /) -> PlotlyFigureItem:
        if not lazy_is_instance(figure, "plotly.basedatatypes.BaseFigure"):
            raise ItemTypeError(f"Type '{figure.__class__}' is not supported.")

        import plotly.io

        instance = cls(plotly.io.to_json(figure, engine="json"))
        instance.__raw__ = figure

        return instance
