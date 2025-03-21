from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsSeriesItem(Item):
    def __init__(self, series_json_str: str):
        self.series_json_str = series_json_str

    @cached_property
    def __raw__(self) -> polars.Series:
        """
        The polars Series from the persistence.

        Its content can differ from the original series because it has been serialized
        using polars' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.series_json_str) as series_stream:
            return polars.read_json(series_stream).to_series(0)

    @property
    def __representation__(self) -> dict:
        return {
            "representation": {
                "media_type": "application/json",
                "value": self.__raw__.to_list(),
            }
        }

    @classmethod
    def factory(cls, series: polars.Series, /) -> PolarsSeriesItem:
        import polars

        if not isinstance(series, polars.Series):
            raise ItemTypeError(f"Type '{series.__class__}' is not supported.")

        instance = cls(series.to_frame().write_json())
        instance.__raw__ = series

        return instance
