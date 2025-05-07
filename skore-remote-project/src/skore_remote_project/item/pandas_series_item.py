"""
PandasSeriesItem.

This module defines the ``PandasSeriesItem`` class used to serialize instances of
``pandas.Series``, using the ``JSON`` format.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    import pandas


class PandasSeriesItem(Item):
    """Serialize instances of ``pandas.Series``, using the ``JSON`` format."""

    ORIENT: Literal["split"] = "split"

    def __init__(self, index_json_str: str, series_json_str: str):
        """
        Initialize a ``PandasSeriesItem``.

        Parameters
        ----------
        index_json_str : str
            The index of the ``pandas.Series`` serialized in a str in the ``JSON``
            format.
        series_json_str : str
            The ``pandas.Series`` serialized in a str in the ``JSON`` format, without
            its index.
        """
        self.index_json_str = index_json_str
        self.series_json_str = series_json_str

    @cached_property
    def __raw__(self) -> pandas.Series:
        """
        Get the value from the ``PandasSeriesItem``.

        Notes
        -----
        Its content can slightly differ from the original because it has been serialized
        using ``pandas.to_json`` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import pandas

        with (
            io.StringIO(self.index_json_str) as index_stream,
            io.StringIO(self.series_json_str) as series_stream,
        ):
            index = pandas.read_json(index_stream, orient=self.ORIENT, dtype=False)
            index = index.set_index(list(index.columns))
            series = pandas.read_json(
                series_stream,
                typ="series",
                orient=self.ORIENT,
                dtype=False,
            )
            series.index = index.index

            return series

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PandasSeriesItem`` instance."""
        return {
            "representation": {
                "media_type": "application/json",
                "value": self.__raw__.fillna("NaN").to_list(),
            }
        }

    @classmethod
    def factory(cls, value: pandas.Series, /) -> PandasSeriesItem:
        """
        Create a new ``PandasSeriesItem`` from an instance of ``pandas.Series``.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value : ``pandas.Series``
            The value to serialize.

        Returns
        -------
        PandasSeriesItem
            A new ``PandasSeriesItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``pandas.Series``.

        Notes
        -----
        The series must be JSON serializable.
        """
        if not lazy_is_instance(value, "pandas.core.series.Series"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        # One native method is available to serialize series with multi-index,
        # while keeping the index names:
        #
        # Using table orientation with JSON serializer:
        #    ```python
        #    json = series.to_json(orient="table")
        #    series = pandas.read_json(json, typ="series", orient="table", dtype=False)
        #    ```
        #
        #    This method fails when an index name is an integer.
        #
        # None of those methods being compatible, we store indexes separately.

        index = value.index.to_frame(index=False)
        series_without_index = value.reset_index(drop=True)

        instance = cls(
            index.to_json(orient=PandasSeriesItem.ORIENT),
            series_without_index.to_json(orient=PandasSeriesItem.ORIENT),
        )
        instance.__raw__ = value

        return instance
