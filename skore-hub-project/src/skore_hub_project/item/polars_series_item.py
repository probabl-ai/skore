"""
PolarsSeriesItem.

This module defines the ``PolarsSeriesItem`` class used to serialize instances of
``polars.Series``, using the ``JSON`` format.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    import polars


class PolarsSeriesItem(Item):
    """Serialize instances of ``polars.Series``, using the ``JSON`` format."""

    def __init__(self, series_json_str: str):
        """
        Initialize a ``PolarsSeriesItem``.

        Parameters
        ----------
        series_json_str : str
            The ``polars.Series`` serialized in a str in the ``JSON`` format.
        """
        self.series_json_str = series_json_str

    @cached_property
    def __raw__(self) -> polars.Series:
        """
        Get the value from the ``PolarsSeriesItem``.

        Notes
        -----
        Its content can slightly differ from the original because it has been serialized
        using ``polars.to_json`` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.series_json_str) as series_stream:
            return polars.read_json(series_stream).to_series(0)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PolarsSeriesItem`` instance."""
        return {
            "representation": {
                "media_type": "application/json",
                "value": self.__raw__.to_list(),
            }
        }

    @classmethod
    def factory(cls, value: polars.Series, /) -> PolarsSeriesItem:
        """
        Create a new ``PolarsSeriesItem`` from an instance of ``polars.Series``.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value : ``polars.Series``
            The value to serialize.

        Returns
        -------
        PolarsSeriesItem
            A new ``PolarsSeriesItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``polars.Series``.

        Notes
        -----
        The series must be JSON serializable.
        """
        if not lazy_is_instance(value, "polars.series.series.Series"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        instance = cls(value.to_frame().write_json())
        instance.__raw__ = value

        return instance
