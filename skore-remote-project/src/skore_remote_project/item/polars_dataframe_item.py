"""
PolarsDataFrameItem.

This module defines the ``PolarsDataFrameItem`` class used to serialize instances of
``polars.DataFrame``, using the ``JSON`` format.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    import polars


class PolarsDataFrameItem(Item):
    """Serialize instances of ``polars.DataFrame``, using the ``JSON`` format."""

    def __init__(self, dataframe_json_str: str):
        """
        Initialize a ``PolarsDataFrameItem``.

        Parameters
        ----------
        dataframe_json_str : str
            The ``polars.DataFrame`` serialized in a str in the ``JSON`` format.
        """
        self.dataframe_json_str = dataframe_json_str

    @cached_property
    def __raw__(self) -> polars.DataFrame:
        """
        Get the value from the ``PolarsDataFrameItem``.

        Notes
        -----
        Its content can slightly differ from the original because it has been serialized
        using ``polars.to_json`` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.dataframe_json_str) as df_stream:
            return polars.read_json(df_stream)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PolarsDataFrameItem`` instance."""
        return {
            "representation": {
                "media_type": "application/vnd.dataframe",
                "value": self.__raw__.to_pandas().fillna("NaN").to_dict(orient="tight"),
            }
        }

    @classmethod
    def factory(cls, value: polars.DataFrame, /) -> PolarsDataFrameItem:
        """
        Create a new ``PolarsDataFrameItem`` from an instance of ``polars.DataFrame``.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value : ``polars.DataFrame``
            The value to serialize.

        Returns
        -------
        PolarsDataFrameItem
            A new ``PolarsDataFrameItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``polars.DataFrame``.

        Notes
        -----
        The dataframe must be JSON serializable.
        """
        if not lazy_is_instance(value, "polars.dataframe.frame.DataFrame"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        instance = cls(value.write_json())
        instance.__raw__ = value

        return instance
