"""
PandasDataFrameItem.

This module defines the ``PandasDataFrameItem`` class used to serialize instances of
``pandas.DataFrame``, using the ``JSON`` format.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    import pandas


class PandasDataFrameItem(Item):
    """Serialize instances of ``pandas.DataFrame``, using the ``JSON`` format."""

    ORIENT: Literal["split"] = "split"

    def __init__(self, index_json_str: str, dataframe_json_str: str):
        """
        Initialize a ``PandasDataFrameItem``.

        Parameters
        ----------
        index_json_str : str
            The index of the ``pandas.DataFrame`` serialized in a str in the ``JSON``
            format.
        dataframe_json_str : str
            The ``pandas.DataFrame`` serialized in a str in the ``JSON`` format, without
            its index.
        """
        self.index_json_str = index_json_str
        self.dataframe_json_str = dataframe_json_str

    @cached_property
    def __raw__(self) -> pandas.DataFrame:
        """
        Get the value from the ``PandasDataFrameItem``.

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
            io.StringIO(self.dataframe_json_str) as df_stream,
        ):
            index = pandas.read_json(index_stream, orient=self.ORIENT, dtype=False)
            index = index.set_index(list(index.columns))
            dataframe = pandas.read_json(df_stream, orient=self.ORIENT, dtype=False)
            dataframe.index = index.index

            return dataframe

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``PandasDataFrameItem`` instance."""
        return {
            "representation": {
                "media_type": "application/vnd.dataframe",
                "value": self.__raw__.fillna("NaN").to_dict(orient="tight"),
            }
        }

    @classmethod
    def factory(cls, value: pandas.DataFrame, /) -> PandasDataFrameItem:
        """
        Create a new ``PandasDataFrameItem`` from an instance of ``pandas.DataFrame``.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value : ``pandas.DataFrame``
            The value to serialize.

        Returns
        -------
        PandasDataFrameItem
            A new ``PandasDataFrameItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``pandas.DataFrame``.

        Notes
        -----
        The dataframe must be JSON serializable.
        """
        if not lazy_is_instance(value, "pandas.core.frame.DataFrame"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        # Two native methods are available to serialize dataframe with multi-index,
        # while keeping the index names:
        #
        # 1. Using table orientation with JSON serializer:
        #    ```python
        #    json = dataframe.to_json(orient="table")
        #    dataframe = pandas.read_json(json, orient="table", dtype=False)
        #    ```
        #
        #    This method fails when an index/column name is an integer.
        #
        # 2. Using record orientation with indexes as columns:
        #    ```python
        #    dataframe = dataframe.reset_index()
        #    json = dataframe.to_json(orient="records")
        #    dataframe = pandas.read_json(json, orient="records", dtype=False)
        #    ```
        #
        #    This method fails when the index has the same name as one of the columns.
        #
        # None of those methods being compatible, we decide to store indexes separately.

        index = value.index.to_frame(index=False)
        dataframe_without_index = value.reset_index(drop=True)

        instance = cls(
            index.to_json(orient=PandasDataFrameItem.ORIENT),
            dataframe_without_index.to_json(orient=PandasDataFrameItem.ORIENT),
        )
        instance.__raw__ = value

        return instance
