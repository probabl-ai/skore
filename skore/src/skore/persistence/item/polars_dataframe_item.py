"""PolarsDataFrameItem.

This module defines the PolarsDataFrameItem class,
which represents a polars DataFrame item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsToJSONError(Exception):
    """Something happened while converting a polars DataFrame to JSON."""


class PolarsDataFrameItem(Item):
    """
    A class to represent a polars DataFrame item.

    This class encapsulates a polars DataFrame along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        dataframe_json: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a PolarsDataFrameItem.

        Parameters
        ----------
        dataframe_json : str
            The JSON representation of the dataframe, without its index.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.dataframe_json = dataframe_json

    @property
    def dataframe(self) -> polars.DataFrame:
        """
        The polars DataFrame from the persistence.

        Its content can differ from the original dataframe because it has been
        serialized using polars' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.dataframe_json) as df_stream:
            dataframe = polars.read_json(df_stream)
            return dataframe

    @classmethod
    def factory(cls, dataframe: polars.DataFrame, /, **kwargs) -> PolarsDataFrameItem:
        """
        Create a new PolarsDataFrameItem instance from a polars DataFrame.

        Parameters
        ----------
        dataframe : polars.DataFrame
            The polars DataFrame to store.

        Returns
        -------
        PolarsDataFrameItem
            A new PolarsDataFrameItem instance.

        Notes
        -----
        The dataframe must be JSON serializable.
        """
        import polars

        if not isinstance(dataframe, polars.DataFrame):
            raise ItemTypeError(f"Type '{dataframe.__class__}' is not supported.")

        try:
            dataframe_json = dataframe.write_json()
        except Exception as e:
            raise PolarsToJSONError("Conversion to JSON failed") from e

        return cls(dataframe_json=dataframe_json, **kwargs)
