"""PolarsDataFrameItem.

This module defines the PolarsDataFrameItem class,
which represents a polars DataFrame item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsDataFrameItem(Item):
    """
    A class to represent a polars DataFrame item.

    This class encapsulates a polars DataFrame along with its
    creation and update timestamps.
    """

    ORIENT = "split"

    def __init__(
        self,
        index_json: str,
        dataframe_json: str,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PolarsDataFrameItem.

        Parameters
        ----------
        index_json : str
            The JSON representation of the dataframe's index.
        dataframe_json : str
            The JSON representation of the dataframe, without its index.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.index_json = index_json
        self.dataframe_json = dataframe_json

    @cached_property
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
            dataframe = polars.read_json(df_stream, dtype=False)
            return dataframe

    @classmethod
    def factory(cls, dataframe: polars.DataFrame) -> PolarsDataFrameItem:
        """
        Create a new PolarsDataFrameItem instance from a polars DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
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

        # Two native methods are available to serialize dataframe with multi-index,
        # while keeping the index names:
        #
        # 1. Using table orientation with JSON serializer:
        #    ```python
        #    json = dataframe.to_json(orient="table")
        #    dataframe = polars.read_json(json, orient="table", dtype=False)
        #    ```
        #
        #    This method fails when an index/column name is an integer.
        #
        # 2. Using record orientation with indexes as columns:
        #    ```python
        #    dataframe = dataframe.reset_index()
        #    json = dataframe.to_json(orient="records")
        #    dataframe = polars.read_json(json, orient="records", dtype=False)
        #    ```
        #
        #    This method fails when the index has the same name as one of the columns.
        #
        # None of those methods being compatible, we decide to store indexes separately.

        index = dataframe.index.to_frame(index=False)
        dataframe = dataframe.reset_index(drop=True)

        return cls(
            index_json=index.to_json(orient=PolarsDataFrameItem.ORIENT),
            dataframe_json=dataframe.write_json(),
        )
