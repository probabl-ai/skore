"""PandasDataFrameItem.

This module defines the PandasDataFrameItem class,
which represents a pandas DataFrame item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import pandas


class PandasDataFrameItem(Item):
    """
    A class to represent a pandas DataFrame item.

    This class encapsulates a pandas DataFrame along with its
    creation and update timestamps.
    """

    ORIENT: Literal["split"] = "split"

    def __init__(
        self,
        index_json: str,
        dataframe_json: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a PandasDataFrameItem.

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
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.index_json = index_json
        self.dataframe_json = dataframe_json

    @property
    def dataframe(self) -> pandas.DataFrame:
        """
        The pandas DataFrame from the persistence.

        Its content can differ from the original dataframe because it has been
        serialized using pandas' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import pandas

        with (
            io.StringIO(self.index_json) as index_stream,
            io.StringIO(self.dataframe_json) as df_stream,
        ):
            index = pandas.read_json(
                index_stream,
                orient=self.ORIENT,
                dtype=False,  # type: ignore
            )
            index = index.set_index(list(index.columns))
            dataframe = pandas.read_json(
                df_stream,
                orient=self.ORIENT,
                dtype=False,  # type: ignore
            )
            dataframe.index = index.index

            return dataframe

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame, /, **kwargs) -> PandasDataFrameItem:
        """
        Create a new PandasDataFrameItem instance from a pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame to store.

        Returns
        -------
        PandasDataFrameItem
            A new PandasDataFrameItem instance.

        Notes
        -----
        The dataframe must be JSON serializable.
        """
        import pandas

        if not isinstance(dataframe, pandas.DataFrame):
            raise ItemTypeError(f"Type '{dataframe.__class__}' is not supported.")

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

        index = dataframe.index.to_frame(index=False)
        dataframe = dataframe.reset_index(drop=True)

        return cls(
            index_json=index.to_json(orient=PandasDataFrameItem.ORIENT),
            dataframe_json=dataframe.to_json(orient=PandasDataFrameItem.ORIENT),
            **kwargs,
        )
