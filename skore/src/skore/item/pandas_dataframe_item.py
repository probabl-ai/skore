"""PandasDataFrameItem.

This module defines the PandasDataFrameItem class,
which represents a pandas DataFrame item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

from skore.item.item import Item


class PandasDataFrameItem(Item):
    """
    A class to represent a pandas DataFrame item.

    This class encapsulates a pandas DataFrame along with its
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
        Initialize a PandasDataFrameItem.

        Parameters
        ----------
        index_json : json
            The JSON representation of the dataframe's index.
        dataframe_json : json
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
    def dataframe(self) -> pandas.DataFrame:
        """The pandas DataFrame."""
        import io

        import pandas

        with (
            io.StringIO(self.index_json) as index_stream,
            io.StringIO(self.dataframe_json) as df_stream,
        ):
            index = pandas.read_json(index_stream, orient=self.ORIENT, dtype=False)
            index = index.set_index(list(index.columns))
            dataframe = pandas.read_json(df_stream, orient=self.ORIENT, dtype=False)
            dataframe.index = index.index

            return dataframe

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame) -> PandasDataFrameItem:
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
            raise TypeError(f"Type '{dataframe.__class__}' is not supported.")

        index = dataframe.index.to_frame(index=False)
        dataframe = dataframe.reset_index(drop=True)
        instance = cls(
            index_json=index.to_json(orient=PandasDataFrameItem.ORIENT),
            dataframe_json=dataframe.to_json(orient=PandasDataFrameItem.ORIENT),
        )

        return instance
