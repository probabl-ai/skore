from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from .item import Item, ItemTypeError, Representation

if TYPE_CHECKING:
    import pandas


class PandasDataFrameItem(Item):
    ORIENT: Literal["split"] = "split"

    def __init__(self, index_json_str: str, dataframe_json_str: str):
        self.index_json_str = index_json_str
        self.dataframe_json_str = dataframe_json_str

    @cached_property
    def __raw__(self) -> pandas.DataFrame:
        """
        The pandas DataFrame from the persistence.

        Its content can differ from the original dataframe because it has been
        serialized using pandas' `to_json` function and not pickled, in order to be
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
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/vnd.dataframe",
            value=self.__raw__.fillna("NaN").to_dict(orient="tight"),
        )

    @classmethod
    def factory(cls, dataframe: pandas.DataFrame, /) -> PandasDataFrameItem:
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
        dataframe_without_index = dataframe.reset_index(drop=True)

        instance = cls(
            index.to_json(orient=PandasDataFrameItem.ORIENT),
            dataframe_without_index.to_json(orient=PandasDataFrameItem.ORIENT),
        )
        instance.__raw__ = dataframe

        return instance
