from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from .item import Item, ItemTypeError, Representation

if TYPE_CHECKING:
    import pandas


class PandasSeriesItem(Item):
    ORIENT: Literal["split"] = "split"

    def __init__(self, index_json_str: str, series_json_str: str):
        self.index_json_str = index_json_str
        self.series_json_str = series_json_str

    @cached_property
    def __raw__(self) -> pandas.Series:
        """
        The pandas Series from the persistence.

        Its content can differ from the original series because it has been serialized
        using pandas' `to_json` function and not pickled, in order to be
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
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/json",
            value=self.__raw__.fillna("NaN").to_list(),
        )

    @classmethod
    def factory(cls, series: pandas.Series, /, **kwargs) -> PandasSeriesItem:
        """
        Create a new PandasSeriesItem instance from a pandas Series.

        Parameters
        ----------
        series : pd.Series
            The pandas Series to store.

        Returns
        -------
        PandasSeriesItem
            A new PandasSeriesItem instance.
        """
        import pandas

        if not isinstance(series, pandas.Series):
            raise ItemTypeError(f"Type '{series.__class__}' is not supported.")

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

        index = series.index.to_frame(index=False)
        series_without_index = series.reset_index(drop=True)

        instance = cls(
            index.to_json(orient=PandasSeriesItem.ORIENT),
            series_without_index.to_json(orient=PandasSeriesItem.ORIENT),
            **kwargs,
        )
        instance.__raw__ = series

        return instance
