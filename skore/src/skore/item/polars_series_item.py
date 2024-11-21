"""PolarsSeriesItem.

This module defines the PolarsSeriesItem class,
which represents a polars Series item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsSeriesItem(Item):
    """
    A class to represent a polars Series item.

    This class encapsulates a polars Series along with its
    creation and update timestamps.
    """

    ORIENT = "split"

    def __init__(
        self,
        index_json: str,
        series_json: str,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PolarsSeriesItem.

        Parameters
        ----------
        index_json : str
            The JSON representation of the series's index.
        series_json : str
            The JSON representation of the series, without its index.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.index_json = index_json
        self.series_json = series_json

    @cached_property
    def series(self) -> polars.Series:
        """
        The polars Series from the persistence.

        Its content can differ from the original series because it has been serialized
        using polars' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with (
            io.StringIO(self.index_json) as index_stream,
            io.StringIO(self.series_json) as series_stream,
        ):
            index = polars.read_json(index_stream, orient=self.ORIENT, dtype=False)
            index = index.set_index(list(index.columns))
            series = polars.read_json(
                series_stream,
                typ="series",
                orient=self.ORIENT,
                dtype=False,
            )
            series.index = index.index

            return series

    @classmethod
    def factory(cls, series: polars.Series) -> PolarsSeriesItem:
        """
        Create a new PolarsSeriesItem instance from a polars Series.

        Parameters
        ----------
        series : pd.Series
            The polars Series to store.

        Returns
        -------
        PolarsSeriesItem
            A new PolarsSeriesItem instance.
        """
        import polars

        if not isinstance(series, polars.Series):
            raise ItemTypeError(f"Type '{series.__class__}' is not supported.")

        # One native method is available to serialize series with multi-index,
        # while keeping the index names:
        #
        # Using table orientation with JSON serializer:
        #    ```python
        #    json = series.to_json(orient="table")
        #    series = polars.read_json(json, typ="series", orient="table", dtype=False)
        #    ```
        #
        #    This method fails when an index name is an integer.
        #
        # None of those methods being compatible, we store indexes separately.

        index = series.index.to_frame(index=False)
        series = series.reset_index(drop=True)

        return cls(
            index_json=index.to_json(orient=PolarsSeriesItem.ORIENT),
            series_json=series.to_json(orient=PolarsSeriesItem.ORIENT),
        )
