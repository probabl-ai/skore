"""PandasSeriesItem.

This module defines the PandasSeriesItem class,
which represents a pandas Series item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import pandas


class PandasSeriesItem(Item):
    """
    A class to represent a pandas Series item.

    This class encapsulates a pandas Series along with its
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
        Initialize a PandasSeriesItem.

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
    def series(self) -> pandas.Series:
        """
        The pandas Series from the persistence.

        Its content can differ from the original series because it has been serialized
        using pandas' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import pandas

        with (
            io.StringIO(self.index_json) as index_stream,
            io.StringIO(self.series_json) as series_stream,
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

    def as_serializable_dict(self):
        """Get a serializable dict from the item.

        Derived class must call their super implementation
        and merge the result with their output.
        """
        d = super().as_serializable_dict()
        d.update(
            {
                "value": self.series.fillna("NaN").to_list(),
                "media_type": "text/markdown",
            }
        )
        return d

    @classmethod
    def factory(cls, series: pandas.Series) -> PandasSeriesItem:
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
        series = series.reset_index(drop=True)

        return cls(
            index_json=index.to_json(orient=PandasSeriesItem.ORIENT),
            series_json=series.to_json(orient=PandasSeriesItem.ORIENT),
        )
