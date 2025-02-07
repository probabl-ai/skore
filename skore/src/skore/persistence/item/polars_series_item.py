"""PolarsSeriesItem.

This module defines the PolarsSeriesItem class,
which represents a polars Series item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from .item import Item, ItemTypeError

if TYPE_CHECKING:
    import polars


class PolarsSeriesItem(Item):
    """
    A class to represent a polars Series item.

    This class encapsulates a polars Series along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        series_json: str,
        created_at: Union[str, None] = None,
        updated_at: Union[str, None] = None,
        note: Union[str, None] = None,
    ):
        """
        Initialize a PolarsSeriesItem.

        Parameters
        ----------
        series_json : str
            The JSON representation of the series, without its index.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        note : Union[str, None]
            An optional note.
        """
        super().__init__(created_at, updated_at, note)

        self.series_json = series_json

    @property
    def series(self) -> polars.Series:
        """
        The polars Series from the persistence.

        Its content can differ from the original series because it has been serialized
        using polars' `to_json` function and not pickled, in order to be
        environment-independent.
        """
        import io

        import polars

        with io.StringIO(self.series_json) as series_stream:
            series = polars.read_json(series_stream).to_series(0)

            return series

    @classmethod
    def factory(cls, series: polars.Series, /, **kwargs) -> PolarsSeriesItem:
        """
        Create a new PolarsSeriesItem instance from a polars Series.

        Parameters
        ----------
        series : polars.Series
            The polars Series to store.

        Returns
        -------
        PolarsSeriesItem
            A new PolarsSeriesItem instance.
        """
        import polars

        if not isinstance(series, polars.Series):
            raise ItemTypeError(f"Type '{series.__class__}' is not supported.")

        return cls(series_json=series.to_frame().write_json(), **kwargs)
