"""PandasSeriesItem.

This module defines the PandasSeriesItem class,
which represents a pandas Series item.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

from skore.item.item import Item


class PandasSeriesItem(Item):
    """
    A class to represent a pandas Series item.

    This class encapsulates a pandas Series along with its
    creation and update timestamps.
    """

    def __init__(
        self,
        series_list: list,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a PandasSeriesItem.

        Parameters
        ----------
        series_list : list
            The list representation of the series.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.series_list = series_list

    @cached_property
    def series(self) -> pandas.Series:
        """The pandas Series."""
        import pandas

        return pandas.Series(self.series_list)

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
            raise TypeError(f"Type '{series.__class__}' is not supported.")

        instance = cls(series_list=series.to_list())

        # add series as cached property
        instance.series = series

        return instance
