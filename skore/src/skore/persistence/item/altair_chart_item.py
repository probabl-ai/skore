"""AltairChartItem.

This module defines the AltairChartItem class, used to persist Altair Charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skore.persistence.item.item import Item, ItemTypeError
from skore.persistence.item.media_item import lazy_is_instance

if TYPE_CHECKING:
    from altair.vegalite.v5.schema.core import TopLevelSpec as AltairChart


class AltairChartItem(Item):
    """A class used to persist a Altair chart."""

    def __init__(
        self,
        chart_str: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        """
        Initialize a AltairChartItem.

        Parameters
        ----------
        chart_str : bytes
            The JSON str of the Altair chart.
        created_at : str, optional
            The creation timestamp in ISO format.
        updated_at : str, optional
            The last update timestamp in ISO format.
        note : str, optional
            A note.
        """
        super().__init__(created_at, updated_at, note)

        self.chart_str = chart_str

    @classmethod
    def factory(cls, chart: AltairChart, /, **kwargs) -> AltairChartItem:
        """
        Create a new AltairChartItem instance from a Altair chart.

        Parameters
        ----------
        chart : altair.vegalite.v5.schema.core.TopLevelSpec
            The Altair chart to store.

        Returns
        -------
        AltairChartItem
            A new AltairChartItem instance.
        """
        if not lazy_is_instance(chart, "altair.vegalite.v5.schema.core.TopLevelSpec"):
            raise ItemTypeError(f"Type '{chart.__class__}' is not supported.")

        return cls(chart.to_json(), **kwargs)

    @property
    def chart(self) -> AltairChart:
        """The chart from the persistence."""
        import altair

        return altair.Chart.from_json(self.chart_str)
