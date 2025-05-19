"""
AltairChartItem.

This module defines the ``AltairChartItem`` class used to serialize instances of
``altair`` charts, using the ``JSON`` format.
"""

from __future__ import annotations

from json import loads
from typing import TYPE_CHECKING

from .item import Item, ItemTypeError, lazy_is_instance

if TYPE_CHECKING:
    from altair.vegalite.v5.schema.core import TopLevelSpec as AltairChart


class AltairChartItem(Item):
    """Serialize instances of ``altair`` charts, using the ``JSON`` format."""

    def __init__(self, chart_json_str: str):
        """
        Initialize a ``AltairChartItem``.

        Parameters
        ----------
        chart_json_str : str
            The ``altair`` chart serialized in a str in the ``JSON`` format.
        """
        self.chart_json_str = chart_json_str

    @property
    def __raw__(self) -> AltairChart:
        """Get the value from the ``AltairChartItem`` instance."""
        import altair

        return altair.Chart.from_json(self.chart_json_str)

    @property
    def __representation__(self) -> dict:
        """Get the representation of the ``AltairChartItem`` instance."""
        return {
            "representation": {
                "media_type": "application/vnd.vega.v5+json",
                "value": loads(self.chart_json_str),
            }
        }

    @classmethod
    def factory(cls, value: AltairChart, /) -> AltairChartItem:
        """
        Create a new ``AltairChartItem`` from an instance of ``altair`` chart.

        It uses the ``JSON`` format.

        Parameters
        ----------
        value: ``altair`` chart.
            The value to serialize.

        Returns
        -------
        AltairChartItem
            A new ``AltairChartItem`` instance.

        Raises
        ------
        ItemTypeError
            If ``value`` is not an instance of ``altair`` chart.
        """
        if not lazy_is_instance(value, "altair.vegalite.v5.schema.core.TopLevelSpec"):
            raise ItemTypeError(f"Type '{value.__class__}' is not supported.")

        return cls(value.to_json())
