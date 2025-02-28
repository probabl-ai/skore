from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .item import Item, ItemTypeError, Representation, lazy_is_instance

if TYPE_CHECKING:
    from altair.vegalite.v5.schema.core import TopLevelSpec as AltairChart


class AltairChartItem(Item):
    def __init__(
        self,
        chart_json_str: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        note: Optional[str] = None,
    ):
        super().__init__(created_at, updated_at, note)

        self.chart_json_str = chart_json_str

    @property
    def __raw__(self) -> AltairChart:
        import altair

        return altair.Chart.from_json(self.chart_json_str)

    @property
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/vnd.vega.v5+json",
            value=self.chart_json_str,
        )

    @classmethod
    def factory(cls, chart: AltairChart, /, **kwargs) -> AltairChartItem:
        if not lazy_is_instance(chart, "altair.vegalite.v5.schema.core.TopLevelSpec"):
            raise ItemTypeError(f"Type '{chart.__class__}' is not supported.")

        return cls(chart.to_json(), **kwargs)
