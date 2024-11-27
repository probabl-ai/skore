"""Define SkrubTableReportItem.

SkrubTableReportItems represents a skrub TableReport item
with creation and update timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore.item.item import ItemTypeError
from skore.item.media_item import MediaItem

if TYPE_CHECKING:
    from skrub import TableReport


class SkrubTableReportItem(MediaItem):
    """
    A class to represent a skrub TableReport.

    This class encapsulates a skrub TableReport
    along with its creation and update timestamps.
    """

    @classmethod
    def factory(cls, table_report: TableReport) -> MediaItem:  # type: ignore[override]
        """
        Create a new SkrubTableReportItem with the current timestamp.

        Parameters
        ----------
        table_report : TableReport
            The report to store.

        Returns
        -------
        SkrubTableReportItem
            A new SkrubTableReportItem instance.
        """
        if getattr(table_report, "html_snippet", None) is None:
            raise ItemTypeError(f"Type '{table_report.__class__}' is not supported.")

        html = table_report.html_snippet()
        return super().factory_str(html, media_type="text/html")  # type: ignore
