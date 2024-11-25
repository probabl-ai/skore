"""Define SkrubTableReportItem.

SkrubTableReportItems represents a skrub TableReport item
with creation and update timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    from skrub import TableReport


class SkrubTableReportItem(Item):
    """
    A class to represent a skrub TableReport.

    This class encapsulates a skrub TableReport
    along with its creation and update timestamps.
    """

    def __init__(
        self,
        html_snippet: str,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a SkrubTableReportItem.

        Parameters
        ----------
        html_snippet: str
            The table report as an HTML snippet to store.
        created_at : str, optional
            The creation timestamp as ISO format.
        updated_at : str, optional
            The last update timestamp as ISO format.
        """
        super().__init__(created_at, updated_at)

        self.html_snippet = html_snippet

    @classmethod
    def factory(cls, table_report: TableReport) -> SkrubTableReportItem:  # type: ignore[override]
        """
        Create a new SkrubTableReportItem with the current timestamp.

        Parameters
        ----------
        primitive : Primitive
            The primitive value to store.

        Returns
        -------
        SkrubTableReportItem
            A new SkrubTableReportItem instance.
        """
        if getattr(table_report, "html_snippet", None) is None:
            raise ItemTypeError(f"Type '{table_report.__class__}' is not supported.")

        html = table_report.html_snippet()
        return cls(html_snippet=html)
