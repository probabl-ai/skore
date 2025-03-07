"""SkrubTableReportItem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .item import ItemTypeError, Representation, lazy_is_instance
from .pickle_item import PickleItem

if TYPE_CHECKING:
    from skrub import TableReport


class SkrubTableReportItem(PickleItem):
    @property
    def __representation__(self) -> Representation:
        return Representation(media_type="text/html", value=self.__raw__.html_snippet())

    @classmethod
    def factory(cls, report: TableReport, /) -> SkrubTableReportItem:
        """
        Create a new SkrubTableReportItem from a skrub ``TableReport``.

        Parameters
        ----------
        report : TableReport
            The report to store.

        Returns
        -------
        SkrubTableReportItem
            A new SkrubTableReportItem instance.
        """
        if not lazy_is_instance(report, "skrub._reporting._table_report.TableReport"):
            raise ItemTypeError(f"Type '{report.__class__}' is not supported.")

        return super().factory(report)
