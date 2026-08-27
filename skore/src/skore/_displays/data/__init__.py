from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._displays.data.table_report import TableReportDisplay

__all__ = ["TableReportDisplay"]

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={"table_report": ["TableReportDisplay"]},
)
