from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["TableReportDisplay"]

__lazy__ = {
    "TableReportDisplay": "skore._displays.data.table_report",
}

if TYPE_CHECKING:
    from skore._displays.data.table_report import TableReportDisplay


def __getattr__(name: str) -> Any:
    if module_name := __lazy__.get(name):
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
