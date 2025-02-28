from __future__ import annotations

from contextlib import suppress
from typing import Any, Optional

from .altair_chart_item import AltairChartItem
from .item import Item, ItemTypeError
from .jsonable_item import JSONableItem
from .matplotlib_figure_item import MatplotlibFigureItem
from .pickle_item import PickleItem


def object_to_item(object: Any, /, *, note: Optional[str] = None) -> Item:
    if not isinstance(note, (type(None), str)):
        raise TypeError(f"`note` must be a string (found '{type(note)}')")

    for cls in (
        AltairChartItem,
        MatplotlibFigureItem,
        JSONableItem,
    ):
        with suppress(ImportError, ItemTypeError):
            # ImportError:
            #     The factories are responsible to import third-party libraries in a
            #     lazy way. If library is missing, an ImportError exception will
            #     automatically be thrown.
            # ItemTypeError:
            #     The factories are responsible for checking that parameters are of
            #     the correct type. If not, they throw a ItemTypeError exception.
            return cls.factory(object, note=note)
    return PickleItem.factory(object, note=note)


__all__ = [
    "AltairChartItem",
    "Item",
    "JSONableItem",
    "MatplotlibFigureItem",
    "PickleItem",
    "object_to_item",
]
