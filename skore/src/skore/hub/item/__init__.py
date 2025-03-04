from __future__ import annotations

from contextlib import suppress
from typing import Any

from .altair_chart_item import AltairChartItem
from .item import Item, ItemTypeError
from .jsonable_item import JSONableItem
from .matplotlib_figure_item import MatplotlibFigureItem
from .pickle_item import PickleItem


def object_to_item(object: Any, /) -> Item:
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
            return cls.factory(object)
    return PickleItem.factory(object)


__all__ = [
    "AltairChartItem",
    "Item",
    "JSONableItem",
    "MatplotlibFigureItem",
    "PickleItem",
    "object_to_item",
]
