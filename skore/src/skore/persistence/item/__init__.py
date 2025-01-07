"""Item types for the skore package."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from skore.item import skrub_table_report_item as SkrubTableReportItem
from skore.item.cross_validation_item import CrossValidationItem
from skore.item.item import Item, ItemTypeError
from skore.item.item_repository import ItemRepository
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.pandas_series_item import PandasSeriesItem
from skore.item.polars_dataframe_item import PolarsDataFrameItem
from skore.item.polars_series_item import PolarsSeriesItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem


def object_to_item(object: Any) -> Item:
    """Transform an object into an Item."""
    for cls in (
        PrimitiveItem,
        PandasDataFrameItem,
        PandasSeriesItem,
        PolarsDataFrameItem,
        PolarsSeriesItem,
        NumpyArrayItem,
        SklearnBaseEstimatorItem,
        MediaItem,
        SkrubTableReportItem,
        CrossValidationItem,
    ):
        with suppress(ImportError, ItemTypeError):
            # ImportError:
            #     The factories are responsible to import third-party libraries in a
            #     lazy way. If library is missing, an ImportError exception will
            #     automatically be thrown.
            # ItemTypeError:
            #     The factories are responsible for checking that parameters are of the
            #     correct type. If not, they throw a ItemTypeError exception.
            return cls.factory(object)

    return PickleItem(object)


def item_to_object(item: Item) -> Any:
    if isinstance(item, PrimitiveItem):
        return item.primitive
    elif isinstance(item, NumpyArrayItem):
        return item.array
    elif isinstance(item, PandasDataFrameItem):
        return item.dataframe
    elif isinstance(item, PandasSeriesItem):
        return item.series
    elif isinstance(item, PolarsDataFrameItem):
        return item.dataframe
    elif isinstance(item, PolarsSeriesItem):
        return item.series
    elif isinstance(item, SklearnBaseEstimatorItem):
        return item.estimator
    elif isinstance(item, CrossValidationItem):
        return item.cv_results_serialized
    elif isinstance(item, MediaItem):
        return item.media_bytes
    elif isinstance(item, PickleItem):
        return repr(item.pickle_bytes)
    else:
        raise ValueError(f"Item {item} is not a known item type.")


__all__ = [
    "CrossValidationItem",
    "Item",
    "ItemRepository",
    "MediaItem",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PandasSeriesItem",
    "PolarsDataFrameItem",
    "PolarsSeriesItem",
    "PrimitiveItem",
    "SklearnBaseEstimatorItem",
    "SkrubTableReportItem",
    "object_to_item",
]
