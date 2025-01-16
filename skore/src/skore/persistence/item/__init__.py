"""Item types for the skore package."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal, Optional

from . import skrub_table_report_item as SkrubTableReportItem
from .cross_validation_reporter_item import CrossValidationReporterItem
from .item import Item, ItemTypeError
from .media_item import MediaItem, MediaType
from .numpy_array_item import NumpyArrayItem
from .pandas_dataframe_item import PandasDataFrameItem
from .pandas_series_item import PandasSeriesItem
from .pickle_item import PickleItem
from .pillow_image_item import PillowImageItem
from .polars_dataframe_item import PolarsDataFrameItem
from .polars_series_item import PolarsSeriesItem
from .primitive_item import PrimitiveItem
from .sklearn_base_estimator_item import SklearnBaseEstimatorItem


def object_to_item(
    object: Any,
    /,
    *,
    note: Optional[str] = None,
    display_as: Optional[Literal["HTML", "MARKDOWN", "SVG"]] = None,
) -> Item:
    """Transform an object into an Item."""
    if display_as is not None:
        if not isinstance(object, str):
            raise TypeError("`object` must be a str if `display_as` is specified")

        if display_as not in MediaType.__members__:
            raise ValueError(f"`display_as` must be in {list(MediaType.__members__)}")

        item = MediaItem.factory_str(
            media=object,
            media_type=MediaType[display_as].value,
        )
    else:
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
            CrossValidationReporterItem,
            PillowImageItem,
        ):
            with suppress(ImportError, ItemTypeError):
                # ImportError:
                #     The factories are responsible to import third-party libraries in a
                #     lazy way. If library is missing, an ImportError exception will
                #     automatically be thrown.
                # ItemTypeError:
                #     The factories are responsible for checking that parameters are of
                #     the correct type. If not, they throw a ItemTypeError exception.
                item = cls.factory(object)
                break
        else:
            item = PickleItem.factory(object)

    if not isinstance(note, (type(None), str)):
        raise TypeError(f"`note` must be a string (found '{type(note)}')")

    # Since the item classes are now private, and to avoid having to pass the `note`
    # parameter in the factories of each item class, we define the content of the
    # `note` attribute dynamically.
    item.note = note

    return item


def item_to_object(item: Item) -> Any:
    """Transform an Item into its original object."""
    if isinstance(item, PrimitiveItem):
        return item.primitive
    elif isinstance(item, NumpyArrayItem):
        return item.array
    elif isinstance(item, (PandasDataFrameItem, PolarsDataFrameItem)):
        return item.dataframe
    elif isinstance(item, (PandasSeriesItem, PolarsSeriesItem)):
        return item.series
    elif isinstance(item, SklearnBaseEstimatorItem):
        return item.estimator
    elif isinstance(item, CrossValidationReporterItem):
        return item.reporter
    elif isinstance(item, MediaItem):
        return item.media_bytes
    elif isinstance(item, PillowImageItem):
        return item.image
    elif isinstance(item, PickleItem):
        return item.object
    else:
        raise ValueError(f"Item {item} is not a known item type.")


__all__ = [
    "CrossValidationReporterItem",
    "Item",
    "MediaItem",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PandasSeriesItem",
    "PickleItem",
    "PillowImageItem",
    "PolarsDataFrameItem",
    "PolarsSeriesItem",
    "PrimitiveItem",
    "SklearnBaseEstimatorItem",
    "SkrubTableReportItem",
    "item_to_object",
    "object_to_item",
]
