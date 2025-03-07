"""Item types for the skore package."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal, Optional

from .altair_chart_item import AltairChartItem
from .item import Item, ItemTypeError
from .matplotlib_figure_item import MatplotlibFigureItem
from .media_item import MediaItem, MediaType
from .numpy_array_item import NumpyArrayItem
from .pandas_dataframe_item import PandasDataFrameItem
from .pandas_series_item import PandasSeriesItem
from .pickle_item import PickleItem
from .pillow_image_item import PillowImageItem
from .plotly_figure_item import PlotlyFigureItem
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
    if not isinstance(note, (type(None), str)):
        raise TypeError(f"`note` must be a string (found '{type(note)}')")

    if display_as is not None:
        if not isinstance(object, str):
            raise TypeError("`object` must be a string if `display_as` is specified")

        if display_as not in MediaType.__members__:
            raise ValueError(
                f"`display_as` must be one of {', '.join(MediaType.__members__)}"
            )

        return MediaItem.factory(object, MediaType[display_as].value, note=note)

    for cls in (
        AltairChartItem,
        MatplotlibFigureItem,
        MediaItem,
        NumpyArrayItem,
        PandasDataFrameItem,
        PandasSeriesItem,
        PillowImageItem,
        PlotlyFigureItem,
        PolarsDataFrameItem,
        PolarsSeriesItem,
        PrimitiveItem,
        SklearnBaseEstimatorItem,
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
    elif isinstance(item, MediaItem):
        return item.media
    elif isinstance(item, PillowImageItem):
        return item.image
    elif isinstance(item, PlotlyFigureItem):
        return item.figure
    elif isinstance(item, AltairChartItem):
        return item.chart
    elif isinstance(item, MatplotlibFigureItem):
        return item.figure
    elif isinstance(item, PickleItem):
        return item.object
    else:
        raise ValueError(f"Item {item} is not a known item type.")


__all__ = [
    "AltairChartItem",
    "Item",
    "MatplotlibFigureItem",
    "MediaItem",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PandasSeriesItem",
    "PickleItem",
    "PillowImageItem",
    "PlotlyFigureItem",
    "PolarsDataFrameItem",
    "PolarsSeriesItem",
    "PrimitiveItem",
    "SklearnBaseEstimatorItem",
    "item_to_object",
    "object_to_item",
]
