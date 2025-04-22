"""
Module definition of the ``Item`` classes.

``Item`` is an internal concept that is used as a DTO (Data Transfer Object) to exchange
python objects between ``skore`` and ``skore hub``.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from .altair_chart_item import AltairChartItem
from .item import Item, ItemTypeError
from .jsonable_item import JSONableItem
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
from .sklearn_base_estimator_item import SklearnBaseEstimatorItem
from .skore_estimator_report_item import SkoreEstimatorReportItem
from .skrub_table_report_item import SkrubTableReportItem


def object_to_item(object: Any, /) -> Item:
    """Serialize any python object into an ``Item``."""
    for cls in (
        # Lexicographically sorted, the order of execution doesn't matter
        AltairChartItem,
        MatplotlibFigureItem,
        NumpyArrayItem,
        PandasDataFrameItem,
        PandasSeriesItem,
        PillowImageItem,
        PlotlyFigureItem,
        PolarsDataFrameItem,
        PolarsSeriesItem,
        SklearnBaseEstimatorItem,
        SkoreEstimatorReportItem,
        SkrubTableReportItem,
        # JSONable must be the penultimate
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
    "MediaItem",
    "MediaType",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PandasSeriesItem",
    "PickleItem",
    "PillowImageItem",
    "PlotlyFigureItem",
    "PolarsDataFrameItem",
    "PolarsSeriesItem",
    "SklearnBaseEstimatorItem",
    "SkoreEstimatorReportItem",
    "SkrubTableReportItem",
    "object_to_item",
]
