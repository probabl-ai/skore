"""Item types for the skore package."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from skore.item.cross_validation_item import CrossValidationItem
from skore.item.item import Item
from skore.item.item_repository import ItemRepository
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.pandas_series_item import PandasSeriesItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem


def object_to_item(object: Any) -> Item:
    """Transform an object into an Item."""
    for cls in (
        PrimitiveItem,
        PandasDataFrameItem,
        PandasSeriesItem,
        NumpyArrayItem,
        SklearnBaseEstimatorItem,
        CrossValidationItem,
        MediaItem,
    ):
        with suppress(ImportError, TypeError):
            # ImportError:
            #     The factories are responsible to import third-party libraries in a
            #     lazy way. If library is missing, an ImportError exception will
            #     automatically be thrown.
            # TypeError:
            #     The factories are responsible for checking that parameters are of the
            #     correct type. If not, they throw a TypeError exception.
            return cls.factory(object)

    raise NotImplementedError(f"Type '{object.__class__}' is not supported.")


__all__ = [
    "Item",
    "ItemRepository",
    "MediaItem",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PandasSeriesItem",
    "PrimitiveItem",
    "SklearnBaseEstimatorItem",
    "object_to_item",
]
