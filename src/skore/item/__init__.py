"""Item types for the skore package."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from skore.item.item import Item
from skore.item.item_repository import ItemRepository
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem


def object_to_item(object: Any) -> Item:
    """Transform an object into an Item."""
    for cls in (
        PrimitiveItem,
        PandasDataFrameItem,
        NumpyArrayItem,
        SklearnBaseEstimatorItem,
        MediaItem,
    ):
        with suppress(ImportError, TypeError):
            return cls.factory(object)

    raise NotImplementedError(f"Type '{object.__class__}' is not supported.")


__all__ = [
    "Item",
    "ItemRepository",
    "MediaItem",
    "NumpyArrayItem",
    "PandasDataFrameItem",
    "PrimitiveItem",
    "SklearnBaseEstimatorItem",
    "object_to_item",
]
