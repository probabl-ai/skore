"""Item types and metadata for the skore package.

This module defines various item types used in the skore package, as well as
the Metadata type and a union of all item types.

Types:
    Metadata: A dictionary mapping strings to strings, used for storing
        metadata about items.
    Item: A union of all item types supported by skore.
"""

from __future__ import annotations

from typing import Union

from skore.item.item_repository import ItemRepository
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem

Metadata = dict[str, str]
Item = Union[
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
]

__all__ = ["ItemRepository"]
