from __future__ import annotations

from typing import TYPE_CHECKING

from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem

if TYPE_CHECKING:
    from typing import Union

    Metadata = dict[str, str]
    Item = Union[
        PrimitiveItem,
        PandasDataFrameItem,
        NumpyArrayItem,
        SklearnBaseEstimatorItem,
        MediaItem,
    ]
