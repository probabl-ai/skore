"""ItemRepository for managing storage and retrieval of items.

This module provides the ItemRepository class, which is responsible for
storing, retrieving, and deleting items in a storage system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore.item.item import Item
    from skore.persistence.abstract_storage import AbstractStorage


from skore.item.cross_validation_item import CrossValidationItem
from skore.item.media_item import MediaItem
from skore.item.numpy_array_item import NumpyArrayItem
from skore.item.pandas_dataframe_item import PandasDataFrameItem
from skore.item.pandas_series_item import PandasSeriesItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem


class ItemRepository:
    """
    A repository for managing storage and retrieval of items.

    This class provides methods to get, put, and delete items from a storage system.
    """

    ITEM_CLASS_NAME_TO_ITEM_CLASS = {
        "MediaItem": MediaItem,
        "NumpyArrayItem": NumpyArrayItem,
        "PandasDataFrameItem": PandasDataFrameItem,
        "PandasSeriesItem": PandasSeriesItem,
        "PrimitiveItem": PrimitiveItem,
        "CrossValidationItem": CrossValidationItem,
        "SklearnBaseEstimatorItem": SklearnBaseEstimatorItem,
    }

    def __init__(self, storage: AbstractStorage):
        """
        Initialize the ItemRepository with a storage system.

        Parameters
        ----------
        storage : AbstractStorage
            The storage system to be used by the repository.
        """
        self.storage = storage

    def get_item(self, key) -> Item:
        """
        Retrieve an item from storage.

        Parameters
        ----------
        key : Any
            The key used to identify the item in storage.

        Returns
        -------
        Item
            The retrieved item.
        """
        value = self.storage[key]
        item_class_name = value["item_class_name"]
        item_class = ItemRepository.ITEM_CLASS_NAME_TO_ITEM_CLASS[item_class_name]
        item = value["item"]

        return item_class(**item)

    def put_item(self, key, item: Item) -> None:
        """
        Store an item in storage.

        Parameters
        ----------
        key : Any
            The key to use for storing the item.
        item : Item
            The item to be stored.
        """
        item_parameters = item.__parameters__

        if key in self.storage:
            item_parameters["created_at"] = self.storage[key]["item"]["created_at"]

        self.storage[key] = {
            "item_class_name": item.__class__.__name__,
            "item": item_parameters,
        }

    def delete_item(self, key):
        """
        Delete an item from storage.

        Parameters
        ----------
        key : Any
            The key of the item to be deleted.
        """
        del self.storage[key]

    def keys(self) -> list[str]:
        """
        Get all keys of items stored in the repository.

        Returns
        -------
        list[str]
            A list of all keys in the storage.
        """
        return list(self.storage.keys())
