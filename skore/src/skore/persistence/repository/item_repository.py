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
from skore.item.polars_dataframe_item import PolarsDataFrameItem
from skore.item.polars_series_item import PolarsSeriesItem
from skore.item.primitive_item import PrimitiveItem
from skore.item.sklearn_base_estimator_item import SklearnBaseEstimatorItem


class ItemRepository:
    """
    A repository for managing storage and retrieval of items.

    This class provides methods to get, put, and delete items from a storage system.

    Additionally, it keeps a record of all previously inserted items, by treating the
    storage as a map from keys to *lists of* values.
    """

    ITEM_CLASS_NAME_TO_ITEM_CLASS = {
        "MediaItem": MediaItem,
        "NumpyArrayItem": NumpyArrayItem,
        "PandasDataFrameItem": PandasDataFrameItem,
        "PandasSeriesItem": PandasSeriesItem,
        "PolarsDataFrameItem": PolarsDataFrameItem,
        "PolarsSeriesItem": PolarsSeriesItem,
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

    @staticmethod
    def __deconstruct_item(item: Item) -> dict:
        return {
            "item_class_name": item.__class__.__name__,
            "item": item.__parameters__,
        }

    @staticmethod
    def __construct_item(value) -> Item:
        item_class_name = value["item_class_name"]
        item_class = ItemRepository.ITEM_CLASS_NAME_TO_ITEM_CLASS[item_class_name]
        item = value["item"]

        return item_class(**item)

    def get_item(self, key) -> Item:
        """
        Get an item from storage.

        In practice, since each key is associated with a list of values,
        this will return the latest one.

        Parameters
        ----------
        key : Any
            The key used to identify the item in storage.

        Returns
        -------
        Item
            The retrieved item.
        """
        return ItemRepository.__construct_item(self.storage[key][-1])

    def get_item_versions(self, key) -> list[Item]:
        """
        Get all the versions of an item associated with `key` from the storage.

        The list is ordered from oldest to newest "put" date.

        Parameters
        ----------
        key : Any
            The key used to identify the item in storage.

        Returns
        -------
        list[Item]
            The retrieved list of items.
        """
        return [ItemRepository.__construct_item(value) for value in self.storage[key]]

    def put_item(self, key, item: Item) -> None:
        """
        Store an item in storage.

        This appends to a list of items previously associated with key `key`.

        Parameters
        ----------
        key : Any
            The key to use for storing the item.
        item : Item
            The item to be stored.
        """
        _item = ItemRepository.__deconstruct_item(item)

        if key in self.storage:
            items = self.storage[key]
            _item["item"]["created_at"] = items[0]["item"]["created_at"]

            self.storage[key] = items + [_item]
        else:
            self.storage[key] = [_item]

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
