"""ItemRepository for managing storage and retrieval of items.

This module provides the ItemRepository class, which is responsible for
storing, retrieving, and deleting items in a storage system.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Union

import skore.persistence.item

if TYPE_CHECKING:
    from skore.persistence.item import Item
    from skore.persistence.storage import AbstractStorage


class ItemRepository:
    """
    A repository for managing storage and retrieval of items.

    This class provides methods to get, put, and delete items from a storage system.

    Additionally, it keeps a record of all previously inserted items, by treating the
    storage as a map from keys to *lists of* values.
    """

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
        item_class = getattr(skore.persistence.item, item_class_name)
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
            A list of all keys.
        """
        return list(self.storage.keys())

    def __iter__(self) -> Iterator[str]:
        """
        Yield the keys of items stored in the repository.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys.
        """
        yield from self.storage

    def set_item_note(self, key: str, note: str, *, version=-1):
        """Attach a note to key ``key``.

        Parameters
        ----------
        key : str
            The key of the item to annotate.
            May be qualified with a version number through the ``version`` argument.
        note : str
            The note to be attached.
        version : int, default=-1
            The version of the key to annotate. Default is the latest version.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.
        TypeError
            If ``key`` or ``note`` is not a string.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key should be a string; got {type(key)}")
        if not isinstance(note, str):
            raise TypeError(f"Note should be a string; got {type(note)}")

        try:
            old = self.storage[key]
            old[version]["item"]["note"] = note
            self.storage[key] = old
        except IndexError as e:
            raise KeyError((key, version)) from e

    def get_item_note(self, key: str, *, version=-1) -> Union[str, None]:
        """Retrieve a note previously attached to key ``key``.

        Parameters
        ----------
        key : str
            The key of the annotated item.
            May be qualified with a version number through the ``version`` argument.
        version : int, default=-1
            The version of the annotated key. Default is the latest version.

        Returns
        -------
        The attached note, or None if no note is attached.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.
        """
        try:
            return self.storage[key][version]["item"]["note"]
        except IndexError as e:
            raise KeyError((key, version)) from e

    def delete_item_note(self, key: str, *, version=-1):
        """Delete a note previously attached to key ``key``.

        Parameters
        ----------
        key : str
            The key of the annotated item.
            May be qualified with a version number through the ``version`` argument.
        version : int, default=-1
            The version of the annotated key. Default is the latest version.

        Raises
        ------
        KeyError
            If the ``(key, version)`` couple does not exist.
        """
        try:
            old = self.storage[key]
            old[version]["item"]["note"] = None
            self.storage[key] = old
        except IndexError as e:
            raise KeyError((key, version)) from e
