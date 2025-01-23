"""In-memory storage."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from diskcache import Cache

from .abstract_storage import AbstractStorage


class DirectoryDoesNotExist(Exception):
    """Directory does not exist."""


class DiskCacheStorage(AbstractStorage):
    """
    Disk-based storage implementation using diskcache.

    This class provides a persistent storage solution using the diskcache library,
    which allows for efficient caching of data on disk.

    Parameters
    ----------
    directory : Path
        The directory path where the cache will be stored.

    Attributes
    ----------
    storage : Cache
        The underlying diskcache Cache object.
    """

    def __init__(self, directory: Path):
        """
        Initialize the DiskCacheStorage with the specified directory.

        Parameters
        ----------
        directory : Path
            The directory path where the cache will be stored.
        """
        if not directory.exists():
            raise DirectoryDoesNotExist(f"Directory {directory} does not exist.")

        self.directory = directory

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve an item from the storage.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        Any
            The value associated with the given key.

        Raises
        ------
        KeyError
            If the key is not found in the storage.
        """
        with Cache(self.directory) as storage:
            return storage[key]

    def __setitem__(self, key: str, value: Any):
        """
        Set an item in the storage.

        Parameters
        ----------
        key : str
            The key to associate with the value.
        value : Any
            The value to store.
        """
        with Cache(self.directory) as storage:
            storage[key] = value

    def __delitem__(self, key: str):
        """
        Delete an item from the storage.

        Parameters
        ----------
        key : str
            The key of the item to delete.

        Raises
        ------
        KeyError
            If the key is not found in the storage.
        """
        with Cache(self.directory) as storage:
            del storage[key]

    def keys(self) -> Iterator[str]:
        """
        Get an iterator over the keys in the storage.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys in the storage.
        """
        with Cache(self.directory) as storage:
            return storage.iterkeys()

    def values(self) -> Iterator[Any]:
        """
        Get an iterator over the values in the storage.

        Returns
        -------
        Iterator[Any]
            An iterator yielding all values in the storage.
        """
        with Cache(self.directory) as storage:
            for key in storage.iterkeys():
                yield storage[key]

    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Get an iterator over the (key, value) pairs in the storage.

        Returns
        -------
        Iterator[tuple[str, Any]]
            An iterator yielding all (key, value) pairs in the storage.
        """
        with Cache(self.directory) as storage:
            for key in storage.iterkeys():
                yield (key, storage[key])

    def __repr__(self) -> str:
        """
        Return a string representation of the storage.

        Returns
        -------
        str
            A string representation of the storage.
        """
        return f"DiskCacheStorage(directory='{self.directory}')"
