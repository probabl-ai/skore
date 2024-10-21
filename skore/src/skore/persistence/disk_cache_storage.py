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
        self.storage = Cache(directory)

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
        return self.storage[key]

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
        self.storage[key] = value

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
        del self.storage[key]

    def keys(self) -> Iterator[str]:
        """
        Get an iterator over the keys in the storage.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys in the storage.
        """
        return self.storage.iterkeys()

    def values(self) -> Iterator[Any]:
        """
        Get an iterator over the values in the storage.

        Returns
        -------
        Iterator[Any]
            An iterator yielding all values in the storage.
        """
        for key in self.storage.iterkeys():
            yield self.storage[key]

    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Get an iterator over the (key, value) pairs in the storage.

        Returns
        -------
        Iterator[tuple[str, Any]]
            An iterator yielding all (key, value) pairs in the storage.
        """
        for key in self.storage.iterkeys():
            yield (key, self.storage[key])

    def __repr__(self) -> str:
        """
        Return a string representation of the storage.

        Returns
        -------
        str
            A string representation of the storage.
        """
        return f"DiskCacheStorage(directory='{self.storage.directory}')"
