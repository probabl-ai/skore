"""In-memory storage."""

from collections.abc import Iterator
from typing import Any

from .abstract_storage import AbstractStorage


class InMemoryStorage(AbstractStorage):
    """In-memory storage."""

    def __init__(self):
        """
        Initialize an empty in-memory storage.

        The storage is implemented as a dictionary.
        """
        self.storage = {}

    def __getitem__(self, key: str) -> Any:
        """
        Get the item for the specified key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        Any
            The value associated with the key.

        Raises
        ------
        KeyError
            If the key is not found in the storage.
        """
        return self.storage[key]

    def __setitem__(self, key: str, value: Any):
        """
        Set the item for the specified key.

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
        Delete the item for the specified key.

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
        Yield the keys.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys in the storage.
        """
        return iter(self.storage.keys())

    def values(self) -> Iterator[Any]:
        """
        Yield the values.

        Returns
        -------
        Iterator[Any]
            An iterator yielding all values in the storage.
        """
        return iter(self.storage.values())

    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Yield the pairs (key, value).

        Returns
        -------
        Iterator[tuple[str, Any]]
            An iterator yielding all (key, value) pairs in the storage.
        """
        return iter(self.storage.items())

    def __repr__(self) -> str:
        """
        Return a string representation of the storage.

        Returns
        -------
        str
            A string representation of the storage.
        """
        return "InMemoryStorage()"
