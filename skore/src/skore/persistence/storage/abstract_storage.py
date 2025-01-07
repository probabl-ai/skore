"""Abstract storage interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class AbstractStorage(ABC):
    """Persist data in a storage."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def keys(self) -> Iterator[str]:
        """
        Yield the keys in the storage.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys in the storage.
        """

    @abstractmethod
    def values(self) -> Iterator[Any]:
        """
        Yield the values in the storage.

        Returns
        -------
        Iterator[Any]
            An iterator yielding all values in the storage.
        """

    @abstractmethod
    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Yield the pairs (key, value) in the storage.

        Returns
        -------
        Iterator[tuple[str, Any]]
            An iterator yielding all (key, value) pairs in the storage.
        """

    def __contains__(self, key: str) -> bool:
        """
        Return True if the storage has the specified key, else False.

        Parameters
        ----------
        key : str
            The key to check for existence in the storage.

        Returns
        -------
        bool
            True if the key is in the storage, else False.
        """
        return key in self.keys()

    def __len__(self) -> int:
        """
        Return the number of items in the storage.

        Returns
        -------
        int
            The number of items in the storage.
        """
        return len(list(self.keys()))

    def __iter__(self) -> Iterator[str]:
        """
        Yield the keys in the storage.

        Returns
        -------
        Iterator[str]
            An iterator yielding all keys in the storage.
        """
        return self.keys()
