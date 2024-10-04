"""Implement a repository for Views."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore.persistence.abstract_storage import AbstractStorage
    from skore.view.view import View


class ViewRepository:
    """
    A repository for managing storage and retrieval of Views.

    This class provides methods to get, put, and delete Views from a storage system.
    """

    def __init__(self, storage: AbstractStorage):
        """
        Initialize the ViewRepository with a storage system.

        Parameters
        ----------
        storage : AbstractStorage
            The storage system to be used by the repository.
        """
        self.storage = storage

    def get_view(self, key: str) -> View:
        """
        Retrieve the View from storage.

        Parameters
        ----------
        key : str
            A key at which to look for a View.

        Returns
        -------
        View
            The retrieved View.

        Raises
        ------
        KeyError
            When `key` is not present in the underlying storage.
        """
        return self.storage[key]

    def put_view(self, key: str, view: View):
        """
        Store a view in storage.

        Parameters
        ----------
        view : View
            The view to be stored.
        """
        self.storage[key] = view

    def delete_view(self, key: str):
        """Delete the view from storage."""
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
