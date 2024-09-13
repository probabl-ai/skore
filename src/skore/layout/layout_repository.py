"""LayoutRepository."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore.layout import Layout
    from skore.persistence.abstract_storage import AbstractStorage


class LayoutRepository:
    """
    A repository for managing storage and retrieval of layouts.

    This class provides methods to get, put, and delete layouts from a storage system.
    """

    def __init__(self, storage: AbstractStorage):
        """
        Initialize the LayoutRepository with a storage system.

        Parameters
        ----------
        storage : AbstractStorage
            The storage system to be used by the repository.
        """
        self.storage = storage

    def get_layout(self) -> Layout:
        """
        Retrieve the layout from storage.

        Returns
        -------
        Layout
            The retrieved layout.
        """
        return self.storage["layout"]

    def put_layout(self, layout: Layout):
        """
        Store a layout in storage.

        Parameters
        ----------
        layout : Layout
            The layout to be stored.
        """
        self.storage["layout"] = layout

    def delete_layout(self):
        """Delete the layout from storage."""
        del self.storage["layout"]

    def keys(self) -> list[str]:
        """
        Get all keys of items stored in the repository.

        Returns
        -------
        list[str]
            A list of all keys in the storage.
        """
        return list(self.storage.keys())
