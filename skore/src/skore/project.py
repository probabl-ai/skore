"""Define a Project."""

import logging
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal, Union

from skore.item import (
    Item,
    ItemRepository,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PandasSeriesItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
    object_to_item,
)
from skore.persistence.disk_cache_storage import DirectoryDoesNotExist, DiskCacheStorage
from skore.view.view import View
from skore.view.view_repository import ViewRepository

logger = logging.getLogger(__name__)


class ProjectPutError(Exception):
    """One more key-value pairs could not be saved in the Project."""


class Project:
    """A project is a collection of items that are stored in a storage."""

    def __init__(
        self,
        item_repository: ItemRepository,
        view_repository: ViewRepository,
    ):
        self.item_repository = item_repository
        self.view_repository = view_repository

    @singledispatchmethod
    def put(self, key: str, value: Any, on_error: Literal["warn", "raise"] = "warn"):
        """Add a value to the Project.
        
        If an item with the same key already exists, its value is replaced by the new one.
        If `on_error` is "raise", any error stops the execution. If `on_error`
        is "warn" (or anything other than "raise"), a warning is shown instead.

        Parameters
        ----------
        key : str
            The key to associate with `value` in the Project. Must be a string.
        value : Any
            The value to associate with `key` in the Project.
        on_error : {"warn", "raise"}, optional
            Upon error (e.g. if the key is not a string), whether to raise an error or
            to print a warning. Default is "warn".

        Raises
        ------
        ProjectPutError
            If the key-value pair cannot be saved properly, and `on_error` is "raise".
        """
        try:
            item = object_to_item(value)
            self.put_item(key, item)
        except (NotImplementedError, TypeError) as e:
            if on_error == "raise":
                raise ProjectPutError(
                    "Key-value pair could not be inserted in the Project"
                ) from e

            logger.warning(
                "Key-value pair could not be inserted in the Project "
                f"due to the following error: {e}"
            )

    @put.register
    def put_several(
        self, key_to_value: dict, on_error: Literal["warn", "raise"] = "warn"
    ):
        """Add several values to the Project.

        If `on_error` is "raise", the first error stops the execution (so the
        later key-value pairs will not be inserted). If `on_error` is "warn" (or
        anything other than "raise"), errors do not stop the execution, and are
        shown as they come as warnings; all the valid key-value pairs are inserted.

        Parameters
        ----------
        key_to_value : dict[str, Any]
            The key-value pairs to put in the Project. Keys must be strings.
        on_error : {"warn", "raise"}, optional
            Upon error (e.g. if a key is not a string), whether to raise an error or
            to print a warning. Default is "warn".

        Raises
        ------
        ProjectPutError
            If a key-value pair in `key_to_value` cannot be saved properly,
            and `on_error` is "raise".
        """
        for key, value in key_to_value.items():
            self.put(key, value, on_error=on_error)

    def put_item(self, key: str, item: Item):
        """Add an Item to the Project."""
        if not isinstance(key, str):
            raise TypeError(
                f"Key must be a string; key '{key}' is of type '{type(key)}'"
            )

        self.item_repository.put_item(key, item)

    def get(self, key: str) -> Any:
        """Get the value corresponding to `key` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        item = self.get_item(key)

        if isinstance(item, PrimitiveItem):
            return item.primitive
        elif isinstance(item, NumpyArrayItem):
            return item.array
        elif isinstance(item, PandasDataFrameItem):
            return item.dataframe
        elif isinstance(item, PandasSeriesItem):
            return item.series
        elif isinstance(item, SklearnBaseEstimatorItem):
            return item.estimator
        elif isinstance(item, MediaItem):
            return item.media_bytes
        else:
            raise ValueError(f"Item {item} is not a known item type.")

    def get_item(self, key: str) -> Item:
        """Get the item corresponding to `key` from the Project.

        Parameters
        ----------
        key: str
            The key corresponding to the item to get.

        Returns
        ----------
        item : Item
            The Item corresponding to key `key`.

        Raises
        ------
        KeyError
            If the key does not correspond to any item."""
        return self.item_repository.get_item(key)

    def list_item_keys(self) -> list[str]:
        """List all item keys in the Project.

        Returns
        -------
        list[str]
            The list of item keys. The list is empty is there is no item.
        """
        return self.item_repository.keys()

    def delete_item(self, key: str):
        """Delete the item corresponding to `key` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the item to delete.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        self.item_repository.delete_item(key)

    def put_view(self, key: str, view: View):
        """Add a view to the Project."""
        self.view_repository.put_view(key, view)

    def get_view(self, key: str) -> View:
        """Get the view corresponding to `key` from the Project.

        Parameters
        ----------
        key : str
            The key of the item to get.

        Returns
        -------
        View
            The view corresponding to `key`.

        Raises
        ------
        KeyError
            If the key does not correspond to any view.
        """
        return self.view_repository.get_view(key)

    def delete_view(self, key: str):
        """Delete the view corresponding to `key` from the Project.

        Parameters
        ----------
        key : str
            The key corresponding to the view to delete.

        Raises
        ------
        KeyError
            If the key does not correspond to any view.
        """
        return self.view_repository.delete_view(key)

    def list_view_keys(self) -> list[str]:
        """List all view keys in the Project.

        Returns
        -------
        list[str]
            The list of view keys. The list is empty is there is no view.
        """
        return self.view_repository.keys()


class ProjectLoadError(Exception):
    """Failed to load project."""


def load(project_name: Union[str, Path]) -> Project:
    """Load an existing Project given a project name or path."""
    # Transform a project name to a directory path:
    # - Resolve relative path to current working directory,
    # - Check that the file ends with the ".skore" extension,
    #    - If not provided, it will be automatically appended,
    # - If project name is an absolute path, we keep that path.

    path = Path(project_name).resolve()

    if path.suffix != ".skore":
        path = path.parent / (path.name + ".skore")

    if not Path(path).exists():
        raise ProjectLoadError(f"Project '{path}' does not exist: did you create it?")

    try:
        # FIXME should those hardcoded string be factorized somewhere ?
        item_storage = DiskCacheStorage(directory=Path(path) / "items")
        item_repository = ItemRepository(storage=item_storage)
        view_storage = DiskCacheStorage(directory=Path(path) / "views")
        view_repository = ViewRepository(storage=view_storage)
        project = Project(
            item_repository=item_repository,
            view_repository=view_repository,
        )
    except DirectoryDoesNotExist as e:
        missing_directory = e.args[0].split()[1]
        raise ProjectLoadError(
            f"Project '{path}' is corrupted: "
            f"directory '{missing_directory}' should exist. "
            "Consider re-creating the project."
        ) from e

    return project
