"""Define a Project."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from skore.item import (
    CrossValidationItem,
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

    def put(self, key: Union[str, dict[str, Any]], value: Optional[Any] = None):
        """Add one or more key-value pairs to the Project.

        If `key` is a string, then `put` adds the single `key`-`value` pair mapping to
        the Project.
        If `key` is a dict, it is interpreted as multiple key-value pairs to add to
        the Project.
        If an item with the same key already exists, its value is replaced by the new
        one.

        The dict format is the same as equivalent to running `put` for each individual
        key-value pair. In other words,
        ```python
        project.put({"hello": 1, "goodbye": 2})
        ```
        is equivalent to
        ```python
        project.put("hello", 1)
        project.put("goodbye", 2)
        ```
        In particular, this means that if some key-value pair is invalid
        (e.g. if a key is not a string, or a value's type is not supported),
        then all the key-value pairs up to the first offending key-value pair will
        be successfully inserted, *and then* an error will be raised.

        Parameters
        ----------
        key : str | dict[str, Any]
            The key to associate with `value` in the Project,
            or dict of key-value pairs to add to the Project.
        value : Any, optional
            The value to associate with `key` in the Project.
            If `key` is a dict, this argument is ignored.

        Raises
        ------
        ProjectPutError
            If the key-value pair(s) cannot be saved properly.
        """
        if isinstance(key, dict):
            for key_, value in key.items():
                self.put_one(key_, value)
        else:
            self.put_one(key, value)

    def put_one(self, key: str, value: Any):
        """Add a key-value pair to the Project.

        Parameters
        ----------
        key : str
            The key to associate with `value` in the Project. Must be a string.
        value : Any
            The value to associate with `key` in the Project.

        Raises
        ------
        ProjectPutError
            If the key-value pair cannot be saved properly.
        """
        try:
            item = object_to_item(value)
            self.put_item(key, item)
        except (NotImplementedError, TypeError) as e:
            raise ProjectPutError(
                "Key-value pair could not be inserted in the Project"
            ) from e

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
        elif isinstance(item, CrossValidationItem):
            return item.cv_results_serialized
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
        -------
        item : Item
            The Item corresponding to key `key`.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return self.item_repository.get_item(key)

    def list_item_keys(self) -> list[str]:
        """List all item keys in the Project.

        Returns
        -------
        list[str]
            The list of item keys. The list is empty if there is no item.
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
            The list of view keys. The list is empty if there is no view.
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
