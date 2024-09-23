"""Define a Project."""

import logging
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, Literal

from skore.item import (
    Item,
    ItemRepository,
    MediaItem,
    NumpyArrayItem,
    PandasDataFrameItem,
    PrimitiveItem,
    SklearnBaseEstimatorItem,
    object_to_item,
)
from skore.layout import Layout, LayoutRepository
from skore.persistence.disk_cache_storage import DirectoryDoesNotExist, DiskCacheStorage

logger = logging.getLogger(__name__)


class Project:
    """A project is a collection of items that are stored in a storage."""

    def __init__(
        self,
        item_repository: ItemRepository,
        layout_repository: LayoutRepository,
    ):
        self.item_repository = item_repository
        self.layout_repository = layout_repository

    @singledispatchmethod
    def put(self, key: str, value: Any, on_error: Literal["warn", "raise"] = "warn"):
        """Add a value to the Project.

        Parameters
        ----------
        key : str
            The key to associate with `value` in the Project. Must be a string.
        value : Any
            The value to associate with `key` in the Project.
        on_error : "warn" or "raise", optional
            Upon error (e.g. if the key is not a string), whether to raise an error or
            to print a warning. Default is "warn".
        """
        self.put_several({key: value}, on_error)

    @put.register
    def put_several(
        self, key_to_value: dict, on_error: Literal["warn", "raise"] = "warn"
    ):
        """Add several values to the Project.

        All valid key-value pairs will be added to the Project; errors caused
        by invalid key-value pairs (e.g. if the key is not a string) will be
        collected and shown all at once after all the insertions, either as a
        warning or an Exception, depending on the value of `on_error`.

        Parameters
        ----------
        key : str
            The key to associate with `value` in the Project. Must be a string.
        value : Any
            The value to associate with `key` in the Project.
        on_error : "warn" or "raise", optional
            Upon error (e.g. if the key is not a string), whether to raise an error or
            to print a warning. Default is "warn".
        """
        errors = []

        for key, value in key_to_value.items():
            key_error = None
            value_error = None

            if not isinstance(key, str):
                key_error = TypeError(
                    f"All keys must be strings; key '{key}' is of type '{type(key)}'"
                )
                errors.append(key_error)

            try:
                item = object_to_item(value)
            except NotImplementedError as e:
                value_error = e
                errors.append(value_error)

            if key_error or value_error:
                continue

            self.put_item(key, item)

        if errors:
            if on_error == "raise":
                raise ExceptionGroup(
                    "Several items could not be inserted in the Project", errors
                )

            logger.warning(
                "Several items could not be inserted in the Project "
                f"due to the following errors: {errors}"
            )

    def put_item(self, key: str, item: Item):
        """Add an Item to the Project."""
        self.item_repository.put_item(key, item)

    def get(self, key: str) -> Any:
        """Get the value corresponding to `key` from the Project."""
        item = self.get_item(key)

        if isinstance(item, PrimitiveItem):
            return item.primitive
        elif isinstance(item, NumpyArrayItem):
            return item.array
        elif isinstance(item, PandasDataFrameItem):
            return item.dataframe
        elif isinstance(item, SklearnBaseEstimatorItem):
            return item.estimator
        elif isinstance(item, MediaItem):
            return item.media_bytes
        else:
            raise ValueError(f"Item {item} is not a known item type.")

    def get_item(self, key: str) -> Item:
        """Add the Item corresponding to `key` from the Project."""
        return self.item_repository.get_item(key)

    def list_keys(self) -> list[str]:
        """List all keys in the Project."""
        return self.item_repository.keys()

    def delete_item(self, key: str):
        """Delete an item from the Project."""
        self.item_repository.delete_item(key)

    def put_report_layout(self, layout: Layout):
        """Add a report layout to the Project."""
        self.layout_repository.put_layout(layout)

    def get_report_layout(self) -> Layout:
        """Get the report layout corresponding to `key` from the Project."""
        try:
            return self.layout_repository.get_layout()
        except KeyError:
            return []


class ProjectLoadError(Exception):
    """Failed to load project."""


def load(project_name: str | Path) -> Project:
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
        layout_storage = DiskCacheStorage(directory=Path(path) / "layouts")
        layout_repository = LayoutRepository(storage=layout_storage)
        project = Project(
            item_repository=item_repository,
            layout_repository=layout_repository,
        )
    except DirectoryDoesNotExist as e:
        missing_directory = e.args[0].split()[1]
        raise ProjectLoadError(
            f"Project '{path}' is corrupted: "
            f"directory '{missing_directory}' should exist. "
            "Consider re-creating the project."
        ) from e

    return project
