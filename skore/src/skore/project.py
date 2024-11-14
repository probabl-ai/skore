"""Define a Project."""

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from skore.exceptions import (
    InvalidProjectNameError,
    ProjectAlreadyExistsError,
    ProjectCreationError,
    ProjectPermissionError,
)
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

        If ``key`` is a string, then `put` adds the single ``key``-``value`` pair
        mapping to the Project.
        If ``key`` is a dict, it is interpreted as multiple key-value pairs to add to
        the Project.
        If an item with the same key already exists, its value is replaced by the new
        one.

        The dict format is the same as equivalent to running :func:`~skore.Project.put`
        for each individual key-value pair. In other words,

        .. code-block:: python

            project.put({"hello": 1, "goodbye": 2})

        is equivalent to

        .. code-block:: python

            project.put("hello", 1)
            project.put("goodbye", 2)

        In particular, this means that if some key-value pair is invalid
        (e.g. if a key is not a string, or a value's type is not supported),
        then all the key-value pairs up to the first offending key-value pair will
        be successfully inserted, *and then* an error will be raised.

        Parameters
        ----------
        key : str | dict[str, Any]
            The key to associate with ``value`` in the Project,
            or dict of key-value pairs to add to the Project.
        value : Any, optional
            The value to associate with ``key`` in the Project.
            If ``key`` is a dict, this argument is ignored.

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
            The key to associate with ``value`` in the Project. Must be a string.
        value : Any
            The value to associate with ``key`` in the Project.

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
        """Get the value corresponding to ``key`` from the Project.

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
        """Get the item corresponding to ``key`` from the Project.

        Parameters
        ----------
        key: str
            The key corresponding to the item to get.

        Returns
        -------
        item : Item
            The Item corresponding to key ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return self.item_repository.get_item(key)

    def get_item_versions(self, key: str) -> list[Item]:
        """
        Get all the versions of an item associated with ``key`` from the Project.

        The list is ordered from oldest to newest "put" date.

        Parameters
        ----------
        key : str
            The key corresponding to the item to get.

        Returns
        -------
        list[Item]
            The list of items corresponding to key ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any item.
        """
        return self.item_repository.get_item_versions(key)

    def list_item_keys(self) -> list[str]:
        """List all item keys in the Project.

        Returns
        -------
        list[str]
            The list of item keys. The list is empty if there is no item.
        """
        return self.item_repository.keys()

    def delete_item(self, key: str):
        """Delete the item corresponding to ``key`` from the Project.

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
        """Get the view corresponding to ``key`` from the Project.

        Parameters
        ----------
        key : str
            The key of the item to get.

        Returns
        -------
        View
            The view corresponding to ``key``.

        Raises
        ------
        KeyError
            If the key does not correspond to any view.
        """
        return self.view_repository.get_view(key)

    def delete_view(self, key: str):
        """Delete the view corresponding to ``key`` from the Project.

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


def _validate_project_name(project_name: str) -> tuple[bool, Optional[Exception]]:
    """Validate the project name (the part before ".skore").

    Returns `(True, None)` if validation succeeded and `(False, Exception(...))`
    otherwise.
    """
    # The project name (including the .skore extension) must be between 5 and 255
    # characters long.
    # FIXME: On Linux the OS already checks filename lengths
    if len(project_name) + len(".skore") > 255:
        return False, InvalidProjectNameError(
            "Project name length cannot exceed 255 characters."
        )

    # Reserved Names: The following names are reserved and cannot be used:
    # CON, PRN, AUX, NUL
    # COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9
    # LPT1, LPT2, LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, LPT9
    reserved_patterns = "|".join(["CON", "PRN", "AUX", "NUL", r"COM\d+", r"LPT\d+"])
    if re.fullmatch(f"^({reserved_patterns})$", project_name):
        return False, InvalidProjectNameError(
            "Project name must not be a reserved OS filename."
        )

    # Allowed Characters:
    # Alphanumeric characters (a-z, A-Z, 0-9)
    # Underscore (_)
    # Hyphen (-)
    # Starting Character: The project name must start with an alphanumeric character.
    if not re.fullmatch(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", project_name):
        return False, InvalidProjectNameError(
            "Project name must contain only alphanumeric characters, '_' and '-'."
        )

    # Case Sensitivity: File names are case-insensitive on Windows and case-sensitive
    # on Unix-based systems. The CLI should warn users about potential case conflicts
    # on Unix systems.

    return True, None


def create(
    project_name: Union[str, Path],
    working_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Project:
    """Create a project file named according to `project_name`.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.
    working_dir : Path or None
        If `project_name` is not an absolute path, it will be considered relative to
        `working_dir`. If `project_name` is an absolute path, `working_dir` will have
        no effect. If set to None (the default), `working_dir` will be re-set to the
        current working directory.
    overwrite : bool
        If True, overwrite an existing project with the same name. If False, raise an
        error if a project with the same name already exists.

    Returns
    -------
    The project directory path
    """
    project_path = Path(project_name)

    # Remove trailing ".skore" if it exists to check the name is valid
    checked_project_name: str = project_path.name.split(".skore")[0]

    validation_passed, validation_error = _validate_project_name(checked_project_name)
    if not validation_passed:
        raise ProjectCreationError(
            f"Unable to create project file '{project_path}'."
        ) from validation_error

    # The file must end with the ".skore" extension.
    # If not provided, it will be automatically appended.
    # If project name is an absolute path, we keep that path

    # NOTE: `working_dir` has no effect if `checked_project_name` is absolute
    if working_dir is None:
        working_dir = Path.cwd()
    project_directory = working_dir / (
        project_path.with_name(checked_project_name + ".skore")
    )

    if project_directory.exists():
        if not overwrite:
            raise ProjectAlreadyExistsError(
                f"Unable to create project file '{project_directory}' because a file "
                "with that name already exists. Please choose a different name or "
                "use the --overwrite flag with the CLI or overwrite=True with the API."
            )
        shutil.rmtree(project_directory)

    try:
        project_directory.mkdir(parents=True)
    except PermissionError as e:
        raise ProjectPermissionError(
            f"Unable to create project file '{project_directory}'. "
            "Please check your permissions for the current directory."
        ) from e
    except Exception as e:
        raise ProjectCreationError(
            f"Unable to create project file '{project_directory}'."
        ) from e

    # Once the main project directory has been created, created the nested directories

    items_dir = project_directory / "items"
    try:
        items_dir.mkdir()
    except Exception as e:
        raise ProjectCreationError(
            f"Unable to create project file '{items_dir}'."
        ) from e

    views_dir = project_directory / "views"
    try:
        views_dir.mkdir()
    except Exception as e:
        raise ProjectCreationError(
            f"Unable to create project file '{views_dir}'."
        ) from e

    p = load(project_directory)
    p.put_view("default", View(layout=[]))

    logger.info(f"Project file '{project_directory}' was successfully created.")
    return p
