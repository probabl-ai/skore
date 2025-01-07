"""Load project helper."""

from pathlib import Path
from typing import Union

from skore.item import ItemRepository
from skore.persistence.disk_cache_storage import DirectoryDoesNotExist, DiskCacheStorage
from skore.project.project import Project
from skore.view.view_repository import ViewRepository


class ProjectLoadError(Exception):
    """Failed to load project."""


def _load(project_name: Union[str, Path]) -> Project:
    """Load an existing Project given a project name or path.

    Transforms a project name to a directory path as follows:
    - Resolves relative path to current working directory,
    - Checks that the file ends with the ".skore" extension,
    - If not provided, it will be automatically appended,
    - If project name is an absolute path, keeps that path.
    """
    path = Path(project_name).resolve()

    if path.suffix != ".skore":
        path = path.parent / (path.name + ".skore")

    if not Path(path).exists():
        raise FileNotFoundError(f"Project '{path}' does not exist: did you create it?")

    try:
        # FIXME: Should those hardcoded strings be factorized somewhere ?
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
