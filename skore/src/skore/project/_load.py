"""Helper to load a project."""

from pathlib import Path
from typing import Union

from skore.exceptions import ProjectLoadError
from skore.persistence.repository import ItemRepository, ViewRepository
from skore.persistence.storage.disk_cache_storage import (
    DirectoryDoesNotExist,
    DiskCacheStorage,
)
from skore.project._launch import ServerInfo
from skore.project.project import Project


def _load(project_name: Union[str, Path]) -> Project:
    """Load an existing Project given a project name or path.

    Transforms a project name to a directory path as follows:
    - Resolves relative path to current working directory,
    - Checks that the file ends with the ".skore" extension,
    - If not provided, it will be automatically appended,
    - If project name is an absolute path, keeps that path.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.

    Returns
    -------
    Project
        The loaded Project instance.
    """
    from skore import console  # avoid circular import

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
            name=path.name,
            path=path,
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

    server_info = ServerInfo.rejoin(project)
    if server_info is not None:
        project._server_info = server_info
        console.print(
            f"Project '{project.name}' rejoined skore UI at URL: "
            f"http://localhost:{server_info.port}"
        )

    return project
