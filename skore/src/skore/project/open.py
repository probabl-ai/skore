"""Command to open a Project."""

from pathlib import Path
from typing import Union

from skore.project.create import _create
from skore.project.load import _load
from skore.project.project import Project


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    create: bool = True,
    overwrite: bool = False,
) -> Project:
    """Open a project given a project name or path.

    This function creates the project if it does not exist, and it overwrites
    a pre-existing project if ``overwrite`` is set to ``True``.

    Parameters
    ----------
    project_path: Path-like, default="project.skore"
        The relative or absolute path of the project.
    create: bool, default=True
        Create the project if it does not exist.
    overwrite: bool, default=False
        Overwrite the project file if it already exists and ``create`` is ``True``.
        Has no effect otherwise.

    Returns
    -------
    Project
        The opened Project instance.

    Raises
    ------
    FileNotFoundError
        If path is not found and ``create`` is set to ``False``
    ProjectCreationError
        If project creation fails for some reason (e.g. if the project path is invalid)
    """
    if create and not overwrite:
        try:
            return _load(project_path)
        except FileNotFoundError:
            return _create(project_path, overwrite=overwrite)

    if not create:
        return _load(project_path)

    return _create(project_path, overwrite=overwrite)
