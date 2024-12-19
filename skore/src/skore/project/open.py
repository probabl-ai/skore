"""Command to open a Project."""

from pathlib import Path
from typing import Union

from skore.project.project import Project

from .create import create as create_project
from .load import load


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    create: bool = True,
    overwrite: bool = False,
) -> Project:
    """Open a project given a project name or path.

    Parameters
    ----------
    project_path: Path-like, default is "project.skore"
        The relative or absolute path of the project.
    create: bool, default is True
        Create the project if it does not exist.
    overwrite: bool, default is False
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
        If project path is invalid, or if the project already exists and
        ``overwrite`` is set to ``False``
    """
    # The default
    if create and not overwrite:
        try:
            return load(project_path)
        except FileNotFoundError:
            return create_project(project_path, overwrite=overwrite)

    if not create:
        return load(project_path)

    return create_project(project_path, overwrite=overwrite)
