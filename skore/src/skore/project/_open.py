"""Helper to open a project."""

from pathlib import Path
from typing import Literal, Optional, Union

from skore.project.project import Project


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    if_exists: Optional[Literal["raise", "load"]] = "raise",
) -> Project:
    """Open a project given a project name or path and launch skore UI.

    This function :
        - opens the project if it already exists,
        - creates the project if it does not exist,
        - and creates by overwriting a pre-existing project if ``overwrite`` is set to
        ``True``.

    Parameters
    ----------
    project_path: Path-like, optional
        The relative or absolute path of the project, default "project.skore".
    if_exists: Literal["raise", "load"], optional
        Raise an exception if the project already exists, or load it, default "raise".

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
    return Project(project_path, if_exists=if_exists)
