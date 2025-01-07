"""Create project helper."""

import re
import shutil
from pathlib import Path
from typing import Optional, Union

from skore.exceptions import (
    InvalidProjectNameError,
    ProjectCreationError,
    ProjectPermissionError,
)
from skore.project.load import _load
from skore.project.project import Project, logger
from skore.utils._logger import logger_context
from skore.view.view import View


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


def _create(
    project_name: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = False,
) -> Project:
    """Create a project file named according to ``project_name``.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path. If relative,
        will be interpreted as relative to the current working directory.
    overwrite : bool
        If ``True``, overwrite an existing project with the same name.
        If ``False``, raise an error if a project with the same name already exists.
    verbose : bool
        Whether or not to display info logs to the user.

    Returns
    -------
    The created project
    """
    from skore import console  # avoid circular import

    with logger_context(logger, verbose):
        project_path = Path(project_name)

        # Remove trailing ".skore" if it exists to check the name is valid
        checked_project_name: str = project_path.name.split(".skore")[0]

        validation_passed, validation_error = _validate_project_name(
            checked_project_name
        )
        if not validation_passed:
            raise ProjectCreationError(
                f"Unable to create project file '{project_path}'."
            ) from validation_error

        # The file must end with the ".skore" extension.
        # If not provided, it will be automatically appended.
        # If project name is an absolute path, we keep that path

        project_directory = project_path.with_name(checked_project_name + ".skore")

        if project_directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Unable to create project file '{project_directory}' because a "
                    "file with that name already exists. Please choose a different "
                    "name or use the --overwrite flag with the CLI or overwrite=True "
                    "with the API."
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

        # Once the main project directory has been created, created the nested
        # directories

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

        p = _load(project_directory)
        p.put_view("default", View(layout=[]))

        console.rule("[bold cyan]skore[/bold cyan]")
        console.print(f"Project file '{project_directory}' was successfully created.")
        return p
