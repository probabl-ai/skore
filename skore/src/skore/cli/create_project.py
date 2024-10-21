"""Implement the "create project" feature."""

import re
from pathlib import Path
from typing import Optional, Union

from skore.cli import logger
from skore.project import load
from skore.view.view import View


class InvalidProjectNameError(Exception):
    """The project name does not fit with one or more of the project name rules.

    - The project name must start with an alphanumeric character, and must not contain
    special characters other than '_' (underscore) and '-' (hyphen).
    - The project name must be at most 255 characters long (including ".skore").
    - The project name must not be a reserved OS file name.
    For example, CON, AUX, NUL... on Windows.
    """


def validate_project_name(project_name: str) -> tuple[bool, Optional[Exception]]:
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


class ProjectCreationError(Exception):
    """Project creation failed."""


class ProjectAlreadyExistsError(Exception):
    """A project with this name already exists."""


class ProjectPermissionError(Exception):
    """Permissions in the directory do not allow creating a file."""


def __create(
    project_name: Union[str, Path], working_dir: Optional[Path] = None
) -> Path:
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

    Returns
    -------
    The project directory path
    """
    project_path = Path(project_name)

    # Remove trailing ".skore" if it exists to check the name is valid
    checked_project_name: str = project_path.name.split(".skore")[0]

    validation_passed, validation_error = validate_project_name(checked_project_name)
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

    try:
        project_directory.mkdir()
    except FileExistsError as e:
        raise ProjectAlreadyExistsError(
            f"Unable to create project file '{project_directory}' because a file "
            "with that name already exists. Please choose a different name or delete "
            "the existing file."
        ) from e
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
