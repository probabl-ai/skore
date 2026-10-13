"""Custom exceptions."""


class InvalidProjectNameError(Exception):
    """The project name does not fit with one or more of the project name rules.

    - The project name must start with an alphanumeric character, and must not contain
    special characters other than '_' (underscore) and '-' (hyphen).
    - The project name must be at most 255 characters long (including ".skore").
    - The project name must not be a reserved OS file name.
    For example, CON, AUX, NUL... on Windows.
    """


class ProjectCreationError(Exception):
    """Project creation failed."""


class ProjectPermissionError(Exception):
    """Permissions in the directory do not allow creating a file."""


class ProjectLoadError(Exception):
    """Failed to load project."""
