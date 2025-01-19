from pathlib import Path
from typing import Union

from skore.project._manage import _create, _launch, _load
from skore.project.project import Project


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    create: bool = True,
    overwrite: bool = False,
    serve: bool = True,
    port: int | None = None,
    verbose: bool = False,
) -> Project:
    """Open a project given a project name or path and launch skore UI.

    This function :
        - opens the project if it already exists,
        - creates the project if it does not exist,
        - and creates by overwriting a pre-existing project if ``overwrite`` is set to
        ``True``.

    Parameters
    ----------
    project_path: Path-like, default="project.skore"
        The relative or absolute path of the project.
    create: bool, default=True
       Whether or not to create the project, if it does not already exist.
    overwrite: bool, default=False
        Overwrite the project file if it already exists and ``create`` is ``True``.
        Has no effect otherwise.
    serve: bool, default=True
        Whether to launch the skore UI.
    port: int | None, default=None
        Port at which to bind the UI server. If ``None``, the server will be bound to
        an available port.
    verbose : bool, default=False
        Whether or not to display info logs to the user.

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
    try:
        project = _load(project_path)
        if overwrite:
            # let _create handle the overwrite flag
            project = _create(project_path, overwrite=overwrite, verbose=verbose)
    except FileNotFoundError:
        if create:
            project = _create(project_path, overwrite=overwrite, verbose=verbose)
        else:
            raise

    if serve:
        _launch(project, port=port, verbose=verbose)

    return project
