"""Helper to open a project."""

from pathlib import Path
from typing import Union

from skore.project.project import Project


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    create: bool = True,
    overwrite: bool = False,
    serve: bool = True,
    keep_alive: Union[str, bool] = "auto",
    port: Union[int, None] = None,
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
    keep_alive : Union[str, bool], default="auto"
        Whether to keep the server alive once the main process finishes. When False,
        the server will be terminated when the main process finishes. When True,
        the server will be kept alive and thus block the main process from exiting.
        When `"auto"`, the server will be kept alive if the current execution context
        is not a notebook-like environment.
    port: int, default=None
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
    from skore.project._create import _create
    from skore.project._launch import _launch
    from skore.project._load import _load

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
        _launch(project, keep_alive=keep_alive, port=port, verbose=verbose)

    return project
