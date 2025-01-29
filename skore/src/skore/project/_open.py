"""Helper to open a project."""

from pathlib import Path
from typing import Literal, Optional, Union

from skore.project.project import Project


def open(
    project_path: Union[str, Path] = "project.skore",
    *,
    if_exists: Optional[Literal["raise", "load"]] = "raise",
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
    project_path: Path-like, optional
        The relative or absolute path of the project, default "project.skore".
    if_exists: Literal["raise", "load"], optional
        Raise an exception if the project already exists, or load it, default "raise".
    serve: bool, optional
        Whether to launch the skore UI, default True.
    keep_alive : Union[str, bool], optional
        Whether to keep the server alive once the main process finishes. When False,
        the server will be terminated when the main process finishes. When True,
        the server will be kept alive and thus block the main process from exiting.
        When `"auto"`, the server will be kept alive if the current execution context
        is not a notebook-like environment. Default "auto".
    port: int, optional
        Port at which to bind the UI server. If ``None``, the server will be bound to
        an available port. Default None.
    verbose : bool, optional
        Whether or not to display info logs to the user, default False.

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
    from skore.project._launch import _launch

    project = Project(project_path, if_exists=if_exists)

    if serve:
        _launch(project, keep_alive=keep_alive, port=port, verbose=verbose)

    return project
