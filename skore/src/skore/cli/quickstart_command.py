"""Implement the "quickstart" command."""

from pathlib import Path
from typing import Optional, Union

from skore.cli import logger
from skore.cli.launch_dashboard import __launch
from skore.exceptions import ProjectAlreadyExistsError
from skore.project import create
from skore.utils._logger import with_logging


@with_logging(logger)
def __quickstart(
    project_name: Union[str, Path],
    working_dir: Optional[Path],
    overwrite: bool,
    port: int,
    open_browser: bool,
    verbose: bool = False,
):
    """Quickstart a Skore project.

    Create it if it does not exist, then launch the web UI.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.
    working_dir : Path or None
        If ``project_name`` is not an absolute path, it will be considered relative to
        ``working_dir``. If `project_name` is an absolute path, ``working_dir`` will
        have no effect. If set to ``None`` (the default), ``working_dir`` will be re-set
        to the current working directory.
    overwrite : bool
        If ``True``, overwrite an existing project with the same name.
        If ``False``, raise an error if a project with the same name already exists.
    port : int
        Port at which to bind the UI server.
    open_browser : bool
        Whether to automatically open a browser tab showing the UI.
    """
    try:
        create(project_name=project_name, working_dir=working_dir, overwrite=overwrite)
    except ProjectAlreadyExistsError:
        logger.info(
            f"Project file '{project_name}' already exists. Skipping creation step."
        )

    __launch(
        project_name=project_name,
        port=port,
        open_browser=open_browser,
    )
