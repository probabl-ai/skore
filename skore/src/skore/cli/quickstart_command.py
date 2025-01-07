"""Implement the "quickstart" command."""

from pathlib import Path
from typing import Union

from skore.cli import logger
from skore.cli.launch_dashboard import __launch
from skore.project.create import _create
from skore.utils._logger import logger_context


def __quickstart(
    project_name: Union[str, Path],
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
    overwrite : bool
        If ``True``, overwrite an existing project with the same name.
        If ``False``, simply warn that a project already exists.
    port : int
        Port at which to bind the UI server.
    open_browser : bool
        Whether to automatically open a browser tab showing the UI.
    verbose : bool
        Whether to increase logging verbosity.
    """
    with logger_context(logger, verbose):
        try:
            _create(
                project_name=project_name,
                overwrite=overwrite,
                verbose=verbose,
            )
        except FileExistsError:
            logger.info(
                f"Project file '{project_name}' already exists. Skipping creation step."
            )

        path = Path(project_name)

        __launch(
            project_name=path,
            port=port,
            open_browser=open_browser,
            verbose=verbose,
        )
