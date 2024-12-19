"""Implement the "quickstart" command."""

from pathlib import Path
from typing import Union

from skore.cli import logger
from skore.cli.launch_dashboard import __launch
from skore.project import create
from skore.utils._logger import logger_context


def __quickstart(
    project_name: Union[str, Path],
    working_dir: Union[Path, None],
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
    verbose : bool
        Whether to increase logging verbosity.
    """
    with logger_context(logger, verbose):
        try:
            create(
                project_name=project_name,
                working_dir=working_dir,
                overwrite=overwrite,
                verbose=verbose,
            )
        except FileExistsError:
            logger.info(
                f"Project file '{project_name}' already exists. Skipping creation step."
            )

        path = (
            Path(project_name)
            if working_dir is None
            else Path(working_dir, project_name)
        )

        __launch(
            project_name=path,
            port=port,
            open_browser=open_browser,
            verbose=verbose,
        )
