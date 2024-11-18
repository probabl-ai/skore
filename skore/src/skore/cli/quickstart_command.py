"""Implement the "quickstart" command."""

from skore.cli import logger
from skore.cli.launch_dashboard import __launch
from skore.exceptions import ProjectAlreadyExistsError
from skore.project import create


def __quickstart():
    """Quickstart a Skore project.

    Create it if it does not exist, then launch the web UI.

    Parameters
    ----------
    port : int
        Port at which to bind the UI server.
    """
    project_name = "project.skore"

    try:
        create(project_name=project_name)
    except ProjectAlreadyExistsError:
        logger.info(
            f"Project file '{project_name}' already exists. Skipping creation step."
        )

    __launch(
        project_name="project.skore",
        port=22140,
        open_browser=True,
    )
