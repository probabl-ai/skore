"""Implement the "quickstart" command."""

from skore import logger
from skore.create_project import ProjectAlreadyExists, create_project
from skore.dashboard.dashboard import __launch


def __quickstart():
    """Quickstart a Skore project.

    Create it if it does not exist, then launch the dashboard.

    Parameters
    ----------
    port : int
        Port at which to bind the UI server.
    """
    directory = "project.skore"
    try:
        create_project(directory=directory)
    except ProjectAlreadyExists:
        logger.info(
            f"Project file '{directory}' already exists. Skipping creation step."
        )

    __launch(
        directory="project.skore",
        port=22140,
        open_browser=True,
    )
