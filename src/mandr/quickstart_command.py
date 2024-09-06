"""Implement the "quickstart" command."""

from mandr import logger
from mandr.create_project import ProjectAlreadyExists, create_project
from mandr.dashboard.dashboard import __launch


def __quickstart():
    """Quickstart a Mandr project.

    Create it if it does not exist, then launch the dashboard.

    Parameters
    ----------
    port : int
        Port at which to bind the UI server.
    """
    project_name = "project.mandr"
    try:
        create_project(project_name=project_name)
    except ProjectAlreadyExists:
        logger.info(
            f"Project file '{project_name}' already exists. Skipping creation step."
        )

    __launch(
        project_name="project.mandr",
        port=22140,
        open_browser=True,
    )
