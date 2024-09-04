"""Implement the "launch dashboard" command."""

import os
from pathlib import Path

from mandr.dashboard.dashboard import Dashboard


class ProjectNotFound(Exception):
    """Project was not found."""


def launch_dashboard(project_name: str | Path, port: int, open_browser: bool) -> Path:
    """Launch dashboard to visualize a project.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.
    port : int
        Port at which to bind the UI server.
    open_browser: bool
        Whether to automatically open a browser tab showing the dashboard.

    Returns
    -------
    The project directory path
    """
    if not Path(project_name).exists():
        raise ProjectNotFound(
            f"Project '{project_name}' not found. "
            "Maybe you forget to create it? Please check the file name and try again."
        )

    # FIXME: Passing the project name through environment variables is smelly
    if os.environ.get("MANDR_ROOT") is None:
        os.environ["MANDR_ROOT"] = project_name

    dashboard = Dashboard(port=port)
    dashboard.open(open_browser=open_browser)

    return dashboard, project_name
