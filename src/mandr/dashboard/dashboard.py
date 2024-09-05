"""Implement the "launch" command."""

import os
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from mandr import logger


class ProjectNotFound(Exception):
    """Project was not found."""

    project_path: Path


def __open_browser(port: int):
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}")


def launch_dashboard(project_name: str | Path, port: int, open_browser: bool):
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
    A tuple with the dashboard and the project directory path if succeeded,
    None if failed
    """
    if Path(project_name).exists():
        pass
    elif Path(project_name + ".mandr").exists():
        project_name = project_name + ".mandr"
    else:
        raise ProjectNotFound(
            f"Project '{project_name}' not found. "
            "Maybe you forget to create it? Please check the file name and try again."
        )

    # FIXME: Passing the project name through environment variables is smelly
    if os.environ.get("MANDR_ROOT") is None:
        os.environ["MANDR_ROOT"] = project_name

    logger.info(
        f"Running dashboard for project file '{project_name}' at URL http://localhost:{port}"
    )

    if open_browser:
        threading.Thread(target=lambda: __open_browser(port=port)).start()

    # TODO: Check beforehand that port is not already bound
    config = uvicorn.Config(
        app="mandr.dashboard.app:create_dashboard_app",
        port=port,
        log_level="error",
        factory=True,
    )
    server = uvicorn.Server(config=config)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Closing dashboard")
