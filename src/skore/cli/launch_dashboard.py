"""Implement the "launch" command."""

import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from skore.cli import logger
from skore.project import load
from skore.ui.app import create_app


class ProjectNotFound(Exception):
    """Project was not found."""

    project_path: Path


def __open_browser(port: int):
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}")


def __launch(project_name: str | Path, port: int, open_browser: bool):
    """Launch the UI to visualize a project.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.
    port : int
        Port at which to bind the UI server.
    open_browser: bool
        Whether to automatically open a browser tab showing the UI.
    """
    project = load(project_name)
    app = create_app(project=project)

    if open_browser:
        threading.Thread(target=lambda: __open_browser(port=port)).start()

    try:
        # TODO: check port is free
        logger.info(
            f"Running skore UI from '{project_name}' at URL http://localhost:{port}"
        )
        uvicorn.run(app, port=port, log_level="error")
    except KeyboardInterrupt:
        logger.info("Closing skore UI")
