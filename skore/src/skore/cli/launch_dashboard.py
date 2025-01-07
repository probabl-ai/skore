"""Implement the "launch" command."""

import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import FastAPI

from skore.cli import logger
from skore.project import open
from skore.ui.app import create_app
from skore.utils._logger import logger_context


def __launch(
    project_name: Union[str, Path],
    port: int,
    open_browser: bool,
    verbose: bool = False,
):
    """Launch the UI to visualize a project.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.
    port : int
        Port at which to bind the UI server.
    open_browser: bool
        Whether to automatically open a browser tab showing the UI.
    verbose: bool
        Whether to display info logs to the user.
    """
    from skore import console  # avoid circular import

    with logger_context(logger, verbose):
        project = open(project_name)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if open_browser:
                webbrowser.open(f"http://localhost:{port}")
            yield

        app = create_app(project=project, lifespan=lifespan)

        try:
            # TODO: check port is free
            console.rule("[bold cyan]skore-UI[/bold cyan]")
            console.print(
                f"Running skore UI from '{project_name}' at URL http://localhost:{port}"
            )
            uvicorn.run(app, port=port, log_level="error")
        except KeyboardInterrupt:
            console.print("Closing skore UI")
