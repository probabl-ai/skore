"""Helpers to create, load, and launch projects."""

import atexit
import contextlib
import json
import multiprocessing
import os
import socket
import webbrowser
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import FastAPI

from skore.project.project import Project, logger
from skore.utils._logger import logger_context


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def run_server(project: Project, port: int, open_browser: bool):
    """Run the server in a separate process."""
    from skore import console
    from skore.ui.app import create_app

    async def lifespan(app: FastAPI):
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        yield

    app = create_app(project=project, lifespan=lifespan)
    console.rule("[bold cyan]skore-UI[/bold cyan]")
    console.print(
        f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
    )

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")


def get_server_info_path(project: Project) -> Path:
    """Get path to the server info file for this project."""
    # return Path(project.path) / ".server_info.json"
    return Path("/tmp") / "skore_server_info.json"


def save_server_info(project: Project, port: int, pid: int):
    """Save server information to disk."""
    info = {"port": port, "pid": pid}
    with open(get_server_info_path(project), "w") as f:
        json.dump(info, f)


def load_server_info(project: Project) -> Union[dict, None]:
    """Load server information from disk if it exists."""
    try:
        with open(get_server_info_path(project)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cleanup_server(project: Project):
    """Cleanup server resources."""
    info = load_server_info(project)
    if info is None:
        return

    try:
        # Try to terminate the process
        os.kill(info["pid"], 15)  # SIGTERM

        from skore import console

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        console.print(
            f"Server that was running at http://localhost:{info['port']} has "
            "been closed"
        )
    except ProcessLookupError:
        pass  # Process already terminated
    finally:
        # Always clean up the info file
        with contextlib.suppress(FileNotFoundError):
            os.remove(get_server_info_path(project))


def _launch(
    project: Project,
    port: Union[int, None] = None,
    open_browser: bool = True,
    verbose: bool = False,
):
    """Launch the UI to visualize a project.

    Parameters
    ----------
    project : Project
        The project to be launched.
    port : int, default=None
        Port at which to bind the UI server. If ``None``, the server will be bound to
        an available port.
    open_browser: bool, default=True
        Whether to automatically open a browser tab showing the UI.
    verbose: bool, default=False
        Whether to display info logs to the user.
    """
    if port is None:
        port = find_free_port()

    # Check if server is already running
    info = load_server_info(project)
    if info is not None:
        try:
            # Check if process is still alive
            os.kill(info["pid"], 0)
            from skore import console

            console.print(
                f"Server is already running at http://localhost:{info['port']}"
            )
            return
        except ProcessLookupError:
            # Process is dead, clean up the file
            cleanup_server(project)

    with logger_context(logger, verbose):
        process = multiprocessing.Process(
            target=run_server, args=(project, port, open_browser), daemon=True
        )
        process.start()

        # Save server info to disk
        save_server_info(project, port, process.pid)

        # Register cleanup on exit
        atexit.register(cleanup_server, project)
