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

import psutil
import uvicorn
from fastapi import FastAPI

from skore.project.project import Project, logger
from skore.utils._logger import logger_context


def find_free_port(min_port: int = 22140, max_attempts: int = 100) -> int:
    """Find first available port starting from min_port.

    Note: Jupyter has the same brute force way to find a free port.
    see: https://github.com/jupyter/jupyter_core/blob/fa513c1550bbd1ebcc14a4a79eb8c5d95e3e23c9/tests/dotipython_empty/profile_default/ipython_notebook_config.py#L28

    Parameters
    ----------
    min_port : int, default=22140
        Minimum port number to start searching from.
    max_attempts : int, default=100
        Maximum number of ports to try before giving up.

    Returns
    -------
    int
        First available port `number >= min_port`.

    Raises
    ------
    RuntimeError
        If no free port found after `max_attempts`.
    """
    for port in range(min_port, min_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                s.listen(1)
                return s.getsockname()[1]
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find free port after {max_attempts}"
        f"attempts starting from {min_port}"
    )


def get_server_info_path(project: Project) -> Path:
    """Get path to the server info file for this project."""
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


def run_server(
    project: Project, port: int, open_browser: bool, ready_event: multiprocessing.Event
):
    """Run the server in a separate process.

    Parameters
    ----------
    project : Project
        The project to launch.
    port : int
        The port to use for the server.
    open_browser : bool
        Whether to open the browser.
    ready_event : multiprocessing.Event
        Event to signal that the server is ready.
    """
    from skore.ui.app import create_app

    async def lifespan(app: FastAPI):
        ready_event.set()  # Signal that the server is ready
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        yield

    app = create_app(project=project, lifespan=lifespan)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")


def cleanup_server(project: Project, timeout: float = 5.0) -> bool:
    """Cleanup server resources and wait for termination.

    Parameters
    ----------
    project : Project
        The project instance.
    timeout : float, default=5.0
        Maximum time to wait for the process to terminate in seconds.

    Returns
    -------
    bool
        True if the server was successfully terminated, False if timeout occurred
        or server wasn't running.
    """
    info = load_server_info(project)
    if info is None:
        return False

    try:
        process = psutil.Process(info["pid"])
        process.terminate()

        try:
            process.wait(timeout=timeout)
            success = True
        except psutil.TimeoutExpired:
            process.kill()
            success = False

        from skore import console

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        status = "gracefully" if success else "forcefully"
        console.print(
            f"Server that was running at http://localhost:{info['port']} has "
            f"been closed {status}"
        )

    except psutil.NoSuchProcess:
        pass  # Process already terminated
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(get_server_info_path(project))

    return True


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
            The project to launch.
        port : int, optional
            The port to use for the server, by default None.
        open_browser : bool, optional
            Whether to open the browser, by default True.
        verbose : bool, optional
            Whether to print verbose output, by default False.
    """
    from skore import console  # avoid circular import

    if port is None:
        port = find_free_port()

    info = load_server_info(project)
    if info is not None:
        try:
            os.kill(info["pid"], 0)

            console.rule("[bold cyan]skore-UI[/bold cyan]")
            console.print(
                f"Server is already running at http://localhost:{info['port']}"
            )
            return
        except ProcessLookupError:  # zombie process
            cleanup_server(project)

    ready_event = multiprocessing.Event()

    with logger_context(logger, verbose):
        process = multiprocessing.Process(
            target=run_server,
            args=(project, port, open_browser, ready_event),
            daemon=True,
        )
        process.start()
        save_server_info(project, port, process.pid)
        ready_event.wait()  # wait for server to been started

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        console.print(
            f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
        )

        atexit.register(cleanup_server, project)
