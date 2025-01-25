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

import joblib
import platformdirs
import psutil
import uvicorn
from fastapi import FastAPI

from skore.project.project import Project, logger
from skore.utils._environment import is_environment_notebook_like
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
        f"Could not find free port after {max_attempts} "
        f"attempts starting from {min_port}"
    )


class ServerInfo:
    """Server information.

    Parameters
    ----------
    project : Project
        The project to associate with the server.
    port : int
        The port to use for the server.
    pid : int
        The PID of the server process.

    Attributes
    ----------
    port : int
        The port to use for the server.
    pid : int
        The PID of the server process.
    pid_file : Path
        The path to the PID file.
    """

    @staticmethod
    def _get_pid_file_path(project: Project) -> Path:
        """Get the path to the PID file."""
        if project.path is not None:
            project_identifier = joblib.hash(str(project.path), hash_name="sha1")
        else:
            project_identifier = joblib.hash(project.name, hash_name="sha1")

        state_path = platformdirs.user_state_path(appname="skore")
        return state_path / f"skore-server-{project_identifier}.json"

    def __init__(self, project: Project, port: int, pid: int):
        self.port = port
        self.pid = pid
        self.pid_file = self._get_pid_file_path(project)

    @classmethod
    def rejoin(cls, project: Project):
        """Rejoin a project to a server.

        Parameters
        ----------
        project : Project
            The project to associate with the server.

        Returns
        -------
        ServerInfo
            The server information.
        """
        pid_file = cls._get_pid_file_path(project)
        if not pid_file.exists():
            return

        info = json.load(pid_file.open())
        return cls(project, info["port"], info["pid"])

    def save_pid_file(self):
        """Save server information to disk."""
        info = {"port": self.port, "pid": self.pid}
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, "w") as f:
            json.dump(info, f)

    def delete_pid_file(self):
        """Delete server information from disk."""
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.pid_file)

    def load_pid_file(self) -> Union[dict, None]:
        """Load server information from disk if it exists."""
        try:
            with open(self.pid_file) as f:
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
            webbrowser.open(f"http://localhost:{port}")  # pragma: no cover
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
    if project._server_info is None:
        return False

    try:
        process = psutil.Process(project._server_info.pid)
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
            f"Server that was running at http://localhost:{project._server_info.port} "
            f"has been closed {status}"
        )

    except psutil.NoSuchProcess:
        pass  # Process already terminated
    finally:
        project._server_info.delete_pid_file()
        project._server_info = None

    return True


def _kill_all_servers():
    state_path = platformdirs.user_state_path(appname="skore")
    for pid_file in state_path.glob("skore-server-*.json"):
        try:
            pid_info = json.load(pid_file.open())
            try:
                process = psutil.Process(pid_info["pid"])
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except psutil.TimeoutExpired:
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        finally:
            pid_file.unlink()


def _launch(
    project: Project,
    keep_alive: Union[str, bool] = "auto",
    port: Union[int, None] = None,
    open_browser: bool = True,
    verbose: bool = False,
):
    """Launch the UI to visualize a project.

    Parameters
    ----------
    project : Project
        The project to launch.
    keep_alive : Union[str, bool], default="auto"
        Whether to keep the server alive once the main process finishes. When False,
        the server will be terminated when the main process finishes. When True,
        the server will be kept alive and thus block the main process from exiting.
        When `"auto"`, the server will be kept alive if the current execution context
        is not a notebook-like environment.
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

    if project._server_info is not None:
        try:
            os.kill(project._server_info.pid, 0)

            console.rule("[bold cyan]skore-UI[/bold cyan]")
            console.print(
                "Server is already running at "
                f"http://localhost:{project._server_info.port}"
            )
            return
        except ProcessLookupError:  # zombie process
            cleanup_server(project)

    ready_event = multiprocessing.Event()

    daemon = is_environment_notebook_like() if keep_alive == "auto" else not keep_alive

    with logger_context(logger, verbose):
        process = multiprocessing.Process(
            target=run_server,
            args=(project, port, open_browser, ready_event),
            daemon=daemon,
        )
        process.start()
        project._server_info = ServerInfo(project, port, process.pid)
        project._server_info.save_pid_file()
        ready_event.wait()  # wait for server to been started

        console.rule("[bold cyan]skore-UI[/bold cyan]")

        msg = f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
        if not daemon:
            msg += " (Press CTRL+C to quit)"
        console.print(msg)

        if not daemon:
            try:
                process.join()
            except KeyboardInterrupt:
                cleanup_server(project)
        else:
            atexit.register(cleanup_server, project)
