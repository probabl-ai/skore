"""Helpers to create, load, and launch projects."""

import atexit
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Union

import joblib
import platformdirs
import psutil

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


def is_server_started(port: int, timeout: float = 10.0, interval: float = 0.1) -> bool:
    """Wait for server to be ready by attempting to connect to the port.

    Parameters
    ----------
    port : int
        The port to check.
    timeout : float, default=10.0
        Maximum time to wait in seconds.
    interval : float, default=0.1
        Time between connection attempts in seconds.

    Returns
    -------
    bool
        True if server is ready, False if timeout occurred.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=interval):
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(interval)
    return False


def start_server_in_subprocess(project: Project, port: int, open_browser: bool):
    """Start the server in a separate process.

    Parameters
    ----------
    project : Project
        The project to associate with the server.
    port : int
        The port to use for the server.
    open_browser : bool
        Whether to open the browser.

    Returns
    -------
    subprocess.Popen
        The process running the server.

    Raises
    ------
    RuntimeError
        If the server failed to start within the timeout period.
    """
    cmd = [
        sys.executable,
        "-m",
        "skore.ui.server",
        "--port",
        str(port),
        "--project-path",
        str(project.path),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if is_server_started(port):
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        return process
    else:
        process.terminate()
        raise RuntimeError("Server failed to start within timeout period")


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


def block_before_cleanup(project: Project, process: subprocess.Popen) -> bool:
    from skore import console

    try:
        console.print("Press Ctrl+C to stop the server...")
        while True:
            if process.poll() is not None:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\nReceived keyboard interrupt, shutting down...")
    finally:
        cleanup_server(project)


def _kill_all_servers():
    """Kill all running servers."""
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
    keep_alive : str or bool, default="auto"
        Whether to keep the server alive once the main process finishes. When False,
        the server will be terminated when the main process finishes. When True,
        the server will be kept alive until interrupted by the user (Ctrl+C).
        When `"auto"`, the server will be kept alive if the current execution context
        is not a notebook-like environment.
    port : int, default=None
        The port to use for the server. If None, a free port will be found starting from
        22140.
    open_browser : bool, default=True
        Whether to open the browser. By default, the browser is opened.
    verbose : bool, default=False
        Whether to print verbose output. By default, no output is printed.
    """
    from skore import console  # avoid circular import

    if port is None:
        port = find_free_port()

    if project._server_info is not None:
        # The project is already attached to a server.
        # We need to check if the server is still running or if we need to restart it.
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

    if keep_alive == "auto":
        keep_alive = is_environment_notebook_like()

    with logger_context(logger, verbose):
        process = start_server_in_subprocess(project, port, open_browser)
        project._server_info = ServerInfo(project, port, process.pid)
        project._server_info.save_pid_file()

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        msg = f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
        console.print(msg)

        if keep_alive:
            atexit.register(cleanup_server, project)
        else:
            atexit.register(block_before_cleanup, project, process)
