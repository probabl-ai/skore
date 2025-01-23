"""Helpers to create, load, and launch projects."""

import asyncio
import atexit
import contextlib
import socket
import threading
import webbrowser
from contextlib import asynccontextmanager
from typing import Union

import uvicorn
from fastapi import FastAPI

from skore.project.project import Project, logger
from skore.utils._environment import is_environment_notebook_like
from skore.utils._logger import logger_context


def find_free_port(min_port: int = 22140, max_attempts: int = 100) -> int:
    """Find first available port starting from min_port.

    Note: Jupyter has the same brutforce way to find a free port.
    see: https://github.com/jupyter/jupyter_core/blob/fa513c1550bbd1ebcc14a4a79eb8c5d95e3e23c9/tests/dotipython_empty/profile_default/ipython_notebook_config.py#L28

    Parameters
    ----------
        min_port : int
            Minimum port number to start searching from.
        max_attempts : int
            Maximum number of ports to try before giving up.

    Returns
    -------
        First available port number >= min_port.

    Raises
    ------
        RuntimeError: If no free port found after max_attempts.
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


class ServerManager:
    """Manages the lifecycle of a server instance.

    Includes starting, stopping, and cleanup.
    Using daemon thread in interactive contexts (vscode & notebook)
    and regular thread in scripts.
    """

    _instance = None
    _port = None
    _server_running = False

    def __init__(self):
        self._loop = None
        self._server_thread = None
        self._server_ready = threading.Event()
        self._cleanup_complete = threading.Event()
        self._server = None
        atexit.register(self._cleanup_server)

    def __del__(self):
        atexit.unregister(self._cleanup_server)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _cleanup_server(self):
        """Cleanup server resources."""
        #
        if not self._server_running:
            self._cleanup_complete.set()
            return

        self._server_running = False
        if self._loop and not self._loop.is_closed():
            with contextlib.suppress(Exception):
                if self._server:
                    # Schedule server shutdown in the event loop
                    async def shutdown():
                        await self._server.shutdown()
                        self._cleanup_complete.set()

                    self._loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(shutdown())
                    )

                for task in asyncio.all_tasks(self._loop):
                    task.cancel()
                self._loop.stop()
                self._loop.close()

            from skore import console

            console.rule("[bold cyan]skore-UI[/bold cyan]")
            console.print(
                f"Server that was running at http://localhost:{self._port} has "
                "been closed"
            )
        self._loop = None
        self._server = None
        self._server_thread = None
        self._server_ready.clear()
        if not self._cleanup_complete.is_set():
            self._cleanup_complete.set()

    async def _run_server_async(
        self,
        project,
        port,
        open_browser,
        console,
        is_running_as_daemon,
    ):
        """Async function to start the server and signal when it's ready."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if open_browser:
                webbrowser.open(f"http://localhost:{port}")
            yield

        from skore.ui.app import create_app

        app = create_app(project=project, lifespan=lifespan)
        server_config = uvicorn.Config(
            app,
            port=port,
            log_level="error",
        )
        self._server = uvicorn.Server(server_config)
        msg = f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
        if not is_running_as_daemon:
            msg += " (Press CTRL+C to quit)"
        console.print(msg)

        # Set ready event when server is about to start
        self._server_ready.set()

        try:
            await self._server.serve()
        except asyncio.CancelledError:
            await self._server.shutdown()
        finally:
            self._server_ready.clear()

    def start_server(
        self,
        project: Project,
        port: Union[int, None] = None,
        open_browser: bool = True,
        timeout: float = 10.0,
    ):
        from skore import console

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        if self._server_running:
            console.print(f"Server is already running at http://localhost:{port}")
            return

        def run_server_loop(is_running_as_daemon):
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._port = port
            self._server_running = True

            try:
                self._loop.run_until_complete(
                    self._run_server_async(
                        project, port, open_browser, console, is_running_as_daemon
                    )
                )
            except Exception:
                self._cleanup_server()

        # if user asked to open the web browser and is running in a notebook like env
        # then start the thread as daemon (it will die when kernel dies)
        # if user asked to open the web browser and is running in a script
        # then start the thread as non daemon user will have to hit ctrl+C to kill it
        is_running_as_daemon = is_environment_notebook_like()
        self._server_thread = threading.Thread(
            target=run_server_loop,
            daemon=is_running_as_daemon,
            args=(is_running_as_daemon,),
        )
        self._server_thread.start()

        # Wait for the server to be ready
        if not self._server_ready.wait(timeout):
            raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def shutdown(self):
        """Shutdown the server and cleanup resources."""
        if not self._server_running:
            return

        if self._loop and not self._loop.is_closed():
            with contextlib.suppress(Exception):
                self._loop.call_soon_threadsafe(self._loop.stop)

        self._cleanup_server()


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

    with logger_context(logger, verbose):
        server_manager = ServerManager.get_instance()
        project._server_manager = server_manager
        server_manager.start_server(project, port, open_browser)
