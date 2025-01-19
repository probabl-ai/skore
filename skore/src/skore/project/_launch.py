"""Helpers to create, load, and launch projects."""

import asyncio
import socket
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from rich.console import Console

from skore.project.project import Project, logger
from skore.utils._logger import logger_context


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


class ServerManager:
    _instance = None
    _executor = None
    _port = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_server(
        self,
        project: Project,
        port: int | None = None,
        open_browser: bool = True,
    ):
        from skore import console

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        if self._executor is not None:
            console.print(f"Server is already running at http://localhost:{self._port}")
            return

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self._port = port
                loop.run_until_complete(
                    run_server(project, port, open_browser, console)
                )
            except KeyboardInterrupt:
                console.print("Closing skore UI")
            finally:
                loop.close()
                console.print(
                    f"Server that was running at http://localhost:{self._port} has "
                    "been closed"
                )

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(run_in_thread)
        self._executor.shutdown(wait=False)


async def run_server(project: Project, port: int, open_browser: bool, console: Console):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if open_browser:  # This was previously hardcoded in _launch
            webbrowser.open(f"http://localhost:{port}")
        yield

    from skore.ui.app import create_app

    app = create_app(project=project, lifespan=lifespan)
    server_config = uvicorn.Config(app, port=port, log_level="error")
    server = uvicorn.Server(server_config)

    console.print(
        f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
    )

    await server.serve()


def launch(
    project: Project,
    port: int | None = None,
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
        server_manager.start_server(project, port, open_browser)
