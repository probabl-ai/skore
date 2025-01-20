"""Helpers to create, load, and launch projects."""

import asyncio
import contextlib
import socket
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Union

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
    _server_running = False

    def __init__(self):
        self._timestamp = time.time()
        self._loop = None
        self._server_task = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_server(
        self,
        project: Project,
        port: Union[int, None] = None,
        open_browser: bool = True,
    ):
        from skore import console

        console.rule("[bold cyan]skore-UI[/bold cyan]")
        if self._executor is not None and self._server_running:
            console.print(f"Server is already running at http://localhost:{port}")
            return

        def run_in_thread():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._port = port
                self._server_running = True
                self._loop.run_until_complete(
                    run_server(project, port, open_browser, console)
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                self._server_running = False
                try:
                    tasks = asyncio.all_tasks(self._loop)
                    for task in tasks:
                        task.cancel()
                    self._loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
                    self._loop.close()
                except Exception:
                    pass
                console.rule("[bold cyan]skore-UI[/bold cyan]")
                console.print(
                    f"Server that was running at http://localhost:{self._port} has "
                    "been closed"
                )

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(run_in_thread)
        project._server_manager = self
        self._executor.shutdown(wait=False)

    def shutdown(self):
        """Shutdown the server and cleanup resources."""
        if self._executor is not None and self._server_running:
            if self._loop is not None:

                async def stop():
                    tasks = asyncio.all_tasks(self._loop)
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                try:
                    future = asyncio.run_coroutine_threadsafe(stop(), self._loop)
                    future.result(timeout=0.1)
                except Exception:
                    pass

            self._server_running = False
            with contextlib.suppress(Exception):
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._loop = None


async def run_server(project: Project, port: int, open_browser: bool, console: Console):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        try:
            yield
        except asyncio.CancelledError:
            project._server_manager = None

    from skore.ui.app import create_app

    app = create_app(project=project, lifespan=lifespan)
    server_config = uvicorn.Config(
        app,
        port=port,
        log_level="error",
        log_config={
            "version": 1,
            "disable_existing_loggers": True,
            "handlers": {
                "default": {
                    "class": "logging.NullHandler",
                }
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "ERROR"},
                "uvicorn.error": {"handlers": ["default"], "level": "ERROR"},
            },
        },
    )
    server = uvicorn.Server(server_config)
    console.print(
        f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
    )

    try:
        await server.serve()
    except asyncio.CancelledError:
        await server.shutdown()


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
        server_manager.start_server(project, port, open_browser)
