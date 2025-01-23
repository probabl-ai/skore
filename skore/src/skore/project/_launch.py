"""Helpers to create, load, and launch projects."""

import asyncio
import atexit
import contextlib
import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
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


class ServerManager:
    _instance = None
    _port = None
    _server_running = False

    def __init__(self):
        self._timestamp = time.time()
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

    async def _run_server_async(self, project, port, open_browser, console):
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
        self._server = uvicorn.Server(server_config)
        console.print(
            f"Running skore UI from '{project.name}' at URL http://localhost:{port}"
        )

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

        def run_server_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._port = port
            self._server_running = True

            try:
                self._loop.run_until_complete(
                    self._run_server_async(project, port, open_browser, console)
                )
            except Exception:
                self._cleanup_server()

        self._server_thread = threading.Thread(target=run_server_loop, daemon=True)
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
