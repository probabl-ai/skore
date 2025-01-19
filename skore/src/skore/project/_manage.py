"""Helpers to create, load, and launch projects."""

import asyncio
import re
import shutil
import socket
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI
from rich.console import Console

from skore.exceptions import (
    InvalidProjectNameError,
    ProjectCreationError,
    ProjectLoadError,
    ProjectPermissionError,
)
from skore.item import ItemRepository
from skore.persistence.disk_cache_storage import DirectoryDoesNotExist, DiskCacheStorage
from skore.project.project import Project, logger
from skore.utils._logger import logger_context
from skore.view.view import View
from skore.view.view_repository import ViewRepository


def _load(project_name: Union[str, Path]) -> Project:
    """Load an existing Project given a project name or path.

    Transforms a project name to a directory path as follows:
    - Resolves relative path to current working directory,
    - Checks that the file ends with the ".skore" extension,
    - If not provided, it will be automatically appended,
    - If project name is an absolute path, keeps that path.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path.

    Returns
    -------
    Project
        The loaded Project instance.
    """
    path = Path(project_name).resolve()

    if path.suffix != ".skore":
        path = path.parent / (path.name + ".skore")

    if not Path(path).exists():
        raise FileNotFoundError(f"Project '{path}' does not exist: did you create it?")

    try:
        # FIXME: Should those hardcoded strings be factorized somewhere ?
        item_storage = DiskCacheStorage(directory=Path(path) / "items")
        item_repository = ItemRepository(storage=item_storage)
        view_storage = DiskCacheStorage(directory=Path(path) / "views")
        view_repository = ViewRepository(storage=view_storage)
        project = Project(
            name=path.name,
            item_repository=item_repository,
            view_repository=view_repository,
        )
    except DirectoryDoesNotExist as e:
        missing_directory = e.args[0].split()[1]
        raise ProjectLoadError(
            f"Project '{path}' is corrupted: "
            f"directory '{missing_directory}' should exist. "
            "Consider re-creating the project."
        ) from e

    return project


def _validate_project_name(project_name: str) -> tuple[bool, Optional[Exception]]:
    """Validate the project name (the part before ".skore").

    Returns `(True, None)` if validation succeeded and `(False, Exception(...))`
    otherwise.
    """
    # The project name (including the .skore extension) must be between 5 and 255
    # characters long.
    # FIXME: On Linux the OS already checks filename lengths
    if len(project_name) + len(".skore") > 255:
        return False, InvalidProjectNameError(
            "Project name length cannot exceed 255 characters."
        )

    # Reserved Names: The following names are reserved and cannot be used:
    # CON, PRN, AUX, NUL
    # COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9
    # LPT1, LPT2, LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, LPT9
    reserved_patterns = "|".join(["CON", "PRN", "AUX", "NUL", r"COM\d+", r"LPT\d+"])
    if re.fullmatch(f"^({reserved_patterns})$", project_name):
        return False, InvalidProjectNameError(
            "Project name must not be a reserved OS filename."
        )

    # Allowed Characters:
    # Alphanumeric characters (a-z, A-Z, 0-9)
    # Underscore (_)
    # Hyphen (-)
    # Starting Character: The project name must start with an alphanumeric character.
    if not re.fullmatch(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", project_name):
        return False, InvalidProjectNameError(
            "Project name must contain only alphanumeric characters, '_' and '-'."
        )

    # Case Sensitivity: File names are case-insensitive on Windows and case-sensitive
    # on Unix-based systems. The CLI should warn users about potential case conflicts
    # on Unix systems.

    return True, None


def _create(
    project_name: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = False,
) -> Project:
    """Create a project file named according to ``project_name``.

    Parameters
    ----------
    project_name : Path-like
        Name of the project to be created, or a relative or absolute path. If relative,
        will be interpreted as relative to the current working directory.
    overwrite : bool, default=False
        If ``True``, overwrite an existing project with the same name.
        If ``False``, raise an error if a project with the same name already exists.
    verbose : bool, default=False
        Whether or not to display info logs to the user.

    Returns
    -------
    Project
        The created project
    """
    from skore import console  # avoid circular import

    with logger_context(logger, verbose):
        project_path = Path(project_name)

        # Remove trailing ".skore" if it exists to check the name is valid
        checked_project_name: str = project_path.name.split(".skore")[0]

        validation_passed, validation_error = _validate_project_name(
            checked_project_name
        )
        if not validation_passed:
            raise ProjectCreationError(
                f"Unable to create project file '{project_path}'."
            ) from validation_error

        # The file must end with the ".skore" extension.
        # If not provided, it will be automatically appended.
        # If project name is an absolute path, we keep that path

        project_directory = project_path.with_name(checked_project_name + ".skore")

        if project_directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Unable to create project file '{project_directory}' because a "
                    "file with that name already exists. Please choose a different "
                    "name or use the --overwrite flag with the CLI or overwrite=True "
                    "with the API."
                )
            shutil.rmtree(project_directory)

        try:
            project_directory.mkdir(parents=True)
        except PermissionError as e:
            raise ProjectPermissionError(
                f"Unable to create project file '{project_directory}'. "
                "Please check your permissions for the current directory."
            ) from e
        except Exception as e:
            raise ProjectCreationError(
                f"Unable to create project file '{project_directory}'."
            ) from e

        # Once the main project directory has been created, created the nested
        # directories

        items_dir = project_directory / "items"
        try:
            items_dir.mkdir()
        except Exception as e:
            raise ProjectCreationError(
                f"Unable to create project file '{items_dir}'."
            ) from e

        views_dir = project_directory / "views"
        try:
            views_dir.mkdir()
        except Exception as e:
            raise ProjectCreationError(
                f"Unable to create project file '{views_dir}'."
            ) from e

        p = _load(project_directory)
        p.put_view("default", View(layout=[]))

        console.rule("[bold cyan]skore[/bold cyan]")
        console.print(f"Project file '{project_directory}' was successfully created.")
        return p


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


def _launch(
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
