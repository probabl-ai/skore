"""FastAPI factory used to create the dashboard to display stores."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from skore.api import create_api_app
from skore.project import Project
from skore.storage.filesystem import FileSystem


def create_dashboard_app(project: Project | None = None) -> FastAPI:
    if not project:
        directory = Path.cwd() / "project.skore"
        directory.mkdir(exist_ok=True)

        filesystem = FileSystem(directory=directory)
        project = Project(filesystem)

    app = create_api_app(project=project)

    # Mount frontend from the static directory.
    app.mount(
        "/",
        StaticFiles(
            directory=(Path(__file__).parent / "static"),
            html=True,
            follow_symlink=True,
        ),
        name="static",
    )

    return app
