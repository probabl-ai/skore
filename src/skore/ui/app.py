"""FastAPI factory used to create the API to interact with stores."""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from skore.project import Project
from skore.storage.filesystem import FileSystem

from .dependencies import get_static_path
from .report import router as report_router


def create_app(project: Project | None = None) -> FastAPI:
    """FastAPI factory used to create the API to interact with `stores`."""
    app = FastAPI()

    # Give the app access to the project
    if not project:
        project_path = Path.cwd() / "project.skore"
        project_path.mkdir(exist_ok=True)

        filesystem = FileSystem(directory=project_path)
        project = Project(filesystem)

    app.state.project = project

    # Enable CORS support on all routes, for all origins and methods.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers from bottom to top.
    # Include routers always after all routes have been defined/imported.
    router = APIRouter(prefix="/api")
    router.include_router(report_router)

    # Include all sub routers.
    app.include_router(router)

    # Mount frontend from the static directory.
    # Should be after the API routes to avoid shadowing previous routes.
    static_path = get_static_path()
    if static_path.exists():
        app.mount(
            "/",
            StaticFiles(
                directory=static_path,
                html=True,
                follow_symlink=True,
            ),
            name="static",
        )

    return app
