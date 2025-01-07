"""FastAPI factory used to create the API to interact with stores."""

import sys
from typing import Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import Lifespan

from skore.project import Project, open
from skore.ui.dependencies import get_static_path
from skore.ui.project_routes import router as project_router


def create_app(
    project: Optional[Project] = None, lifespan: Optional[Lifespan] = None
) -> FastAPI:
    """FastAPI factory used to create the API to interact with `stores`."""
    app = FastAPI(lifespan=lifespan)

    # Give the app access to the project
    if not project:
        project = open("project.skore")

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
    router.include_router(project_router)

    # Include all sub routers.
    app.include_router(router)

    # Mount skore-ui from the static directory.
    # Should be after the API routes to avoid shadowing previous routes.
    static_path = get_static_path()
    if static_path.exists():
        # The mimetypes module may fail to set the
        # correct MIME type for javascript files.
        # More info on this here: https://github.com/encode/starlette/issues/829
        # So force it...
        if "win" in sys.platform:
            import mimetypes

            mimetypes.add_type("application/javascript", ".js")

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
