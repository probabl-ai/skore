"""FastAPI factory used to create the API to interact with stores."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skore.api.routes import ROOT_ROUTER
from skore.project import Project


def create_api_app(project: Project) -> FastAPI:
    """FastAPI factory used to create the API to interact with `stores`."""
    app = FastAPI()
    # Give the app access to the project directory
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
    app.include_router(ROOT_ROUTER)

    return app
