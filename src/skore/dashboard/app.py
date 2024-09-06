"""FastAPI factory used to create the dashboard to display stores."""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from skore.api import create_api_app


def create_dashboard_app():
    """FastAPI factory used to create the dashboard to display stores."""
    app = create_api_app()

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
