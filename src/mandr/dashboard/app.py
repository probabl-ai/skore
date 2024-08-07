"""FastAPI factory used to create the dashboard to display stores."""

import asyncio
import webbrowser
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from mandr.api import create_api_app


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


def serve_dashboard_app(port=8000):
    """Serve the dashboard to display stores."""

    async def server_coroutine():
        server = uvicorn.Server(
            uvicorn.Config(
                app="mandr.dashboard.app:create_dashboard_app",
                port=port,
                log_level="error",
                reload=True,
                factory=True,
            )
        )

        await server.serve()

    async def open_browser_after_delay():
        await asyncio.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

    asyncio.create_task(server_coroutine())
    asyncio.create_task(open_browser_after_delay())
