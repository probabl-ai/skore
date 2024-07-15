"""A web dashboard that displays user's mandrs."""

import asyncio
import webbrowser

import uvicorn


def launch_dashboard(port=8000):
    """Launch a dashboard."""

    async def server_coroutine():
        config = uvicorn.Config(
            "mandr.dashboard.webapp:app", port=port, log_level="error", reload=True
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def open_browser_after_delay():
        await asyncio.sleep(1)
        webbrowser.open_new_tab(f"http://localhost:{port}")

    asyncio.create_task(server_coroutine())
    asyncio.create_task(open_browser_after_delay())
