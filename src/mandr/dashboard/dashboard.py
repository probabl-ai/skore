"""Dashboard to display stores."""

import threading
import webbrowser

import uvicorn


class Dashboard:
    """Dashboard to display stores.

    .. highlight:: python
    .. code-block:: python

        import contextlib
        with contextlib.closing(Dashboard()):
            ...

    .. highlight:: python
    .. code-block:: python

        dashboard = Dashboard()
        ...
        dashboard.close()
    """

    def __init__(self, *, port=8000, open_browser=True):
        """Initialize the dashboard's activity."""
        configuration = uvicorn.Config(
            app="mandr.dashboard.app:create_dashboard_app",
            port=port,
            log_level="error",
            reload=True,
            factory=True,
        )

        self.server = uvicorn.Server(configuration)
        self.thread = threading.Thread(target=self.server.run)
        self.thread.start()

        while not self.server.started:
            continue

        if open_browser:
            webbrowser.open(f"http://localhost:{port}")

    def close(self):
        """Close the dashboard's activity."""
        self.server.should_exit = True
        self.thread.join()
