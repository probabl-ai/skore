"""Dashboard to display stores."""

import threading
import webbrowser

import uvicorn


class Dashboard:
    """Dashboard to display stores.

    .. highlight:: python
    .. code-block:: python

        import contextlib
        dashboard = Dashboard()
        with contextlib.closing(dashboard.open()):
            ...

    .. highlight:: python
    .. code-block:: python

        dashboard = Dashboard()
        dashboard.open()
        ...
        dashboard.close()
    """

    def __init__(self, *, port=8000):
        configuration = uvicorn.Config(
            app="mandr.dashboard.app:create_dashboard_app",
            port=port,
            log_level="error",
            reload=True,
            factory=True,
        )

        self.port = port
        self.server = uvicorn.Server(configuration)
        self.thread = threading.Thread(target=self.server.run)

    def open(self, *, open_browser=True):
        """Open the dashboard's activity."""
        self.thread.start()

        while not self.server.started:
            if not self.thread.is_alive():
                self.close()
                raise RuntimeError(
                    "Server failed to start; refer to runtime logs "
                    "for more information."
                )
            continue

        if open_browser:
            webbrowser.open(f"http://localhost:{self.port}")

        return self

    def close(self):
        """Close the dashboard's activity."""
        self.server.should_exit = True
        self.thread.join()
