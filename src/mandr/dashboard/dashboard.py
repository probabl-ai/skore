"""Dashboard to display stores."""

import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import uvicorn

from mandr import logger


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

    def __init__(self, *, port=22140):
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
        try:
            test_server = None
            test_server = HTTPServer(("127.0.0.1", self.port), BaseHTTPRequestHandler)
        except OSError as e:
            if e.errno == 98:
                logger.info(
                    f"Address 127.0.0.1:{self.port} is already in use. "
                    "Check if the dashboard or another service is already running at "
                    "that address."
                )
                return
        finally:
            if test_server is not None:
                test_server.server_close()

        self.thread.start()

        while not self.server.started:
            if not self.thread.is_alive():
                self.close()
                logger.error(
                    "Server failed to start; refer to runtime logs "
                    "for more information."
                )
                return None
            continue

        if open_browser:
            webbrowser.open(f"http://localhost:{self.port}")

        return self

    def close(self):
        """Close the dashboard's activity."""
        self.server.should_exit = True
        self.thread.join()
