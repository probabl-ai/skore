"""Create an HTTP server which will wait for oauth success callback."""

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread
from typing import Callable
from urllib.parse import parse_qs, urlparse

SUCCESS_PAGE = """
<script>window.close();</script>
<p>You can now close this page.</p>
"""


def _get_handler(callback):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            del args

        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", len(SUCCESS_PAGE))
            self.end_headers()
            self.wfile.write(bytes(SUCCESS_PAGE, "utf-8"))

            state_list = params.get("state", [None])
            callback(state_list[0])

    return _Handler


def launch_callback_server(callback: Callable[[str], None]) -> int:
    """Start a temporary HTTP server on a random port to handle callbacks.

    The server automatically shuts down after receiving a callback.

    Parameters
    ----------
    callback : callable
        Function to be called when the server receives a request.
        The callback function should accept one parameter for the state.

    Returns
    -------
    int
        The port number the server is listening on.
    threading.Event
        An event to stop the running server.

    Notes
    -----
    The server runs in a separate thread and will be automatically shut down
    either when the callback is received or when the timeout is reached.
    """
    # We let the OS choose the port to make sure the chosen one is free.
    shutdown_event = threading.Event()

    def callback_wrapper(state):
        callback(state)
        shutdown_event.set()

    server = TCPServer(("", 0), _get_handler(callback_wrapper))
    _, port = server.server_address
    server_thread = Thread(target=server.serve_forever)

    def watchdog():
        shutdown_event.wait()
        server.shutdown()
        server_thread.join()

    watchdog_thread = Thread(target=watchdog)

    server_thread.start()
    watchdog_thread.start()

    return port, shutdown_event
