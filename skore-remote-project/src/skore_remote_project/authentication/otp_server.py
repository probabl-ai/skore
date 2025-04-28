"""Create an HTTP server which will wait for OTP success callback."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread
from typing import Callable
from urllib.parse import parse_qs, urlparse

SUCCESS_PAGE = b"<p>You can now close this page.</p>"


def _get_handler(callback):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            """Disable logging."""

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path != "/":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", 0)
                self.end_headers()
                self.wfile.write(b"")
            else:
                (state,) = parse_qs(parsed.query).get("state", [None])
                callback(state)

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-type", "text/html")
                self.send_header("Content-Length", len(SUCCESS_PAGE))
                self.end_headers()
                self.wfile.write(SUCCESS_PAGE)

    return _Handler


class OTPServer:
    """HTTP server to handle OTP success callbacks.

    Parameters
    ----------
    callback : callable
        Function to be called when the server receives a request.
        The callback function should accept one parameter for the state.

    Attributes
    ----------
    port
        The port number the server is listening on.
    started
        The server is started, or not.
    """

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback

    def start(self, port=0):
        """Start the server.

        Parameters
        ----------
        port : int, default 0
            The port number the server is listening on.
            By default, we let the system choose a free port.
        """
        if not self.started:
            self.__tcp = TCPServer(("", port), _get_handler(self.callback))
            self.__thread = Thread(target=self.__tcp.serve_forever)
            self.__thread.start()

        return self

    @property
    def port(self):
        """Property indicating the port number the server is listening on."""
        try:
            return self.__tcp.server_address[1]
        except AttributeError:
            raise SystemError("Server not started") from None

    @property
    def started(self):
        """Property indicating whether the server is started or not."""
        return hasattr(self, "_OTPServer__tcp")

    def stop(self):
        """Stop the server."""
        if self.started:
            # Stop server and thread
            self.__tcp.shutdown()
            self.__thread.join()

            # Delete start related attributes
            del self.__tcp
            del self.__thread
