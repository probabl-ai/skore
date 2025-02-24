import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread


def _get_handler(callback):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            del args

        def do_GET(self):
            s = "coucou"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", len(s))
            self.end_headers()
            self.wfile.write(bytes(s, "utf-8"))

            state = None
            callback(state)

    return _Handler


def launch_callback_server(callback, timeout) -> int:
    """Launch a HTTP server that will wait for server success web hook."""
    # We let the OS choose the port to make sure the chosen one is free.
    shutdown_event = threading.Event()

    def callback_wrapper(state):
        callback(state)
        shutdown_event.set()

    server = TCPServer(("", 0), _get_handler(callback_wrapper))
    _, port = server.server_address
    server_thread = Thread(target=server.serve_forever)

    def watchdog():
        shutdown_event.wait(timeout=timeout)
        server.shutdown()
        server_thread.join()

    watchdog_thread = Thread(target=watchdog)

    server_thread.start()
    watchdog_thread.start()

    return port


if __name__ == "__main__":
    from skore import console

    port = launch_callback_server(lambda s: console.print(s), 500)
    console.print(f"Listening for success callback on http://localhost:{port}")
