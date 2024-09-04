"""Test CLI properly starts serving the app."""

import subprocess
from contextlib import contextmanager
from time import monotonic

import httpx


@contextmanager
def terminate(process):
    """Terminate `process` upon exit.

    Adapted from [contextlib.closing](https://docs.python.org/3/library/contextlib.html#contextlib.closing).
    """
    try:
        yield process
    finally:
        process.terminate()


def test_launch():
    """If the `launch` subcommand is called, the app is properly served at the
    specified port."""

    PORT = 22140
    # Limit time of test to make sure it doesn't run forever
    MAX_TIME = 10  # seconds

    with terminate(
        subprocess.Popen(
            f"python -m mandr launch --no-open-browser --port {PORT}".split()
        )
    ):
        start = monotonic()
        while monotonic() - start < MAX_TIME:
            try:
                response = httpx.get(f"http://localhost:{PORT}")
            except httpx.ConnectError:
                # TODO: Deal with the case where the test failed because the command
                # exited with an error
                continue

            assert response.is_success
            assert b"<!DOCTYPE html>" in response.content
            assert b"<title>:mandr.</title>" in response.content
            return

        raise AssertionError(
            "The dashboard took too long to start; maybe the command failed?"
        )
