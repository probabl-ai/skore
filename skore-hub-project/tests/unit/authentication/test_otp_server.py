from http import HTTPStatus

import httpx
import pytest
from skore_hub_project.authentication.otp_server import OTPServer


def test_otp_server_return_error_on_non_index_route():
    server = OTPServer(lambda: ...)

    try:
        server.start()

        with httpx.Client() as client:
            r = client.get(f"http://localhost:{server.port}/favicon.ico")
            assert r.status_code == HTTPStatus.NO_CONTENT
    finally:
        server.stop()


def test_otp_server_port_raise_if_not_started():
    server = OTPServer(lambda: ...)

    with pytest.raises(SystemError):
        server.port  # noqa: B018
