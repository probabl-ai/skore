"""Login to the skore-HUB."""

from __future__ import annotations

import webbrowser
from datetime import datetime, timezone
from time import sleep

from httpx import HTTPError
from rich.align import Align
from rich.panel import Panel

from skore import console
from skore.hub.authentication.otp_server import OTPServer
from skore.hub.authentication.token import Token
from skore.hub.client import api
from skore.hub.client.client import AuthenticationError


def refresh():
    """Attempt to refresh an existing token.

    Returns
    -------
    tuple
        A tuple containing:
        - token: Token
            The token valid or not
        - is_valid: bool
            `True` if the (potentially) refreshed token is valid
            `False` otherwise.
    """
    token = Token()
    if not token.valid:
        return False, token

    now = datetime.now(timezone.utc)
    if token.expires_at <= now:
        # we have a token but it's expired
        # try to refresh it
        try:
            token.refresh()
            return True, token
        except HTTPError:
            return False, token

    # we have a token and it is not expired
    return True, token


def login(timeout=600, auto_otp=True, port=0):
    """Login to the skore-HUB."""
    is_valid, token = refresh()
    if is_valid:
        return token
    if auto_otp:
        access = None
        refreshment = None
        expires_at = None

        def callback(state):
            nonlocal access
            nonlocal refreshment
            nonlocal expires_at

            api.post_oauth_device_callback(state=state, user_code=user_code)
            access, refreshment, expires_at = api.get_oauth_device_token(
                device_code=device_code
            )

        server = OTPServer(callback=callback).start(port=port)
        authorization_url, device_code, user_code = api.get_oauth_device_login(
            success_uri=f"http://localhost:{server.port}"
        )

        webbrowser.open(authorization_url)
        start = datetime.now()

        try:
            while access is None or refreshment is None or expires_at is None:
                if (datetime.now() - start).total_seconds() > timeout:
                    raise AuthenticationError("Timeout") from None

            sleep(0.5)
        finally:
            server.stop()

        return Token(access=access, refreshment=refreshment, expires_at=expires_at)

    else:
        # Request a new authorization URL
        authorization_url, device_code, user_code = api.get_oauth_device_login()

        # Display authentication info to the user
        console.print(
            "\n"
            "🌍 Opening your default browser to start the authentication process.\n"
            "❔ If your browser did not open visit our "
            f"[link={authorization_url}]authentication page[/link].\n"
        )
        console.print(
            Panel(
                Align(f"[bold]{user_code}[/bold]", align="center"),
                title="[cyan]Your unique code is[/cyan]",
                border_style="orange1",
                expand=False,
                padding=1,
                title_align="center",
            )
        )
        # Open the default browser
        webbrowser.open(authorization_url)
        start = datetime.now()

        # Start polling Skore-Hub, waiting for the token
        while True:
            try:
                return Token(*api.get_oauth_device_token(device_code=device_code))
            except HTTPError:
                sleep(0.5)

                if (datetime.now() - start).total_seconds() > timeout:
                    raise AuthenticationError("Timeout") from None
