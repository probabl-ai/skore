"""Login to the skore-HUB."""

from __future__ import annotations

import webbrowser
from datetime import datetime
from time import sleep

from httpx import HTTPError
from rich.align import Align
from rich.panel import Panel

from skore import console
from skore.hub import api
from skore.hub.callback_server import launch_callback_server
from skore.hub.client import AuthenticationError
from skore.hub.token import AuthenticationToken


def login(timeout=600, auto_otp=True):
    """Login to the skore-HUB."""
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

        port = launch_callback_server(callback=callback)
        authorization_url, device_code, user_code = api.get_oauth_device_login(
            success_uri=f"http://localhost:{port}"
        )

        webbrowser.open(authorization_url)

        while access is None or refreshment is None or expires_at is None:
            sleep(0.5)

        return AuthenticationToken(
            access=access, refreshment=refreshment, expires_at=expires_at
        )

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

        # Start polling Skore-Hub, waiting for the token
        start = datetime.now()

        while True:
            try:
                return AuthenticationToken(
                    *api.get_oauth_device_token(device_code=device_code)
                )
            except HTTPError:
                sleep(0.5)

                if (datetime.now() - start).total_seconds() > timeout:
                    raise AuthenticationError("Timeout") from None
