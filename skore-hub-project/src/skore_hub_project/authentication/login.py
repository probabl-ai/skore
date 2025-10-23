"""Login to ``skore hub``."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from time import sleep
from webbrowser import open as open_webbrowser

from httpx import HTTPError, HTTPStatusError, TimeoutException
from rich.align import Align
from rich.panel import Panel

from .. import console
from .logout import logout
from .token import TokenError, access
from .token import persist as persist_token
from .uri import URI, URIError
from .uri import persist as persist_URI


def get_oauth_device_login(success_uri: str | None = None):
    """
    Initiate device OAuth flow.

    Initiates the OAuth device flow.
    Provides the user with a URL and a OTP code to authenticate the device.

    Parameters
    ----------
    success_uri : str, optional
        The URI to redirect to after successful authentication.
        If not provided, defaults to None.

    Returns
    -------
    tuple
        A tuple containing:
        - authorization_url: str
            The URL to which the user needs to navigate
        - device_code: str
            The device code used for authentication
        - user_code: str
            The user code that needs to be entered on the authorization page
    """
    from skore_hub_project.client.client import HUBClient

    url = "identity/oauth/device/login"
    params = {"success_uri": success_uri} if success_uri is not None else {}

    with HUBClient(authenticated=False) as client:
        response = client.get(url, params=params).json()

        return (
            response["authorization_url"],
            response["device_code"],
            response["user_code"],
        )


def get_oauth_device_code_probe(device_code: str, *, timeout=600):
    """
    Ensure authorization code is acknowledged.

    Start polling, wait for the authorization code to be acknowledged by the hub.
    This is mandatory to be authorize to exchange with a token.

    Parameters
    ----------
    device_code : str
        The device code to exchange for tokens.
    """
    from skore_hub_project.client.client import HUBClient

    url = "identity/oauth/device/code-probe"
    params = {"device_code": device_code}

    with HUBClient(authenticated=False) as client:
        start = datetime.now()

        while True:
            try:
                client.get(url, params=params)
            except HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise

                if (datetime.now() - start).total_seconds() >= timeout:
                    raise TimeoutException("Authentication timeout") from exc

                sleep(0.5)
            else:
                break


def post_oauth_device_callback(state: str, user_code: str):
    """
    Validate the user-provided device code.

    This endpoint verifies the code entered by the user during the device auth flow.

    Parameters
    ----------
    state: str
        The unique value identifying the device flow.
    user_code: str
        The code entered by the user.
    """
    from skore_hub_project.client.client import HUBClient

    url = "identity/oauth/device/callback"
    data = {"state": state, "user_code": user_code}

    with HUBClient(authenticated=False) as client:
        client.post(url, data=data)


def get_oauth_device_token(device_code: str):
    """
    Exchanges the device code for an access token.

    This endpoint completes the device authorization flow
    by exchanging the validated device code for an access token.

    Parameters
    ----------
    device_code : str
        The device code to exchange for tokens

    Returns
    -------
    tuple
        A tuple containing:
        - access_token : str
            The OAuth access token
        - refresh_token : str
            The OAuth refresh token
        - expires_at : str
            The expiration datetime as ISO 8601 str of the access token
    """
    from skore_hub_project.client.client import HUBClient

    url = "identity/oauth/device/token"
    params = {"device_code": device_code}

    with HUBClient(authenticated=False) as client:
        response = client.get(url, params=params).json()
        tokens = response["token"]

        return (
            tokens["access_token"],
            tokens["refresh_token"],
            tokens["expires_at"],
        )


def login(*, timeout=600):
    """Login to ``skore hub``."""
    with suppress(URIError, TokenError, HTTPError):
        # Avoid re-login when
        # - the persisted URI is not conflicting with the envar
        # - the persisted token is valid or can be refreshed
        URI()
        access(refresh=True)

        console.print(
            Panel(
                Align(
                    "Already logged in!",
                    align="center",
                ),
                title="[cyan]Skore Hub[/cyan]",
                border_style="orange1",
                expand=False,
                padding=1,
                title_align="center",
            )
        )

        return

    logout()

    url, device_code, user_code = get_oauth_device_login()

    console.rule("[cyan]Skore Hub[/cyan]")
    console.print(
        f"Opening browser; if this fails, please visit this URL to log in:\n{url}",
        soft_wrap=True,
    )

    open_webbrowser(url)

    get_oauth_device_code_probe(device_code, timeout=timeout)
    post_oauth_device_callback(device_code, user_code)

    persist_token(*get_oauth_device_token(device_code))
    persist_URI(URI())
