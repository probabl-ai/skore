"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from time import sleep
from typing import Iterator, Literal
from urllib.parse import urljoin
from webbrowser import open as open_webbrowser

from httpx import HTTPStatusError, TimeoutException

from skore_hub_project import console
from skore_hub_project.authentication.uri import URI


def get_oauth_device_login(success_uri: str | None = None) -> tuple[str, str, str]:
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
    from skore_hub_project.client.client import Client

    url = "identity/oauth/device/login"
    params = {"success_uri": success_uri} if success_uri is not None else {}

    with Client() as client:
        response = client.get(urljoin(URI, url), params=params).json()

        return (
            response["authorization_url"],
            response["device_code"],
            response["user_code"],
        )


def get_oauth_device_code_probe(device_code: str, *, timeout: int = 600) -> None:
    """
    Ensure authorization code is acknowledged.

    Start polling, wait for the authorization code to be acknowledged by the hub.
    This is mandatory to be authorize to exchange with a token.

    Parameters
    ----------
    device_code : str
        The device code to exchange for tokens.
    """
    from skore_hub_project.client.client import Client

    url = "identity/oauth/device/code-probe"
    params = {"device_code": device_code}

    with Client() as client:
        start = datetime.now()

        while True:
            try:
                client.get(urljoin(URI, url), params=params)
            except HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise

                if (datetime.now() - start).total_seconds() >= timeout:
                    raise TimeoutException("Authentication timeout") from exc

                sleep(0.5)
            else:
                break


def post_oauth_device_callback(state: str, user_code: str) -> None:
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
    from skore_hub_project.client.client import Client

    url = "identity/oauth/device/callback"
    data = {"state": state, "user_code": user_code}

    with Client() as client:
        client.post(urljoin(URI, url), data=data)


def get_oauth_device_token(device_code: str) -> tuple[str, str, str]:
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
    from skore_hub_project.client.client import Client

    url = "identity/oauth/device/token"
    params = {"device_code": device_code}

    with Client() as client:
        response = client.get(urljoin(URI, url), params=params).json()
        tokens = response["token"]

        return (
            tokens["access_token"],
            tokens["refresh_token"],
            tokens["expires_at"],
        )


def post_oauth_refresh_token(refresh_token: str) -> tuple[str, str, str]:
    """
    Refresh an access token using a provided refresh token.

    This endpoint allows a client to obtain a new access token
    by providing a valid refresh token.

    Parameters
    ----------
    refresh_token : str
        A valid refresh token

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
    from skore_hub_project.client.client import Client

    url = "identity/oauth/token/refresh"
    json = {"refresh_token": refresh_token}

    with Client() as client:
        response = client.post(urljoin(URI, url), json=json).json()

        return (
            response["access_token"],
            response["refresh_token"],
            response["expires_at"],
        )


class Token:
    def __init__(self, *, timeout: int = 600) -> None:
        url, device_code, user_code = get_oauth_device_login()

        console.rule("[cyan]Skore Hub[/cyan]")
        console.print(
            f"Opening browser; if this fails, please visit this URL to log in:\n{url}",
            soft_wrap=True,
        )

        open_webbrowser(url)

        get_oauth_device_code_probe(device_code, timeout=timeout)
        post_oauth_device_callback(device_code, user_code)

        access, refreshment, expiration = get_oauth_device_token(device_code)

        self.__lock = RLock()
        self.__access = access
        self.__refreshment = refreshment
        self.__expiration = datetime.fromisoformat(expiration)

    def __call__(self) -> dict[str, str]:
        with self.__lock:
            if self.__expiration <= datetime.now(timezone.utc):
                access, refreshment, expiration = post_oauth_refresh_token(
                    self.__refreshment
                )

                self.__access = access
                self.__refreshment = refreshment
                self.__expiration = datetime.fromisoformat(expiration)

            return {"Authorization": f"Bearer {self.__access}"}
