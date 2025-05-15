"""API used to exchange with ``skore hub``."""

from functools import partial
from os import environ
from typing import Optional
from urllib.parse import urljoin

import httpx

URI = environ.get("SKORE_HUB_URI", "https://api.skore.probabl.ai")
Client = partial(
    httpx.Client,
    follow_redirects=True,
    event_hooks={"response": [httpx.Response.raise_for_status]},
)


def get_oauth_device_login(success_uri: Optional[str] = None):
    """Initiate device OAuth flow.

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
    url = urljoin(URI, "identity/oauth/device/login")
    params = {"success_uri": success_uri} if success_uri is not None else {}

    with Client() as client:
        response = client.get(url, params=params).json()

        return (
            response["authorization_url"],
            response["device_code"],
            response["user_code"],
        )


def post_oauth_device_callback(state: str, user_code: str):
    """Validate the user-provided device code.

    This endpoint verifies the code entered by the user during the device auth flow.

    Parameters
    ----------
    state: str
        The unique value identifying the device flow.
    user_code: str
        The code entered by the user.
    """
    url = urljoin(URI, "identity/oauth/device/callback")

    with Client() as client:
        client.post(url, data={"state": state, "user_code": user_code})


def get_oauth_device_token(device_code: str):
    """Exchanges the device code for an access token.

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
    url = urljoin(URI, "identity/oauth/device/token")

    with Client() as client:
        response = client.get(url, params={"device_code": device_code}).json()
        tokens = response["token"]

        return (
            tokens["access_token"],
            tokens["refresh_token"],
            tokens["expires_at"],
        )


def post_oauth_refresh_token(refresh_token: str):
    """Refresh an access token using a provided refresh token.

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
    url = urljoin(URI, "identity/oauth/token/refresh")

    with Client() as client:
        response = client.post(url, json={"refresh_token": refresh_token}).json()

        return (
            response["access_token"],
            response["refresh_token"],
            response["expires_at"],
        )
