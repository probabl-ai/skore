"""Collection of function dedicated to communicate with skore-hub."""

import os
from typing import Optional
from urllib.parse import urljoin

import httpx

URI = os.environ.get("SKORE_HUB_URI", "https://skh.k.probabl.dev")


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
    params = {"success_uri": success_uri} if success_uri is not None else {}

    with httpx.Client() as client:
        response = client.get(
            urljoin(URI, "identity/oauth/device/login"), params=params
        )

        response.raise_for_status()
        data = response.json()

        return (
            data.get("authorization_url"),
            data.get("device_code"),
            data.get("user_code"),
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
    with httpx.Client() as client:
        response = client.post(
            urljoin(URI, "identity/oauth/device/callback"),
            data={
                "state": state,
                "user_code": user_code,
            },
        )

        response.raise_for_status()


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
    with httpx.Client() as client:
        response = client.get(
            urljoin(URI, "identity/oauth/device/token"),
            params={
                "device_code": device_code,
            },
        )

        response.raise_for_status()
        tokens = response.json().get("token")

        return (
            tokens.get("access_token"),
            tokens.get("refresh_token"),
            tokens.get("expires_at"),
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
    with httpx.Client() as client:
        response = client.post(
            urljoin(URI, "identity/oauth/token/refresh"),
            json={"refresh_token": refresh_token},
        )

        response.raise_for_status()
        tokens = response.json()

        return (
            tokens.get("access_token"),
            tokens.get("refresh_token"),
            tokens.get("expires_at"),
        )
