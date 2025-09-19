"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

import json
import pathlib
import tempfile
from datetime import datetime, timezone


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
    from skore_hub_project.client.client import HUBClient

    url = "identity/oauth/token/refresh"
    json = {"refresh_token": refresh_token}

    with HUBClient(authenticated=False) as client:
        response = client.post(url, json=json).json()

        return (
            response["access_token"],
            response["refresh_token"],
            response["expires_at"],
        )


def filepath():
    """Filepath used to save the tokens on disk."""
    return pathlib.Path(tempfile.gettempdir(), "skore.token")


def save(access: str, refreshment: str, expires_at: str):
    """
    Save the token.

    Save the tokens to the disk to prevent user to login more than once, as long as
    the access token is valid or can be refreshed.
    """
    filepath().write_text(
        json.dumps(
            (
                access,
                refreshment,
                expires_at,
            )
        )
    )


def exists() -> bool:
    """Existence of the token."""
    return filepath().exists()


def access(*, refresh=True) -> str:
    """
    Access token.

    Parameters
    ----------
    refresh : bool, optional
        Refresh the token on-the-fly if necessary, default True.
    """
    access, refreshment, expiration = json.loads(filepath().read_text())

    if refresh and datetime.fromisoformat(expiration) <= datetime.now(timezone.utc):
        # Retrieve freshly updated tokens
        access, refreshment, expiration = post_oauth_refresh_token(refreshment)

        # Re-save the refreshed tokens
        save(access, refreshment, expiration)

    return access
