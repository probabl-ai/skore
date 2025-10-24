"""Token used for ``skore hub`` authentication."""

from datetime import datetime, timezone
from json import dumps, loads
from pathlib import Path
from tempfile import gettempdir


class TokenError(Exception):
    """An exception dedicated to ``Token``."""


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


def Filepath() -> Path:
    """Filepath used to persist the token on disk."""
    return Path(gettempdir(), "skore.token")


def persist(access: str, refreshment: str, expiration: str) -> None:
    """
    Persist the token.

    Persist the token on disk to prevent user to login more than once, as long as
    the access token is valid or can be refreshed.
    """
    Filepath().write_text(
        dumps(
            [
                access,
                refreshment,
                expiration,
            ]
        )
    )


def access(*, refresh: bool = True) -> str:
    """
    Access token.

    Parameters
    ----------
    refresh : bool, optional
        Refresh the token on-the-fly if necessary, default True.
    """
    if not Filepath().exists():
        raise TokenError("You are not logged in. Please run `skore-hub-login`.")

    access: str
    refreshment: str
    expiration: str

    access, refreshment, expiration = loads(Filepath().read_text())

    if refresh and datetime.fromisoformat(expiration) <= datetime.now(timezone.utc):
        # Retrieve freshly updated tokens
        access, refreshment, expiration = post_oauth_refresh_token(refreshment)

        # Re-save the refreshed tokens
        persist(access, refreshment, expiration)

    return access
