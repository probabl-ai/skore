"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

import json
import pathlib
import tempfile
from datetime import datetime, timezone

from ..client.api import post_oauth_refresh_token


class Token:
    """
    Wrap access, refresh and expires_at.

    Notes
    -----
    The token is not persisted in RAM to allow user to logout from ``skore hub`` by
    removing the ``token.filepath``.
    """

    @staticmethod
    def filepath():
        """Filepath used to save the tokens on disk."""
        return pathlib.Path(tempfile.gettempdir(), "skore.token")

    @staticmethod
    def save(access_token: str, refresh_token: str, expires_at: str):
        """
        Save the tokens to the disk to prevent user to login more than once, as long as
        the token is valid or can be refreshed.
        """
        filepath = Token.filepath()
        filepath.write_text(
            json.dumps(
                (
                    access_token,
                    refresh_token,
                    expires_at,
                )
            )
        )

    @staticmethod
    def exists() -> bool:
        """Existence of the token."""
        return Token.filepath().exists()

    @staticmethod
    def access() -> str:
        """
        Access used to communicate with the ``skore hub`` API.

        Notes
        -----
        The token is automatically refreshed on purpose.
        """
        access, refresh, expiration = json.loads(Token.filepath().read_text())

        if datetime.fromisoformat(expiration) <= datetime.now(timezone.utc):
            # Retrieve freshly updated tokens
            access, refresh, expiration = post_oauth_refresh_token(refresh)

            # Re-save the refreshed tokens
            Token.save(access, refresh, expiration)

        return access
