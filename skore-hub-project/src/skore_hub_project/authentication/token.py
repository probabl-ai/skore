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
    def save(access: str, refreshment: str, expires_at: str):
        """
        Save the tokens to the disk to prevent user to login more than once, as long as
        the token is valid or can be refreshed.
        """
        Token.filepath().write_text(
            json.dumps(
                (
                    access,
                    refreshment,
                    expires_at,
                )
            )
        )

    @staticmethod
    def exists() -> bool:
        """Existence of the token."""
        return Token.filepath().exists()

    @staticmethod
    def access(*, refresh=True) -> str:
        access, refreshment, expiration = json.loads(Token.filepath().read_text())

        if refresh and datetime.fromisoformat(expiration) <= datetime.now(timezone.utc):
            # Retrieve freshly updated tokens
            access, refreshment, expiration = post_oauth_refresh_token(refreshment)

            # Re-save the refreshed tokens
            Token.save(access, refreshment, expiration)

        return access
