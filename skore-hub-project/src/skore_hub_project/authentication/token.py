"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

import json
import pathlib
import tempfile
from datetime import datetime, timezone

from ..client.api import post_oauth_refresh_token


def filepath():
    """Filepath used to save the tokens on disk."""
    return pathlib.Path(tempfile.gettempdir(), "skore.token")


def save(access: str, refreshment: str, expires_at: str):
    """
    Save the tokens to the disk to prevent user to login more than once, as long as
    the token is valid or can be refreshed.
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
    access, refreshment, expiration = json.loads(filepath().read_text())

    if refresh and datetime.fromisoformat(expiration) <= datetime.now(timezone.utc):
        # Retrieve freshly updated tokens
        access, refreshment, expiration = post_oauth_refresh_token(refreshment)

        # Re-save the refreshed tokens
        save(access, refreshment, expiration)

    return access
