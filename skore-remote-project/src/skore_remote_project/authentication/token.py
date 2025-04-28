"""Token used for ``skore hub`` authentication."""

from __future__ import annotations

import functools
import json
import pathlib
import tempfile
from datetime import datetime
from typing import Optional

from ..client.api import post_oauth_refresh_token


class Token:
    """Wrap access, refresh and expires_at."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        expires_at: Optional[str] = None,
    ):
        if (
            access_token is not None
            and refresh_token is not None
            and expires_at is not None
        ):
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.expires_at = expires_at

            # Save the tokens to the disk to prevent user to login more than once, as
            # long as the token is valid or can be refreshed.
            self.filepath.write_text(
                json.dumps(
                    (
                        access_token,
                        refresh_token,
                        expires_at,
                    )
                )
            )
        else:
            try:
                self.access_token, self.refresh_token, self.expires_at = json.loads(
                    self.filepath.read_text()
                )
            except FileNotFoundError:
                self.valid = False
                return

        # Flag used to ensure the validity of the token when requesting an URL with
        # httpx, in a lazy way.
        self.valid = True

    @property
    def expires_at(self):
        """Datetime of expiration of the tokens."""
        return self.__expires_at

    @expires_at.setter
    def expires_at(self, value: str):
        self.__expires_at = datetime.fromisoformat(value)

    @functools.cached_property
    def filepath(self):
        """Filepath used to save the tokens on disk."""
        return pathlib.Path(tempfile.gettempdir(), "skore.token")

    def refresh(self):
        """Call the API to get a fresh access token."""
        self.__init__(*post_oauth_refresh_token(self.refresh_token))

    def __repr__(self):
        """Repr implementation."""
        return f"Token('{self.access_token:.10}[...]')"
