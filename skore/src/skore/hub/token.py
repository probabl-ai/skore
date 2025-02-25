"""Encapsulate all authentication related stuff."""

from __future__ import annotations

import functools
import json
import pathlib
import tempfile
from datetime import datetime
from typing import Optional

from skore.hub.api import post_oauth_refresh_token


class AuthenticationToken:
    """Wrap access, refresh and expires_at."""

    def __init__(
        self,
        access: Optional[str] = None,
        refreshment: Optional[str] = None,
        expires_at: Optional[str] = None,
    ):
        if all(attr is not None for attr in (access, refreshment, expires_at)):
            self.access = access
            self.refreshment = refreshment
            self.expires_at = datetime.fromisoformat(expires_at)

            self.save()
        else:
            try:
                content = json.loads(self.filepath.read_text())

                self.access, self.refreshment = content[:2]
                self.expires_at = datetime.fromisoformat(content[2])
            except FileNotFoundError:
                self.access = None
                self.refreshment = None
                self.expires_at = None

    @functools.cached_property
    def filepath(self):
        """Filepath used to save the tokens on disk."""
        return pathlib.Path(tempfile.gettempdir(), "skore.token")

    def save(self):
        """Save the tokens to the disk."""
        self.filepath.write_text(
            json.dumps(
                (
                    self.access,
                    self.refreshment,
                    self.expires_at.isoformat(),
                )
            )
        )

    def refresh(self):
        """Call the API to get a fresh access token."""
        content = post_oauth_refresh_token(self.refreshment)

        self.access, self.refreshment = content[:2]
        self.expires_at = datetime.fromisoformat(content[2])
        self.save()

    def is_valid(self):
        """Check that all data are present."""
        return (
            self.access is not None
            and self.refresh is not None
            and self.expires_at is not None
        )

    def __repr__(self):
        """Repr implementation."""
        return f"AuthenticationToken('{self.access:.10}[...]')"
