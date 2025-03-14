"""Override httpx client to pass our bearer token."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from urllib.parse import urljoin

from httpx import URL, Client, Response

from ..authentication.token import Token
from .api import URI


class AuthenticationError(Exception):
    """An exception dedicated to authentication."""


class AuthenticatedClient(Client):
    """Override httpx client to pass our bearer token."""

    def __init__(self, *, raises=False):
        super().__init__()

        self.raises = raises

    @cached_property
    def token(self):
        """Access token."""
        return Token()

    def request(self, method: str, url: URL | str, **kwargs) -> Response:
        """Execute request with access token, and refresh the token if needed."""
        # Check if token is well initialized
        if not self.token.valid:
            raise AuthenticationError("You are not logged in.")

        # Check if token must be refreshed
        if self.token.expires_at <= datetime.now(timezone.utc):
            self.token.refresh()

        # Overload headers with authorization token
        headers = kwargs.pop("headers", None) or {}
        headers["Authorization"] = f"Bearer {self.token.access}"
        response = super().request(
            method,
            urljoin(URI, str(url)),
            headers=headers,
            **kwargs,
        )

        if self.raises:
            response.raise_for_status()

        return response
