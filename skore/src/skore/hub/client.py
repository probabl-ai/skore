"""Override httpx client to pass our bearer token."""

from __future__ import annotations

from datetime import datetime
from functools import cached_property
from urllib.parse import urljoin

from httpx import URL, Client, Response

from skore.hub.api import URI
from skore.hub.token import AuthenticationToken


class AuthenticationError(Exception):
    """An exception dedicated to authentication."""


class AuthenticatedClient(Client):
    """Override httpx client to pass our bearer token."""

    @cached_property
    def token(self):
        """Access token."""
        return AuthenticationToken()

    def request(self, method: str, url: URL | str, **kwargs) -> Response:
        """Execute request with access token, and refresh the token if needed."""
        # Check if token is well initialized
        if not self.token.is_valid():
            raise AuthenticationError("You are not logged in.")

        # Check if token must be refreshed
        if self.token.expires_at <= datetime.now():
            self.token.refresh()

        # Overload headers with authorization token
        headers = kwargs.pop("headers", None) or {}
        headers |= {"Authorization": f"Bearer {self.token.access}"}

        return super().request(
            method,
            urljoin(URI, str(url)),
            headers=headers,
            **kwargs,
        )
