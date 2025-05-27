"""Client exchanging with ``skore hub``."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from functools import cached_property
from urllib.parse import urljoin

from httpx import URL, Client, HTTPStatusError, Response

from ..authentication.token import Token
from .api import URI


class AuthenticationError(Exception):
    """An exception dedicated to authentication."""


class AuthenticatedClient(Client):
    """Client exchanging with ``skore hub``."""

    ERROR_TYPES = {
        1: "Informational response",
        3: "Redirect response",
        4: "Client error",
        5: "Server error",
    }

    def __init__(self, *, raises=False):
        super().__init__(follow_redirects=True, timeout=300)

        self.raises = raises

    @cached_property
    def token(self):
        """Access token."""
        return Token()

    def request(self, method: str, url: URL | str, **kwargs) -> Response:
        """Execute request with access token, and refresh the token if needed."""
        # Check if token is well initialized
        if not self.token.valid:
            raise AuthenticationError(
                "You are not logged in. Please run `skore-hub-login`"
            )

        # Check if token must be refreshed
        if self.token.expires_at <= datetime.now(timezone.utc):
            self.token.refresh()

        # Overload headers with authorization token
        headers = kwargs.pop("headers", None) or {}
        headers["Authorization"] = f"Bearer {self.token.access_token}"
        response = super().request(
            method,
            urljoin(URI, str(url)),
            headers=headers,
            **kwargs,
        )

        if self.raises and not response.is_success:
            status_class = response.status_code // 100
            error_type = self.ERROR_TYPES.get(status_class, "Invalid status code")
            message = (
                f"{error_type} '{response.status_code} {response.reason_phrase}' "
                f"for url '{response.url}'"
            )

            with suppress(Exception):
                message += f": {response.json()['message']}"

            raise HTTPStatusError(message, request=response.request, response=response)

        return response
