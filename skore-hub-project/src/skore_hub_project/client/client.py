"""Client exchanging with ``skore hub``."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from os import environ
from urllib.parse import urljoin

from httpx import URL, Client, Headers, HTTPStatusError, Response
from httpx._types import HeaderTypes

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

        if (apikey := environ.get("SKORE_HUB_API_KEY")) is not None:
            self.__authorization = f"X-API-Key: {apikey}"

    @property
    def __authorization(self):
        token = Token()

        if not token.valid:
            raise AuthenticationError(
                "You are not logged in. Please run `skore-hub-login`"
            )

        if token.expires_at <= datetime.now(timezone.utc):
            token.refresh()

        return f"Bearer {token.access_token}"

    def request(
        self,
        method: str,
        url: URL | str,
        headers: HeaderTypes | None = None,
        **kwargs,
    ) -> Response:
        """Execute request with access token, and refresh the token if needed."""
        headers = Headers(headers)

        # Overload headers with our own authorization token
        headers.update({"Authorization": self.__authorization})

        # Send request
        response = super().request(
            method=method,
            url=urljoin(URI, str(url)),
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
