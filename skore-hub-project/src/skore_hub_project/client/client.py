"""Client exchanging with ``skore hub``."""

import logging
from contextlib import suppress
from http import HTTPStatus
from os import environ
from time import sleep
from typing import Any, Final
from urllib.parse import urljoin

from httpx import (
    URL,
    Headers,
    HTTPError,
    HTTPStatusError,
    NetworkError,
    RemoteProtocolError,
    Response,
    TimeoutException,
)
from httpx import Client as HTTPXClient
from httpx._types import HeaderTypes

from ..authentication import token as Token
from ..authentication.uri import URI

logger = logging.getLogger(__name__)


class Client(HTTPXClient):
    """
    Client with a retry strategy based on exponential backoff algorithm.

    Parameters
    ----------
    retry : bool, optional
        Retry request on fails, default True.
    retry_total: int | None, optional
        Total number of retries to allow, default 10.
        Set to None to remove this constraint.
    retry_backoff_factor : float, optional
        A backoff factor to apply between retries, default 0.25.
    retry_backoff_max : float, optional
        Maximum backoff time, default 120.
    """

    RETRYABLE_STATUS_CODES: Final[frozenset[HTTPStatus]] = frozenset(
        (
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.TOO_EARLY,
            HTTPStatus.TOO_MANY_REQUESTS,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        )
    )

    RETRYABLE_EXCEPTIONS: Final[tuple[type[HTTPError], ...]] = (
        TimeoutException,
        NetworkError,
        RemoteProtocolError,
    )

    STATUS_CLASS_TO_ERROR_TYPE: Final[dict[int, str]] = {
        1: "Informational response",
        3: "Redirect response",
        4: "Client error",
        5: "Server error",
    }

    def __init__(
        self,
        *,
        retry: bool = True,
        retry_total: int | None = 10,
        retry_backoff_factor: float = 0.25,
        retry_backoff_max: float = 120,
    ):
        super().__init__(follow_redirects=True, timeout=30)

        self.retry = retry
        self.retry_total = retry_total if retry_total is not None else float("inf")
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_backoff_max = retry_backoff_max

    def request(self, *args: Any, **kwargs: Any) -> Response:
        """Execute request with retry strategy."""
        retries = 0

        while True:
            try:
                response = super().request(*args, **kwargs)
            except self.RETRYABLE_EXCEPTIONS:
                if not self.retry:
                    raise
            else:
                if response.is_success:
                    return response

                overlimit = retries >= self.retry_total
                unretryable = response.status_code not in self.RETRYABLE_STATUS_CODES

                if (not self.retry) or overlimit or unretryable:
                    break

            timeout = self.retry_backoff_factor * (2**retries)
            retries += 1

            sleep_duration = min(timeout, self.retry_backoff_max)

            logger.warning(
                "Request failed to reach server. "
                f"Trying again in {sleep_duration} seconds."
            )

            sleep(sleep_duration)

        # Raise extended exception with body message when available.
        status_class = response.status_code // 100
        error_type = self.STATUS_CLASS_TO_ERROR_TYPE.get(
            status_class, "Invalid status code"
        )

        message = (
            f"{error_type} '{response.status_code} {response.reason_phrase}' "
            f"for url '{response.url}'"
        )

        with suppress(Exception):
            message += f": {response.json()['message']}"

        raise HTTPStatusError(message, request=response.request, response=response)


class HUBClient(Client):
    """
    Client exchanging with ``skore hub``.

    Parameters
    ----------
    authenticated : bool, optional
        Use headers with API key or token, default True.
    """

    def __init__(self, *, authenticated: bool = True, **kwargs: Any):
        super().__init__(**kwargs)

        self.authenticated = authenticated

    def request(
        self,
        method: str,
        url: URL | str,
        headers: HeaderTypes | None = None,
        **kwargs: Any,
    ) -> Response:
        """Execute request with authorization."""
        headers = Headers(headers)

        # Overload headers with our custom headers (API key or token)
        if self.authenticated:
            if (apikey := environ.get("SKORE_HUB_API_KEY")) is not None:
                headers.update({"X-API-Key": f"{apikey}"})
            else:
                headers.update({"Authorization": f"Bearer {Token.access()}"})

        # Prefix the request by the hub URI when ``url`` is not absolute
        url = urljoin(URI(), str(url))

        return super().request(method=method, url=url, headers=headers, **kwargs)
