"""Module to manage ``host``."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from functools import cache, wraps
from os import environ
from typing import TYPE_CHECKING, ParamSpec, TypeVar
from urllib.parse import urlsplit, urlunsplit

from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME
from skore._plugins.hub.client.client import Client

if TYPE_CHECKING:
    P = ParamSpec("P")
    R = TypeVar("R")


@cache
def resolve(uri: str) -> str:
    """Resolve a user-facing Skore Hub URI to its API URI when available."""
    discovery_url = uri.rstrip("/") + "/.well-known/skore-hub.json"

    with suppress(Exception), Client(retry=False) as client:
        response = client.get(discovery_url, timeout=5)

        if api_url := response.json().get("api_url"):
            return str(api_url)

    return uri


@cache
def normalize(uri: str) -> str:
    """Return ``uri`` with a stable scheme, host and path."""
    parts = urlsplit(uri.strip())

    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
    )


def ensure_host_is_valid(method: Callable[P, R]) -> Callable[P, R]:
    """Ensure host is valid before executing any other operation."""

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        host = kwargs.pop("host", None) or environ.get(HOST_ENV_VAR_NAME)

        if host is None:
            raise ValueError(
                "You must provide an host via the parameter `host` or the environment "
                f"variable `{HOST_ENV_VAR_NAME}`"
            )

        if not isinstance(host, str):
            raise ValueError("`host` must be a str.")

        kwargs["host"] = normalize(resolve(host))

        return method(*args, **kwargs)

    return wrapper
