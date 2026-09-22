"""Module to manage ``host``."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from os import environ
from typing import TYPE_CHECKING, ParamSpec, TypeVar
from urllib.parse import urlsplit, urlunsplit

from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME

if TYPE_CHECKING:
    P = ParamSpec("P")
    R = TypeVar("R")


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
                f"You must provide an host via the parameter `host` or the environment "
                f"variable `{HOST_ENV_VAR_NAME}`"
            )

        if not isinstance(host, str):
            raise ValueError("`host` must be a str.")

        kwargs["host"] = normalize(host)

        return method(*args, **kwargs)

    return wrapper
