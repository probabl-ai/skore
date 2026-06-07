"""API key used for ``skore hub`` authentication."""

from collections.abc import Callable
from os import environ
from typing import Final

ENV_VAR_NAME: Final[str] = "SKORE_HUB_API_KEY"


def APIKey() -> Callable[[], dict[str, str]]:
    """Get the API key used for ``skore hub`` authentication.

    In the form of HTTP header parameters.
    """
    header = {"X-API-Key": environ[ENV_VAR_NAME]}
    return lambda: header
